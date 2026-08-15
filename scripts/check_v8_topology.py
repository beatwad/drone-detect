"""Check that a YOLOv8 model yaml reproduces the compiled reference graph.

The ZCU102 bitstream was built from the live subgraph of
`quantyolov8_4w4a_comact_tidy.onnx`. Anything we train for that build must have
the SAME convolution topology, or the FINN recipe in build_notes §10 stops
applying. This compares every live Conv weight shape in the reference against
every Conv weight shape in the PyTorch model.

The comparison is on the MULTISET of shapes, not the sequence: `named_modules()`
walks C2f in declaration order (cv1, cv2, m.0...) while ONNX is topological
(cv1, m.0.cv1, m.0.cv2, cv2), and the exporter interleaves Detect's independent
cv2/cv3 branches. Order therefore differs for reasons that carry no meaning;
shape counts do not.

The reference graph ends at the raw [1, 4*reg_max + nc, H, W] head output — DFL,
decode and NMS run off-chip (see simple_yolov8_driver.py) — so `Detect.dfl`, a
fixed non-trainable 1x1, is excluded.

    uv run python scripts/check_v8_topology.py \
        --cfg configs/yolov8n_p3_relu.yaml \
        --onnx /home/alex/finn_build_mdanilow/yolov8/quantyolov8_4w4a_comact_tidy.onnx

With --nc 1 the head narrows (Detect's c3 = max(ch[0], min(nc, 100)) is 64 for a
single class, 80 for COCO's), so the last three convs legitimately differ; they
are reported separately rather than counted as failures.
"""

import argparse
from collections import Counter

import onnx
import torch.nn as nn
from onnx import numpy_helper


def live_conv_shapes(onnx_path):
    """Conv weight shapes reachable from the graph outputs, in topological order."""
    model = onnx.load(onnx_path)
    graph = model.graph
    init = {i.name: i for i in graph.initializer}

    producer = {o: n for n in graph.node for o in n.output}
    live, stack = set(), [o.name for o in graph.output]
    while stack:
        node = producer.get(stack.pop())
        if node is None or id(node) in live:
            continue
        live.add(id(node))
        stack.extend(node.input)

    out = []
    for node in graph.node:                       # ONNX node order is topological
        if node.op_type == 'Conv' and id(node) in live:
            w = init.get(node.input[1])
            out.append((node.name, tuple(numpy_helper.to_array(w).shape)))
    return out


def model_conv_shapes(cfg, nc):
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(cfg, nc=nc, verbose=False)
    return [(n, tuple(m.weight.shape)) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and not n.endswith('dfl.conv')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--nc', type=int, default=80)
    args = ap.parse_args()

    ref = live_conv_shapes(args.onnx)
    got = model_conv_shapes(args.cfg, args.nc)

    ref_c = Counter(s for _, s in ref)
    got_c = Counter(s for _, s in got)

    print(f'reference live convs : {len(ref)}')
    print(f'model convs          : {len(got)}   (cfg={args.cfg}, nc={args.nc})')
    print()

    if ref_c == got_c:
        print('MATCH: identical multiset of %d conv weight shapes.' % len(ref))
        return 0

    only_ref = ref_c - got_c
    only_got = got_c - ref_c
    print('%-22s %-10s %-10s' % ('shape', 'reference', 'model'))
    for shape in sorted(set(only_ref) | set(only_got), key=str):
        print('%-22s %-10d %-10d' % (str(shape), ref_c.get(shape, 0), got_c.get(shape, 0)))
    print()
    print(f'{sum(only_ref.values())} conv(s) only in the reference, '
          f'{sum(only_got.values())} only in the model')
    if args.nc != 80:
        print('(with nc != 80 the head narrows: Detect c3 = max(ch[0], min(nc, 100)) '
              'is 64 for one class vs 80 for COCO, so the three cv3 convs differ by design)')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
