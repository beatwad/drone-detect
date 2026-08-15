"""Export the QAT YOLOv8n-P3 to QONNX, and diff it against the reference graph.

The reference bitstream's graph ends at the RAW head tensor,
`[1, 4*reg_max + nc, H, W]` — DFL, decode and NMS run on the ARM side
(`simple_yolov8_driver.py`). `RawV8Head` reproduces that: the cv2/cv3 branches and
the concat that joins them, nothing after.

The `--reference` diff is the cheap, decisive gate before committing ~30 h to a
build: if our graph carries the same op histogram and the same quantized
datatypes as `quantyolov8_4w4a_comact_tidy.onnx`, the FINN recipe in
build_notes §10 applies unchanged.

    uv run python -m export.export_qonnx_v8 \
        --ckpt runs/qat/v8n_p3_w4a4/weights/best.pt --imgsz 192,320 \
        --reference /home/alex/finn_build_mdanilow/yolov8/quantyolov8_4w4a_comact_tidy.onnx
"""
import argparse
import sys
import types
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qat.train_qat_v8 import load_qat_checkpoint  # noqa: E402

# Ops FINN's fpgadataflow backend has a hardware node for, after streamlining.
FINN_OK = {
    'Conv', 'MatMul', 'Gemm', 'MaxPool', 'AveragePool', 'Relu', 'BatchNormalization',
    'Mul', 'Div', 'Sub', 'Reshape', 'Transpose', 'Flatten', 'Identity', 'Pad',
    'Quant', 'BipolarQuant', 'Trunc', 'Resize', 'Upsample', 'Split', 'Clip',
}
FINN_JOIN = {'Add', 'Concat'}


class RawV8Head(nn.Module):
    """Detect reduced to cv2/cv3 + the box|cls concat; DFL and decode dropped."""

    def __init__(self, detect):
        super().__init__()
        self.cv2, self.cv3 = detect.cv2, detect.cv3
        self.f, self.i = detect.f, detect.i
        self.type = 'RawV8Head'
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        out = [torch.cat((c2(xi), c3(xi)), 1)
               for c2, c3, xi in zip(self.cv2, self.cv3, x)]
        return out[0] if len(out) == 1 else out


def use_split_chunks(model):
    """Make every C2f trace to a `Split` node instead of Shape/Gather/Slice.

    `C2f.forward` uses `Tensor.chunk`, which the ONNX exporter unrolls into
    Shape -> Gather -> Div/Mul -> Slice (verified on opset 11 and 13), leaving 12
    Slice nodes FINN has no hardware op for. `torch.split` traces to a single
    Split, which is exactly what the reference graph carries (6 of them). The two
    are numerically identical, so this is a tracing concern only — hence patched
    here at export time and not in the training path.

    Ultralytics ships `C2f.forward_split` for this; rebinding it per instance
    keeps the module tree and every state_dict key untouched.
    """
    from ultralytics.nn.modules.block import C2f
    n = 0
    for m in model.modules():
        if isinstance(m, C2f):
            m.forward = types.MethodType(C2f.forward_split, m)
            n += 1
    return n


def live_nodes(graph):
    """Nodes reachable backwards from the graph outputs."""
    prod = {o: n for n in graph.node for o in n.output}
    live, stack = set(), [o.name for o in graph.output]
    while stack:
        n = prod.get(stack.pop())
        if n is None or id(n) in live:
            continue
        live.add(id(n))
        stack.extend(n.input)
    return [n for n in graph.node if id(n) in live]


def census(onnx_path):
    import onnx
    c = Counter(n.op_type for n in live_nodes(onnx.load(str(onnx_path)).graph))
    ok = {k: v for k, v in c.items() if k in FINN_OK}
    join = {k: v for k, v in c.items() if k in FINN_JOIN}
    unk = {k: v for k, v in c.items() if k not in FINN_OK | FINN_JOIN}
    return c, ok, join, unk


def quant_profile(onnx_path):
    """(op histogram, Quant bit-width/signedness histogram) over the live graph.

    Read from the Quant nodes' `bit_width` inputs rather than from tensor
    annotations, so it works on a freshly exported Brevitas graph as well as on
    the reference's already-streamlined one.
    """
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(onnx_path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    nodes = live_nodes(g)
    ops = Counter(n.op_type for n in nodes)
    bits = Counter()
    for n in nodes:
        if n.op_type != 'Quant':
            continue
        bw = init.get(n.input[3])
        signed = next((a.i for a in n.attribute if a.name == 'signed'), None)
        if bw is not None:
            bits[(int(bw), 'signed' if signed else 'unsigned')] += 1
    return ops, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/qat/v8n_p3_w4a4/weights/best.pt')
    ap.add_argument('--imgsz', default='192,320', help="'H,W' or a square size")
    ap.add_argument('--out', default=None)
    ap.add_argument('--reference', default=None,
                    help='reference QONNX to diff against')
    args = ap.parse_args()

    h, w = ((int(v) for v in args.imgsz.split(',')) if ',' in args.imgsz
            else (int(args.imgsz), int(args.imgsz)))

    model, ck = load_qat_checkpoint(args.ckpt, 'cpu')
    model.model[-1] = RawV8Head(model.model[-1])
    n_split = use_split_chunks(model)
    model.eval()

    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        y = model(x)
    shape = tuple((y[0] if isinstance(y, list) else y).shape)

    out = Path(args.out or ROOT / 'export' /
               f"v8n_p3_w{ck['low_bits']}a{ck['low_bits']}_{h}x{w}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    from brevitas.export import export_qonnx
    export_qonnx(model, args=x, export_path=str(out))

    print(f"checkpoint : {args.ckpt} (epoch {ck.get('epoch')}, "
          f"W{ck['low_bits']} body / W{ck['high_bits']} stem+head)")
    print(f'input      : (1, 3, {h}, {w})')
    print(f'output     : {shape}')
    print(f'written    : {out}  ({out.stat().st_size / 1e6:.2f} MB)')
    print(f'C2f -> split : {n_split}')

    # Constant-fold and tidy, the same step the yolov5 path takes before FINN.
    from qonnx.util.cleanup import cleanup
    clean = out.with_name(out.stem + '_clean.onnx')
    cleanup(str(out), out_file=str(clean))
    print(f'cleaned    : {clean}')
    out = clean

    c, ok, join, unk = census(out)
    print(f'\nop census ({sum(c.values())} live nodes)')
    print(f'  FINN-supported : {dict(sorted(ok.items()))}')
    print(f'  JOINS          : {dict(sorted(join.items())) or "none"}')
    print(f'  unsupported    : {dict(sorted(unk.items())) or "none"}')

    if args.reference:
        ours, our_bits = quant_profile(out)
        theirs, their_bits = quant_profile(args.reference)
        print(f'\n--- vs reference {Path(args.reference).name} ---')
        print(f"{'op':<22} {'ours':>6} {'ref':>6}")
        for op in sorted(set(ours) | set(theirs)):
            print(f'{op:<22} {ours.get(op, 0):>6} {theirs.get(op, 0):>6}')
        print(f"\n{'Quant bit width':<22} {'ours':>6} {'ref':>6}")
        for k in sorted(set(our_bits) | set(their_bits), key=str):
            print(f'{str(k):<22} {our_bits.get(k, 0):>6} {their_bits.get(k, 0):>6}')
        print('\nNOTE the reference is already STREAMLINED (Quant folded into '
              'MultiThreshold, BN kept); ours is raw Brevitas output. Compare '
              'Conv/BN/Concat/Split/Add/MaxPool/Resize counts, and read the Quant '
              'histogram against the reference\'s MultiThreshold count.')


if __name__ == '__main__':
    main()
