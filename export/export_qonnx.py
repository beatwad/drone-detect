"""Phase 5: export a Brevitas-quantized YOLOv5 to QONNX and audit it for FINN.

This is the decisive FINN-vs-DPU test. The question is not "does the export
succeed" -- it is "what operators does the graph contain, and does FINN have a
hardware node for each of them". So the script does both: writes the .onnx, then
prints an op census split into FINN-supported / joins / unsupported.

What is deliberately NOT exported
  Detect's decode (reshape, sigmoid, grid + anchor arithmetic) and NMS. Those are
  elementwise/control-flow work that belongs on the ARM side or in a separate PL
  block; handing them to FINN only gives it ops it must reject. The exported
  graph therefore ends at the raw 1x1 output conv -- see RawHead.

Joins are the thing to look at
  FINN's published failure mode on YOLO is not a missing operator (finn-hlslib
  has AddStreams_Batch, StreamingConcat, DuplicateStreams_Batch,
  UpsampleNearestNeighbour_Batch). It is that a join needs both operands
  rate-matched, so a long skip forces a whole-feature-map FIFO. The census counts
  Add/Concat separately for exactly that reason: pico should report ZERO, and
  yolov5n is the control that should report many.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov5'))
sys.path.insert(0, str(ROOT))

from qat.quantize import load_and_quantize  # noqa: E402

# Ops FINN's fpgadataflow backend has a hardware node for, after streamlining.
FINN_OK = {
    'Conv', 'MatMul', 'Gemm', 'MaxPool', 'AveragePool', 'Relu', 'BatchNormalization',
    'Mul', 'Div', 'Sub', 'Reshape', 'Transpose', 'Flatten', 'Identity', 'Pad',
    'Quant', 'BipolarQuant', 'Trunc', 'Resize', 'Upsample',
}
# Supported as ops, but each one is a rate-matched merge point in the dataflow.
FINN_JOIN = {'Add', 'Concat'}


class RawHead(nn.Module):
    """Detect reduced to its 1x1 output convs; decode dropped (see module doc)."""

    def __init__(self, detect):
        super().__init__()
        self.m = detect.m
        self.f, self.i = detect.f, detect.i
        self.type = 'RawHead'
        self.np = sum(p.numel() for p in detect.m.parameters())

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        out = [conv(xi) for conv, xi in zip(self.m, x)]
        return out[0] if len(out) == 1 else out


def census(onnx_path):
    import onnx
    g = onnx.load(str(onnx_path)).graph
    c = Counter(n.op_type for n in g.node)
    ok = {k: v for k, v in c.items() if k in FINN_OK}
    join = {k: v for k, v in c.items() if k in FINN_JOIN}
    unk = {k: v for k, v in c.items() if k not in FINN_OK | FINN_JOIN}
    return c, ok, join, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--ckpt', default=None,
                    help='calibrated / QAT state_dict (runs/qat/*.pt). Without it the '
                         'activation scales stay at their uninitialised default and the '
                         'exported graph is topologically valid but numerically useless.')
    ap.add_argument('--wbw', type=int, default=4, help='weight bit width')
    ap.add_argument('--abw', type=int, default=8, help='activation bit width')
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    model, stats = load_and_quantize(args.weights, args.wbw, args.abw, device='cpu')
    if args.ckpt:
        # build on CPU and load before any .to(device): Brevitas caches scale
        # tensors as plain attributes that .to() does not follow (see qat/evaluate.py)
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        assert ck['weight_bits'] == args.wbw and ck['act_bits'] == args.abw, \
            f"bit widths differ: ckpt W{ck['weight_bits']}A{ck['act_bits']} vs " \
            f'requested W{args.wbw}A{args.abw}'
        model.load_state_dict(ck['state_dict'])
        print(f"loaded scales from {args.ckpt} (epoch {ck.get('epoch', 'calib-only')})")
    model.model[-1] = RawHead(model.model[-1])
    model.eval()

    x = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        y = model(x)
    shapes = [tuple(t.shape) for t in (y if isinstance(y, list) else [y])]

    out = Path(args.out or ROOT / 'export' /
               f'{Path(args.weights).parts[-3]}_w{args.wbw}a{args.abw}_{args.imgsz}.onnx')
    out.parent.mkdir(parents=True, exist_ok=True)

    from brevitas.export import export_qonnx
    export_qonnx(model, args=x, export_path=str(out))

    print(f'\nquantized  : {stats}')
    print(f'output(s)  : {shapes}')
    print(f'written    : {out}  ({out.stat().st_size / 1e6:.2f} MB)')

    c, ok, join, unk = census(out)
    print(f'\nop census ({sum(c.values())} nodes)')
    print(f'  FINN-supported : {dict(sorted(ok.items()))}')
    print(f'  JOINS          : {dict(sorted(join.items())) or "none"}')
    print(f'  unsupported    : {dict(sorted(unk.items())) or "none"}')
    n_join = sum(join.values())
    print(f'\n  -> {n_join} join node(s), {sum(unk.values())} unsupported node(s)')


if __name__ == '__main__':
    main()
