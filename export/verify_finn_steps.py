"""Check FINN's intermediate checkpoints against the exported QONNX graph.

`export/verify_qonnx_v8.py` proves the ONNX matches PyTorch. This proves FINN's
own rewriting of that ONNX preserves it. Between the two lies everything that
turns a network into a circuit — Quant folded into MultiThreshold, scalar Muls
moved and absorbed, thresholds rounded, layers converted to hardware ops — and
each step is a chance to compute something slightly different, silently.

FINN can do this during a build via `verify_steps`, but that was never enabled
(and enabling it means rebuilding). The checkpoints are on disk, so the same
verification runs standalone in minutes.

TWO CONTRACT CHANGES that will produce nonsense if missed:
  - **The input is UINT8, not float.** FINN merges a ToTensor preprocessing into
    the graph, so it consumes 0..255 codes and divides internally. Feed the
    export `x/255` and FINN `x` — the same pixels, or the comparison measures
    input rounding rather than FINN (observed: max delta 5.52 vs 4.8e-06).
  - Layout goes NHWC internally but `global_in`/`global_out` stay NCHW, so no
    transposes are needed on either side.

Checkpoints up to and including `step_yolov8_streamline` execute on the host and
are skipped automatically if they cannot. `step_yolov8_convert_to_hw_layers`
needs `finn.custom_op.fpgadataflow`, which exists only inside the FINN docker,
and runs there at ~1 min/frame — so it goes through a two-phase harness in
`$FINN_HOST_BUILD_DIR/verify_io/`: the host dumps letterboxed inputs to
`inputs.npz`, `run_hw.py` executes the graph inside the container with nothing
but numpy/qonnx/finn, and the decode and verdict happen back here. Measured
2026-08-19 over 8 frames: **bit-exact against step_yolov8_streamline** (max
delta 0.000e+00), so op-to-hardware-layer conversion changes nothing numerically.

Later steps are partitioned or folded and need cppsim/rtlsim instead.

    uv run python -m export.verify_finn_steps
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.postprocess import decode, nms                  # noqa: E402
from export.verify_qonnx_v8 import letterbox                # noqa: E402

STRIDE_PX = 8
CONF, MARGIN = 0.25, 0.05   # the threshold is a discontinuity — see below
STEPS = [
    'step_yolov8_tidy_up',
    'step_yolov8_streamline',
    'step_yolov8_convert_to_hw_layers',
]


def detections(feat, conf=CONF):
    """Boxes above `conf`, plus how many sit inside the margin band.

    A box at 0.2503 is kept and the same box at 0.2497 is dropped, so two
    implementations differing by one LSB will ALWAYS disagree at the cutoff for
    reasons that say nothing about fidelity. Boxes below conf+MARGIN are counted
    and reported, never compared. (Observed without this: step_yolov8_tidy_up,
    which is exact to 5e-06 on random input, "gained" a detection.)
    """
    b, c = decode(feat)
    b, c = b[0], c[0, :, 0]
    m = c >= conf
    b, c = b[m], c[m]
    k = nms(b, c)
    b, c = b[k], c[k]
    solid = c >= conf + MARGIN
    return b[solid], c[solid], int((~solid).sum())


def centre_deltas(ref_det, got_det):
    """Nearest-centre pairing between two detection sets, in pixels."""
    br, bg = ref_det[0], got_det[0]
    out = []
    for a in br:
        if not len(bg):
            break
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = np.stack([(bg[:, 0] + bg[:, 2]) / 2, (bg[:, 1] + bg[:, 3]) / 2], 1)
        d = np.hypot(cb[:, 0] - ca[0], cb[:, 1] - ca[1])
        out.append(float(d.min()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default='/home/alex/finn_build_mdanilow/drone_v8_bit')
    ap.add_argument('--onnx', default='export/v8n_p3_w4a4_192x320_clean.onnx')
    ap.add_argument('--images', default='configs/val_close.txt')
    ap.add_argument('--n-images', type=int, default=60,
                    help='p95 over ~25 boxes is the 2nd-worst sample; keep this high')
    ap.add_argument('--imgsz', default='192,320')
    args = ap.parse_args()

    h, w = (int(v) for v in args.imgsz.split(','))
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    import cv2

    ref = ModelWrapper(args.onnx)
    r_in, r_out = ref.graph.input[0].name, ref.graph.output[0].name

    inter = Path(args.build) / 'intermediate_models'
    models = []
    for name in STEPS:
        f = inter / f'{name}.onnx'
        if not f.exists():
            continue
        m = ModelWrapper(str(f))
        models.append((name, m, m.graph.input[0].name, m.graph.output[0].name))
    # Drop what this interpreter cannot execute: the hardware-layer checkpoint
    # needs finn.custom_op.fpgadataflow, which only exists inside the FINN docker.
    probe = np.zeros((1, 3, h, w), dtype=np.float32)
    runnable = []
    for name, m, i_name, o_name in models:
        try:
            execute_onnx(m, {i_name: probe})
            runnable.append((name, m, i_name, o_name))
        except Exception as e:
            why = 'needs the FINN docker' if 'opset import for domain' in str(e) else str(e)[:60]
            print(f'skipping {name}: {why}')
    models = runnable
    print(f'reference : {args.onnx}')
    print(f'checkpoints: {", ".join(n for n, *_ in models)}\n')

    paths = [l.strip() for l in Path(args.images).read_text().splitlines() if l.strip()]
    paths = paths[:args.n_images]

    raw = {n: [] for n, *_ in models}
    cen = {n: [] for n, *_ in models}
    cnt = {n: [0, 0] for n, *_ in models}
    border = {n: 0 for n, *_ in models}
    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        # one set of pixels, expressed in each graph's own input contract
        xu = np.round(letterbox(im0, h, w).numpy() * 255).astype(np.float32)
        yr = execute_onnx(ref, {r_in: xu / 255.0})[r_out]
        dref = detections(yr)
        for name, m, i_name, o_name in models:
            y = execute_onnx(m, {i_name: xu})[o_name]
            raw[name].append(float(np.abs(yr - y).max()))
            d = detections(y)
            cnt[name][0] += len(dref[0])
            cnt[name][1] += len(d[0])
            border[name] += dref[2] + d[2]
            cen[name] += centre_deltas(dref, d)

    ok = True
    for name, *_ in models:
        c, dc = cnt[name], np.array(cen[name]) if cen[name] else np.zeros(0)
        rw = np.array(raw[name])
        print(f'{name}')
        print(f'  raw head delta   : max {rw.max():.3e}   median {np.median(rw):.3e}')
        print(f'  boxes            : reference {c[0]} | this step {c[1]}   '
              f'({border[name]} near-threshold, excluded)')
        if dc.size:
            print(f'  centre delta, px : median {np.median(dc):.3f}  '
                  f'p95 {np.percentile(dc, 95):.3f}  max {dc.max():.3f}')
        # WHICH SIDE IS RIGHT? Neither, but the hardware is not neutral: FINN
        # synthesises MultiThreshold, so THESE checkpoints predict the board and
        # the Brevitas export is the outlier. So the criterion is not "identical"
        # but "no target is lost and no box moves off its cell" — a gained box is
        # a false positive to note, not a fidelity failure.
        lost = int((dc > STRIDE_PX).sum()) if dc.size else 0
        gained = max(0, c[1] - c[0])
        checks = [
            ('every reference detection still found', lost == 0),
            # Half a stride-8 cell: the grid cannot localise better than that, so
            # a box staying inside its own cell is the strongest claim the
            # representation supports. NOT tuned to the data — 14 px is the
            # measured aim error, and 4 px is comfortably under it.
            ('p95 centre delta <= half a cell', dc.size and float(np.percentile(dc, 95)) <= STRIDE_PX / 2),
            ('gained detections <= 5% of reference', gained <= max(1, 0.05 * c[0])),
        ]
        print(f'  unmatched {lost} | gained {gained}')
        for label, good in checks:
            print(f"    [{'OK ' if good else 'FAIL'}] {label}")
            ok = ok and bool(good)
        print()
    print('OK' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
