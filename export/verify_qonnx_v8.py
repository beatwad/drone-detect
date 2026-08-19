"""Check the exported YOLOv8n-P3 QONNX graph against the QAT model that produced it.

`export/verify_qonnx.py` does this for the yolov5 path and cannot be reused: it
imports the vendored yolov5 loader and yolov5's anchor-based decode. This is the
same idea for the v8 path — same acceptance criterion, different plumbing.

WHY IT EXISTS AT ALL
  The v8 export was checked STRUCTURALLY (op census and Quant histogram matched
  the reference op-for-op, build_notes §10.11) and `deploy/postprocess.py` was
  checked against Ultralytics' own decode. Neither compares NUMBERS between the
  exported graph and the model. A graph that writes cleanly, carries the right
  ops and computes something slightly different would pass both and only surface
  as bad detections on the board — after a ~24 h rebuild to fix.

ACCEPTANCE CRITERION — why it is a distribution, not a maximum
  Brevitas and qonnx's executor round in OPPOSITE directions for a value sitting
  exactly on a quantizer boundary, so a few activations differ by exactly one
  quantization step. Measured here (random input, `--layers`): every non-zero
  delta is ±1 step to within 1e-7 — 6 of 245,760 elements at layer 0, growing to
  449 of 15,360 by layer 8 as a flipped LSB creates fresh boundary hits
  downstream. That is inherent to comparing two implementations, not a defect.

  It bites harder here than on the yolov5 W8A8 export, where the same check
  passes with a 2 px tolerance: our activation step is 6/15 = 0.4 against
  6/255 = 0.0235, **17× coarser**, and the v8 head has no output activation
  quantizer, so the flip lands straight in the summed logits.

  So a max-based tolerance is unfalsifiable theatre. What is required instead:
  no detection may be gained or lost, the typical box must agree to well under
  a pixel, and no box may move by more than one stride-8 cell — beyond that the grid itself is
  coarser than the disagreement.

  NEITHER SIDE IS "RIGHT". The hardware is a third implementation (MultiThreshold
  comparisons), so this measures the SIZE of implementation noise, not which
  simulator to trust. Judge it against the measured aim error, ~14 px.

    uv run python -m export.verify_qonnx_v8
    uv run python -m export.verify_qonnx_v8 --layers   # per-layer LSB diagnosis
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qat.train_qat_v8 import load_qat_checkpoint          # noqa: E402
from export.export_qonnx_v8 import RawV8Head, use_split_chunks  # noqa: E402
from deploy.postprocess import decode, nms                 # noqa: E402

# Finest per-channel output step of the built accelerator (deploy/*.npz). Used
# only to express the raw-tensor delta in units the hardware can represent.
OUT_LSB_MIN = 1.132e-04
STRIDE_PX = 8          # one P3 cell; the grid is coarser than this disagreement


def load_reference(ckpt):
    """The exact model export_qonnx_v8.py serialises: QAT weights + both patches."""
    model, ck = load_qat_checkpoint(ckpt, 'cpu')
    model.model[-1] = RawV8Head(model.model[-1])
    use_split_chunks(model)
    return model.eval(), ck


def letterbox(im0, h, w):
    """BGR uint8 image -> (1,3,h,w) float tensor, matching the export geometry."""
    from ultralytics.data.augment import LetterBox
    im = LetterBox(new_shape=(h, w), auto=False)(image=im0)
    x = im[:, :, ::-1].transpose(2, 0, 1).copy()
    return torch.from_numpy(x).float().div(255)[None]


def detection_check(ref, onnx_model, paths, h, w, conf=0.25, margin=0.05):
    """Do the two graphs detect the same things on real images?

    Raw-tensor agreement is a proxy; box agreement is the requirement. Boxes
    within `margin` of the threshold are counted and reported, never failed on:
    the threshold is a discontinuity, so two implementations differing by one
    LSB will always disagree there for reasons that say nothing about fidelity.
    """
    from qonnx.core.onnx_exec import execute_onnx
    import cv2

    inp = onnx_model.graph.input[0].name
    out = onnx_model.graph.output[0].name
    n_t = n_o = n_border = 0
    dcen, dcnf = [], []
    draw = 0.0

    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        x = letterbox(im0, h, w)

        with torch.no_grad():
            yt = ref(x)
        yt = (yt[0] if isinstance(yt, list) else yt).numpy()
        yo = execute_onnx(onnx_model, {inp: x.numpy()})[out]
        draw = max(draw, float(np.abs(yt - yo).max()))

        det = []
        for feat in (yt, yo):
            b, c = decode(feat)
            b, c = b[0], c[0, :, 0]
            m = c >= conf
            b, c = b[m], c[m]
            k = nms(b, c)
            det.append((b[k], c[k]))

        (bt, ct), (bo, co) = det
        n_border += int((ct < conf + margin).sum() + (co < conf + margin).sum())
        bt, ct = bt[ct >= conf + margin], ct[ct >= conf + margin]
        bo, co = bo[co >= conf + margin], co[co >= conf + margin]
        n_t += len(bt)
        n_o += len(bo)
        for a, sa in zip(bt, ct):                 # nearest-centre pairing, 2 px
            if not len(bo):
                break
            ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
            cb = np.stack([(bo[:, 0] + bo[:, 2]) / 2, (bo[:, 1] + bo[:, 3]) / 2], 1)
            d = np.hypot(cb[:, 0] - ca[0], cb[:, 1] - ca[1])
            j = int(d.argmin())
            dcen.append(float(d[j]))
            dcnf.append(abs(float(sa) - float(co[j])))

    return dict(n_t=n_t, n_o=n_o, n_border=n_border, draw=draw,
                dcen=np.array(dcen), dcnf=np.array(dcnf))


def layer_diagnosis(ref, onnx_model, h, w):
    """Where does the divergence start, and is it a whole number of LSBs?

    This is what turns a failing tolerance into a diagnosis. Runs both graphs on
    one input, hooks every top-level layer of the torch model, and for each finds
    the ONNX tensor of the same shape that matches best. A delta that is an exact
    multiple of the activation step is boundary rounding; anything else is a bug.

    LIMIT: the ONNX tensor is picked by shape + best agreement, so where several
    same-shape candidates exist (SPPF, the C2f splits) it can latch onto a
    PRE-quantization tensor, and the lattice test then reads as off-lattice.
    Trust the column where the match is unambiguous. The last row is the head,
    which has no output activation quantizer at all — off-lattice by construction.
    """
    from qonnx.core.onnx_exec import execute_onnx

    caught = {}
    handles = [m.register_forward_hook(
        lambda mod, i_, out, i=i: caught.__setitem__(i, out))
        for i, m in enumerate(ref.model)]
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        ref(x)
    for handle in handles:
        handle.remove()

    ctx = execute_onnx(onnx_model, {onnx_model.graph.input[0].name: x.numpy()},
                       return_full_exec_context=True)
    arrs = {k: v for k, v in ctx.items()
            if isinstance(v, np.ndarray) and v.ndim == 4}

    print(f"\n{'layer':<6}{'shape':<22}{'max|delta|':<13}{'differing':<18}"
          f"{'in LSB':<10}{'off-lattice':<12}")
    for i in sorted(caught):
        t = caught[i]
        if not torch.is_tensor(t):
            continue
        t = t.detach().numpy()
        cand = [(np.abs(t - v).max(), k) for k, v in arrs.items() if v.shape == t.shape]
        if not cand:
            continue
        dmax, k = min(cand)
        step = 6 / 255 if i == 0 else 6 / 15      # A8 stem, A4 body (build_notes §10.9)
        diff = (t - arrs[k]).ravel()
        nz = diff[np.abs(diff) > 1e-9]
        mult = np.round(nz / step) if nz.size else np.zeros(0)
        off = float(np.abs(nz - mult * step).max()) if nz.size else 0.0
        lsb = sorted({int(v) for v in mult})[:5]
        print(f'{i:<6}{str(t.shape):<22}{dmax:<13.4e}'
              f'{f"{nz.size}/{t.size}":<18}{str(lsb):<10}{off:<12.2e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='export/v8n_p3_w4a4_192x320_clean.onnx')
    ap.add_argument('--ckpt', default='runs/qat/v8n_p3_w4a4/weights/best.pt')
    ap.add_argument('--imgsz', default='192,320')
    ap.add_argument('--images', default='configs/val_close.txt')
    ap.add_argument('--n-images', type=int, default=40)
    ap.add_argument('--layers', action='store_true',
                    help='per-layer divergence, in units of the activation step')
    args = ap.parse_args()

    h, w = (int(v) for v in args.imgsz.split(','))

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes

    onnx_model = ModelWrapper(args.onnx).transform(InferShapes())
    ref, ck = load_reference(args.ckpt)
    print(f"checkpoint : {args.ckpt} (epoch {ck.get('epoch')}, "
          f"W{ck['low_bits']} body / W{ck['high_bits']} stem+head)")
    print(f'onnx       : {args.onnx} ({len(onnx_model.graph.node)} nodes)')

    # 1. raw tensors on a deterministic input
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        yt = ref(x)
    yt = (yt[0] if isinstance(yt, list) else yt).numpy()
    inp = onnx_model.graph.input[0].name
    yo = execute_onnx(onnx_model, {inp: x.numpy()})[onnx_model.graph.output[0].name]

    d = np.abs(yt - yo)
    print(f'\nraw head tensor {yt.shape} on random input')
    print(f'  max |delta| {d.max():.3e}   mean {d.mean():.3e}   '
          f'= {d.max() / OUT_LSB_MIN:.2f} x the finest hardware output step')
    print(f'  elements differing > 1 step: {int((d > OUT_LSB_MIN).sum())} / {d.size}')

    if args.layers:
        layer_diagnosis(ref, onnx_model, h, w)

    # 2. the criterion that matters
    paths = [l.strip() for l in Path(args.images).read_text().splitlines() if l.strip()]
    paths = paths[:args.n_images]
    r = detection_check(ref, onnx_model, paths, h, w)
    dc, dk = r['dcen'], r['dcnf']

    print(f"\ndetection check ({len(paths)} real close-regime images, conf >= 0.30)")
    print(f"  torch {r['n_t']} boxes | onnx {r['n_o']} boxes")
    if dc.size:
        print(f"  centre delta, px : median {np.median(dc):.3f}  p95 "
              f"{np.percentile(dc, 95):.3f}  max {dc.max():.3f}")
        print(f"                     bit-identical {int((dc == 0).sum())}/{dc.size}, "
              f"<=1px {int((dc <= 1).sum())}/{dc.size}, "
              f">1 cell (8px) {int((dc > STRIDE_PX).sum())}")
        print(f"  conf delta       : median {np.median(dk):.4f}  "
              f"p95 {np.percentile(dk, 95):.4f}  max {dk.max():.4f}")
    print(f"  max raw head delta over these images {r['draw']:.3e}")
    print(f"  ({r['n_border']} near-threshold box(es) excluded)")

    # Zero detections is INCONCLUSIVE, not a pass: a model that finds nothing
    # agrees with itself trivially, and must not be able to certify itself.
    if not r['n_t']:
        print('\n  INCONCLUSIVE (no detections)')
        return 1
    checks = [
        ('no detection gained or lost', r['n_t'] == r['n_o']),
        ('median centre delta < 0.5 px', float(np.median(dc)) < 0.5),
        ('p95 centre delta <= 2 px',    float(np.percentile(dc, 95)) <= 2.0),
        ('no box moves > 1 cell',       float(dc.max()) <= STRIDE_PX),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    good = all(ok for _, ok in checks)
    print(f"\n  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
