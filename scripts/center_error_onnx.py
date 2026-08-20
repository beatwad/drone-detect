"""Aim precision of a FINN checkpoint vs the exported graph — the metric that ships.

`export/verify_finn_steps.py` shows the two graphs disagree by up to one
activation step on boundary values, bimodally per frame. That bounds the noise
but does not say what it costs. This does: the SAME centre-error metric already
recorded for every model in this project, computed from raw head tensors, so a
FINN checkpoint can be scored exactly like a checkpoint.

Why it matters: every accuracy number we have was measured on the PyTorch model,
i.e. under Brevitas' rounding. The board runs MultiThreshold. If the convention
moves the metric, better to know before bring-up than while debugging hardware.

Metric constants and matching are imported from `scripts/center_error.py`, not
copied, so the numbers are comparable to the ones already recorded.

    uv run python -m scripts.center_error_onnx --n 250
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.center_error import CONF, IOU_NMS, IOU_TP, iou_matrix, gt_boxes  # noqa: E402
from deploy.postprocess import decode, nms            # noqa: E402
from export.verify_qonnx_v8 import letterbox          # noqa: E402


def letterbox_params(h0, w0, h, w):
    """Ultralytics LetterBox geometry, and its inverse for mapping boxes back."""
    r = min(h / h0, w / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    return r, (w - nw) / 2, (h - nh) / 2


def predict(feat, h0, w0, h, w):
    """Raw head tensor -> boxes in ORIGINAL image pixels, conf >= CONF."""
    xyxy, conf = decode(feat)
    b, c = xyxy[0], conf[0, :, 0]
    m = c >= CONF
    b, c = b[m], c[m]
    k = nms(b, c, IOU_NMS)
    b = b[k]
    if not len(b):
        return b
    r, dw, dh = letterbox_params(h0, w0, h, w)
    b = b - np.array([dw, dh, dw, dh], dtype=np.float32)
    return b / r


def centre_errors(feats, paths, h, w):
    """feats: dict name -> list of head tensors, aligned with paths."""
    out = {k: [] for k in feats}
    for idx, p in enumerate(paths):
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        h0, w0 = im0.shape[:2]
        gt = gt_boxes(p, w0, h0)
        if not len(gt):
            continue
        for name, fs in feats.items():
            pred = predict(fs[idx], h0, w0, h, w)
            if not len(pred):
                continue
            m = iou_matrix(pred, gt)
            used_p, used_g = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
            for i, j in order:                  # greedy one-to-one, highest IoU first
                if m[i, j] < IOU_TP:
                    break
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                pc = ((pred[i, 0] + pred[i, 2]) / 2, (pred[i, 1] + pred[i, 3]) / 2)
                gc = ((gt[j, 0] + gt[j, 2]) / 2, (gt[j, 1] + gt[j, 3]) / 2)
                px = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]))
                out[name].append((px, px / float(np.hypot(w0, h0))))
    return {k: (np.array(v) if v else np.zeros((0, 2))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='export/v8n_p3_w4a4_192x320_clean.onnx')
    ap.add_argument('--step', default='/home/alex/finn_build_mdanilow/drone_v8_bit'
                                      '/intermediate_models/step_yolov8_streamline.onnx')
    ap.add_argument('--regimes', default='close,mid,long')
    ap.add_argument('--n', type=int, default=250,
                    help='images per regime, sampled uniformly across sources')
    ap.add_argument('--imgsz', default='192,320')
    args = ap.parse_args()

    h, w = (int(v) for v in args.imgsz.split(','))
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx

    ref = ModelWrapper(args.onnx)
    st = ModelWrapper(args.step)
    graphs = {'export (Brevitas rounding)': (ref, False),
              'FINN streamlined (MultiThreshold)': (st, True)}

    print(f'{"regime":<8}{"graph":<36}{"n_TP":>6}{"px":>9}{"normalised":>13}')
    for regime in args.regimes.split(','):
        paths = [l.strip() for l in
                 (ROOT / f'configs/val_{regime}.txt').read_text().splitlines() if l.strip()]
        # STRATIFY. configs/val_*.txt is ordered by source, so paths[:n] covers
        # only the first one or two of nine — and the boundary-hit effect this
        # measures is data-dependent (flat sky), so a source-biased subset is not
        # evidence about the deployed model. Take every k-th instead.
        if args.n and args.n < len(paths):
            paths = paths[::max(1, len(paths) // args.n)][:args.n]
        feats = {k: [] for k in graphs}
        kept = []
        for p in paths:
            im0 = cv2.imread(p)
            if im0 is None:
                continue
            xu = np.round(letterbox(im0, h, w).numpy() * 255).astype(np.float32)
            for name, (m, is_uint8) in graphs.items():
                x = xu if is_uint8 else xu / 255.0
                feats[name].append(
                    execute_onnx(m, {m.graph.input[0].name: x})[m.graph.output[0].name])
            kept.append(p)
        res = centre_errors(feats, kept, h, w)
        for name in graphs:
            e = res[name]
            if len(e):
                print(f'{regime:<8}{name:<36}{len(e):>6}{e[:, 0].mean():>9.2f}'
                      f'{e[:, 1].mean():>13.4f}')
            else:
                print(f'{regime:<8}{name:<36}{0:>6}{"-":>9}{"-":>13}')
        print()


if __name__ == '__main__':
    main()
