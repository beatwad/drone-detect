"""Aim-precision comparison: centre error of matched true positives.

mAP50-95 is dominated by box overlap/size; the aiming subsystem only cares where
the box CENTRE is. This mirrors notebook cell 6b-2 (greedy one-to-one matching,
conf >= CONF, IoU >= IOU_TP) so numbers are comparable to the ones already
recorded for the float models.

Loads both checkpoint families the project still has: plain Ultralytics `.pt` and
the Brevitas QAT checkpoints from `qat/train_qat_v8.py`.

n_TP is reported alongside: centre error is conditioned on matched detections, so
a model that finds fewer drones is being scored on an easier subset. Compare the
error only when n_TP is close.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from ultralytics.utils.torch_utils import select_device  # noqa: E402

CONF, IOU_NMS, IOU_TP = 0.5, 0.45, 0.50


def iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (ap[:, None] + ag[None, :] - inter + 1e-9)


def gt_boxes(img_path, w, h):
    lbl = Path(str(img_path).replace('/images/', '/labels/')).with_suffix('.txt')
    if not lbl.exists():
        return np.zeros((0, 4))
    out = []
    for row in lbl.read_text().split('\n'):
        p = row.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out) if out else np.zeros((0, 4))


def checkpoint_kind(weights):
    """'v8' or 'v8_qat' — the two checkpoint families in this repo.

    v8 pickles its graph as `ultralytics.nn.tasks.*`, an unambiguous marker.
    v8_qat (qat/train_qat_v8.py) carries a state_dict plus the bit widths, since
    Brevitas' runtime-generated quantizer classes do not pickle safely.
    """
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'state_dict' in ck and 'low_bits' in ck:
        return 'v8_qat'
    return 'v8'


def predictor(weights, imgsz, device):
    """Return f(img_path, im0) -> xyxy boxes in ORIGINAL image pixels, conf>=CONF.

    Ultralytics letterboxes to `imgsz` and scales back, so the boxes the matcher
    sees are in source pixels regardless of the input geometry.
    """
    from ultralytics import YOLO
    if checkpoint_kind(weights) == 'v8':
        model = YOLO(weights)
    else:
        sys.path.insert(0, str(ROOT))
        from qat.train_qat_v8 import load_qat_checkpoint  # noqa: E402
        qm, ck = load_qat_checkpoint(weights, device)     # CPU-load, then move
        model = YOLO(ck['float_weights'])                 # reuse its args/names
        qm.args = model.model.args
        model.model = qm

    def predict(p, im0):
        r = model.predict(p, imgsz=imgsz, conf=CONF, iou=IOU_NMS,
                          device=device, max_det=300, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy()
    return predict


def run(weights, paths, imgsz, device):
    predict = predictor(weights, imgsz, device)
    errs = []
    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        h, w = im0.shape[:2]
        pred = predict(p, im0)
        gt = gt_boxes(p, w, h)
        if len(pred) == 0 or len(gt) == 0:
            continue
        m = iou_matrix(pred, gt)
        used_p, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
        for i, j in order:                      # greedy one-to-one, highest IoU first
            if m[i, j] < IOU_TP:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            pc = ((pred[i, 0] + pred[i, 2]) / 2, (pred[i, 1] + pred[i, 3]) / 2)
            gc = ((gt[j, 0] + gt[j, 2]) / 2, (gt[j, 1] + gt[j, 3]) / 2)
            px = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]))
            errs.append((px, px / float(np.hypot(w, h))))
    return np.array(errs) if errs else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--regimes', default='close,mid,long')
    # accepts 320 or 'H,W'. Ultralytics' val() silently rounds a [h, w] to an
    # int ('train and val imgsz must be an integer'); only predict honours a
    # rectangle, and this script goes through predict — so this is the only
    # way to score a model at the non-square shape the bitstream will run.
    ap.add_argument('--imgsz', default='416',
                    help="square size, or 'H,W' for a rectangular input")
    ap.add_argument('--device', default='0')
    args = ap.parse_args()
    device = select_device(args.device)
    imgsz = ([int(v) for v in args.imgsz.split(',')] if ',' in args.imgsz
             else int(args.imgsz))

    print(f'{"model":<24} {"regime":<7} {"mean_px":>8} {"p95_px":>8} {"mean_frac":>10} {"p95_frac":>9} {"n_TP":>6}')
    for w in args.weights:
        for reg in args.regimes.split(','):
            paths = [l.strip() for l in (ROOT / f'configs/val_{reg}.txt').read_text().splitlines() if l.strip()]
            e = run(w, paths, imgsz, device)
            name = Path(w).parts[-3]
            if len(e) == 0:
                print(f'{name:<24} {reg:<7} {"-":>8} {"-":>8} {"-":>10} {"-":>9} {0:>6}')
                continue
            print(f'{name:<24} {reg:<7} {e[:,0].mean():8.3f} {np.percentile(e[:,0],95):8.3f} '
                  f'{e[:,1].mean():10.4f} {np.percentile(e[:,1],95):9.4f} {len(e):6d}')


if __name__ == '__main__':
    main()
