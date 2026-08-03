"""Aim-precision comparison: centre error of matched true positives.

mAP50-95 is dominated by box overlap/size; the aiming subsystem only cares where
the box CENTRE is. This mirrors notebook cell 6b-2 (greedy one-to-one matching,
conf >= CONF, IoU >= IOU_TP) so numbers are comparable to the ones already
recorded for the float models.

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
sys.path.insert(0, str(ROOT / 'yolov5'))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

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


def run(weights, paths, imgsz, device):
    model = DetectMultiBackend(weights, device=device, fp16=False)
    stride = int(model.stride)
    errs = []
    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        h, w = im0.shape[:2]
        im = letterbox(im0, imgsz, stride=stride, auto=False)[0]
        t = torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)[None].to(device)
        with torch.no_grad():
            det = non_max_suppression(model(t), CONF, IOU_NMS, max_det=300)[0]
        if len(det):
            det[:, :4] = scale_boxes(t.shape[2:], det[:, :4], im0.shape).round()
        pred = det[:, :4].cpu().numpy() if len(det) else np.zeros((0, 4))
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
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()
    device = select_device(args.device)

    print(f'{"model":<24} {"regime":<7} {"mean_px":>8} {"p95_px":>8} {"mean_frac":>10} {"p95_frac":>9} {"n_TP":>6}')
    for w in args.weights:
        for reg in args.regimes.split(','):
            paths = [l.strip() for l in (ROOT / f'configs/val_{reg}.txt').read_text().splitlines() if l.strip()]
            e = run(w, paths, args.imgsz, device)
            name = Path(w).parts[-3]
            if len(e) == 0:
                print(f'{name:<24} {reg:<7} {"-":>8} {"-":>8} {"-":>10} {"-":>9} {0:>6}')
                continue
            print(f'{name:<24} {reg:<7} {e[:,0].mean():8.3f} {np.percentile(e[:,0],95):8.3f} '
                  f'{e[:,1].mean():10.4f} {np.percentile(e[:,1],95):9.4f} {len(e):6d}')


if __name__ == '__main__':
    main()
