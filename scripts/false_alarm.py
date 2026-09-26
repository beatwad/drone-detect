"""False alarms vs recall of one checkpoint, against the confidence threshold.

The single-class detector has no "not a drone" class: anything it scores above
the threshold is a drone. This measures how often that happens on frames with
no drone (`val_empty`), and what raising the threshold costs in drones found
(close / mid / long), from one inference pass at a low floor.

  false-alarm rate  share of empty frames with >= 1 box at conf >= t
  recall            share of ground-truth drones matched (IoU >= IOU_TP) by a
                    box at conf >= t, greedy one-to-one as in center_error.py

Single frames only: the val set is de-duplicated stills, so nothing here says
how often a false box PERSISTS over consecutive frames, which is what the
tracker's confirmation has to beat.

    uv run python -m scripts.false_alarm --weights runs/qat/v8n_p3_w4a4/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.center_error as ce                     # noqa: E402

THRS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', default='192,320')
    ap.add_argument('--device', default='0')
    ap.add_argument('--floor', type=float, default=0.10)
    args = ap.parse_args()
    imgsz = [int(v) for v in args.imgsz.split(',')]

    ce.CONF = args.floor                                  # predictor reads it at call time
    from ultralytics import YOLO
    kind = ce.checkpoint_kind(args.weights)
    if kind == 'v8':
        model = YOLO(args.weights)
    else:
        from qat.train_qat_v8 import load_qat_checkpoint
        qm, ck = load_qat_checkpoint(args.weights, ce.select_device(args.device))
        model = YOLO(ck['float_weights'])
        qm.args = model.model.args
        model.model = qm

    def predict(p):
        r = model.predict(p, imgsz=imgsz, conf=args.floor, iou=ce.IOU_NMS,
                          device=args.device, max_det=300, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    print(f"{args.weights}  imgsz {imgsz}  floor {args.floor}")
    print(f"{'set':6s} {'n':>5s} " + " ".join(f"{t:>6.2f}" for t in THRS))
    for regime in ['empty', 'close', 'mid', 'long']:
        paths = [l.strip() for l in open(ROOT / f'configs/val_{regime}.txt') if l.strip()]
        hits = np.zeros(len(THRS))
        n_gt = 0
        for p in paths:
            boxes, conf = predict(p)
            if regime == 'empty':
                hits += np.array([(conf >= t).any() for t in THRS])
                continue
            h, w = cv2.imread(p).shape[:2]
            gt = ce.gt_boxes(p, w, h)
            n_gt += len(gt)
            for k, t in enumerate(THRS):
                b = boxes[conf >= t]
                iou = ce.iou_matrix(b[np.argsort(-conf[conf >= t])], gt)
                used = set()
                for i in range(len(iou)):                 # greedy, highest conf first
                    j = [j for j in np.argsort(-iou[i]) if j not in used and iou[i, j] >= ce.IOU_TP]
                    if j:
                        used.add(j[0])
                hits[k] += len(used)
        denom = len(paths) if regime == 'empty' else n_gt
        label = 'FA %' if regime == 'empty' else 'rec %'
        print(f"{regime:6s} {denom:5d} " + " ".join(f"{100 * v / denom:6.1f}" for v in hits)
              + f"   {label}")


if __name__ == '__main__':
    main()
