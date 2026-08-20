"""Tracking and aim output — the flow specified in README §11. NumPy only.

Turns the accelerator's pre-NMS boxes into the two things the aiming subsystem
consumes: an offset from frame centre, and a `close_enough` flag. One target,
by construction — the system aims at a single drone.

WHERE THIS RUNS
  On the A53s under Linux, for the baseline. That is explicitly temporary: the
  deployment target is this whole chain in PL alongside a MIPI camera, so that
  nothing between exposure and aim command is scheduled by Linux. See CLAUDE.md.

THREE CHOICES THE SPEC LEAVES OPEN
  1. **Alpha-beta, not a full Kalman.** A fixed-gain filter on the centre is what
     a constant-velocity Kalman converges to anyway, and it needs no covariance
     bookkeeping. The gate compares IoU against a predicted *box*, so a size is
     needed too — but size has no useful dynamics here, so it is a plain EMA.
     If measurement noise ever needs to be estimated rather than assumed, this is
     the piece to replace.
  2. **Distances are fractions of the frame diagonal**, not pixels. That is the
     unit aim error is already reported in everywhere else in this project
     (`scripts/center_error.py`, 0.0139 for the shipping model), so the centring
     thresholds can be read against measurements that already exist.
  3. **`dt` is passed in, never measured here.** The whole point of the camera
     trigger discussion is that the filter must advance from the *exposure*
     timestamp, not from when the frame happened to arrive. Handing this module a
     clock reading would bake in exactly the error the trigger exists to remove.

EVERY THRESHOLD BELOW IS A GUESS. They are structurally sensible and nothing
more: no footage has been shot through the real lens yet, and the centring gate
in particular depends on the aiming subsystem's tolerance, which is not specified.
Treat them as placeholders to be measured, not as tuned values.
"""
from typing import NamedTuple

import numpy as np

CONF_THR = 0.25        # seed/candidate confidence
IOU_CLUSTER = 0.55     # boxes this close to the seed are the same drone
IOU_GATE = 0.30        # measurement still agrees with the prediction
MISS_LIMIT = 5         # consecutive gate failures before re-seeding
ALPHA, BETA = 0.6, 0.2  # alpha-beta gains: position, velocity
SIZE_EMA = 0.5         # box size smoothing
D_LOW, D_HIGH = 0.010, 0.020   # centring hysteresis, fraction of frame diagonal
DEBOUNCE = 3           # frames a gate decision must hold before it flips


class Aim(NamedTuple):
    """What the aiming subsystem gets. `offset` is None when there is no track."""
    close_enough: bool
    offset: tuple    # (dx, dy) in pixels from frame centre, or None
    dist: float      # |offset| as a fraction of the frame diagonal
    box: np.ndarray  # smoothed xyxy, or None
    tracking: bool   # the filter holds a track (may still be un-centred)


def _iou_1_to_n(box, boxes):
    """IoU of one xyxy box against an (N, 4) array."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def wbf(boxes, scores):
    """Weighted box fusion: confidence-weighted mean of a cluster."""
    w = scores.reshape(-1, 1)
    return (boxes * w).sum(0) / max(float(w.sum()), 1e-9)


def _centre(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _box_from(centre, size):
    hw, hh = size[0] / 2.0, size[1] / 2.0
    return np.array([centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh])


class AimTracker:
    """README §11, one instance per camera. Call `update` once per frame."""

    def __init__(self, frame_wh, conf_thr=CONF_THR, iou_cluster=IOU_CLUSTER,
                 iou_gate=IOU_GATE, miss_limit=MISS_LIMIT,
                 d_low=D_LOW, d_high=D_HIGH, debounce=DEBOUNCE):
        self.w, self.h = frame_wh
        self.centre = np.array([self.w / 2.0, self.h / 2.0])
        self.diag = float(np.hypot(self.w, self.h))
        self.conf_thr, self.iou_cluster, self.iou_gate = conf_thr, iou_cluster, iou_gate
        self.miss_limit, self.d_low, self.d_high, self.debounce = miss_limit, d_low, d_high, debounce
        self._reset()

    def _reset(self):
        self.pos = None          # filter state: centre, velocity (px, px/s), size
        self.vel = np.zeros(2)
        self.size = None
        self.miss = 0
        self.recent = []         # measurements collected while the gate is failing
        self.close_enough = False
        self._pending = 0        # debounce counter for the centring gate

    # ---- step 7: hysteresis + debounce ------------------------------------
    def _centring_gate(self, dist):
        want = self.close_enough
        if dist < self.d_low:
            want = True
        elif dist > self.d_high:
            want = False
        # between D_low and D_high the flag holds — that is the hysteresis.
        if want == self.close_enough:
            self._pending = 0
        else:
            self._pending += 1
            if self._pending >= self.debounce:
                self.close_enough = want
                self._pending = 0
        return self.close_enough

    def _no_track(self):
        return Aim(False, None, float('nan'), None, self.pos is not None)

    def update(self, boxes, scores, dt):
        """boxes (N,4) xyxy and scores (N,) — PRE-NMS — plus seconds since the
        previous frame's exposure. Returns `Aim`."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        # 1. seed: closest to frame centre among confident boxes
        keep = scores >= self.conf_thr
        if not keep.any():
            self.close_enough = False
            self._pending = 0
            return self._no_track()
        cand, cand_s = boxes[keep], scores[keep]
        d = np.hypot(*(np.stack([_centre(b) for b in cand]) - self.centre).T)
        seed = cand[int(np.argmin(d))]

        # 2-3. cluster around the seed, fuse it into one measurement
        m = _iou_1_to_n(seed, cand) > self.iou_cluster
        measured = wbf(cand[m], cand_s[m])
        z_c, z_size = _centre(measured), np.array([measured[2] - measured[0],
                                                   measured[3] - measured[1]])

        # first measurement: initialise and report a track but no aim yet
        if self.pos is None:
            self.pos, self.vel, self.size = z_c, np.zeros(2), z_size
            self.miss, self.recent = 0, []
            return self._no_track()

        # 4. predict
        pred_c = self.pos + self.vel * dt
        pred_box = _box_from(pred_c, self.size)

        # 5. gate
        if _iou_1_to_n(measured, pred_box.reshape(1, 4))[0] <= self.iou_gate:
            self.miss += 1
            self.recent.append(measured)
            if self.miss > self.miss_limit:
                self._reseed(dt)
            self.close_enough = False
            self._pending = 0
            return self._no_track()
        self.miss, self.recent = 0, []

        # 6. update — alpha-beta correction on the prediction
        resid = z_c - pred_c
        self.pos = pred_c + ALPHA * resid
        if dt > 0:
            self.vel = self.vel + (BETA / dt) * resid
        self.size = (1 - SIZE_EMA) * self.size + SIZE_EMA * z_size

        offset = self.pos - self.centre
        dist = float(np.hypot(*offset)) / self.diag
        return Aim(self._centring_gate(dist), (float(offset[0]), float(offset[1])),
                   dist, _box_from(self.pos, self.size), True)

    def _reseed(self, dt):
        """Restart the filter from the longest run of mutually agreeing recent
        measurements, newest first. A single outlier therefore cannot re-seed the
        track — it takes a run that is self-consistent by the same IoU the gate
        uses."""
        run = [self.recent[-1]]
        for prev in reversed(self.recent[:-1]):
            if _iou_1_to_n(run[-1], prev.reshape(1, 4))[0] <= self.iou_gate:
                break
            run.append(prev)
        run = run[::-1]                       # oldest -> newest
        newest = run[-1]
        self.pos, self.size = _centre(newest), np.array([newest[2] - newest[0],
                                                         newest[3] - newest[1]])
        # Velocity from the run's own span. NOTE the units: the filter carries
        # px/s because predict multiplies by dt, so the per-step displacement has
        # to be divided by dt as well — dropping that is a silent factor of ~1/fps.
        span = max(len(run) - 1, 1) * max(dt, 1e-6)
        self.vel = (_centre(run[-1]) - _centre(run[0])) / span if len(run) > 1 else np.zeros(2)
        self.miss, self.recent = 0, []
