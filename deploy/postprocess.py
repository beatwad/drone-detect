"""Host-side decode for the FINN accelerator's raw head output. NumPy only.

The accelerator emits `(1, 24, 40, 65)` **INT21** in NHWC — box DFL logits (64) and
one class logit, per stride-8 cell. Everything after that runs on the ARM side:
dequantize, DFL expectation, distance-to-box, sigmoid, NMS.

THE DEQUANTIZATION IS NOT OPTIONAL, AND THE REFERENCE DRIVER OMITS IT
  FINN's `step_create_dataflow_partition` leaves the final per-channel `Mul` and
  `Add` in the PARENT graph, outside the accelerator — they are float ops with no
  hardware node. Measured on our build and on the reference build alike:

      parent = Transpose -> StreamingDataflowPartition -> Transpose -> Mul -> Add

  ours       Mul (1,65,1,1)  1.13e-4 .. 3.30e-4     Add  -0.052 .. 2.094
  reference  Mul (1,144,1,1) 3.71e-20 .. 3.75e-4    Add  -99.76 .. 6.42

  So the true value is `int21 * scale[c] + bias[c]`, with scale ~1e-4.
  `yolov8_utils.yolov8_postproc` in the reference driver applies sigmoid and
  softmax straight to the raw integers, i.e. to values ~10^4 too large: sigmoid
  saturates to 0/1 and the DFL softmax degenerates into an argmax. Do not copy it.

  `scale` and `bias` come from the build, not from the model — see
  `deploy/v8n_p3_w4a4_192x320_dequant.npz`, extracted from
  `intermediate_models/dataflow_parent.onnx`. They change whenever the network is
  retrained or rebuilt.

VALIDATION (2026-08-14)
  `decode` was checked against Ultralytics' own `Detect._inference` on the same
  head tensor, over all 960 cells x 25 images: **max box difference 6.1e-5 px,
  max confidence difference 1.2e-7** — float32 rounding. The DFL, anchor and
  sigmoid maths are therefore exact.

  End-to-end through NMS, 43 boxes over 40 images: 39 agree with Ultralytics
  within 1 px, the rest within 3.4 px. That residue is NMS tie-breaking between
  near-duplicate boxes (greedy numpy here vs torchvision there), not a decode
  difference — it changes which of two overlapping boxes represents an object,
  never whether the object is found.
"""
import numpy as np

REG_MAX = 16          # DFL bins per box side
STRIDE = 8            # single P3 head
NC = 1                # drone


def dequantize(raw_nhwc, scale, bias):
    """(1, H, W, C) accelerator ints -> (1, C, H, W) float logits."""
    x = np.asarray(raw_nhwc, dtype=np.float32)
    x = x * scale.reshape(1, 1, 1, -1) + bias.reshape(1, 1, 1, -1)
    return x.transpose(0, 3, 1, 2)


def make_anchors(h, w, stride, offset=0.5):
    """Cell centres in feature-map units, and the matching stride vector."""
    sx = np.arange(w, dtype=np.float32) + offset
    sy = np.arange(h, dtype=np.float32) + offset
    sx, sy = np.meshgrid(sx, sy)
    pts = np.stack((sx, sy), -1).reshape(-1, 2).T          # (2, HW)
    return pts[None], np.full((1, 1, h * w), stride, np.float32)


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)                # stabilised
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode(feat, stride=STRIDE, reg_max=REG_MAX, nc=NC):
    """(1, C, H, W) logits -> (xyxy in input pixels, confidence).

    Boxes come out in the accelerator's own input frame (192x320 here); map them
    back to the source image with whatever crop/letterbox the capture applied.
    """
    b, c, h, w = feat.shape
    assert c == 4 * reg_max + nc, f'expected {4 * reg_max + nc} channels, got {c}'
    anchors, strides = make_anchors(h, w, stride)

    flat = feat.reshape(b, c, -1)
    box, cls = flat[:, :4 * reg_max], flat[:, 4 * reg_max:]

    # DFL: expectation over the softmax of each side's reg_max bins
    box = box.reshape(b, 4, reg_max, -1).transpose(0, 2, 1, 3)
    box = (_softmax(box, axis=1) * np.arange(reg_max, dtype=np.float32).reshape(1, -1, 1, 1)).sum(1)

    lt, rb = box[:, :2], box[:, 2:]
    xyxy = np.concatenate((anchors - lt, anchors + rb), 1) * strides
    return xyxy.transpose(0, 2, 1), (1.0 / (1.0 + np.exp(-cls))).transpose(0, 2, 1)


def decode_confident(raw_nhwc, scale, bias, conf_thr, stride=STRIDE, reg_max=REG_MAX, nc=NC):
    """Accelerator output -> (xyxy (N,4), confidence (N,)) for cells >= conf_thr only.

    Same result as `dequantize` + `decode` + threshold, but the class channel is
    thresholded first and the DFL runs only on the survivors (~10 of 960 cells):
    10.8 ms -> 1.7 ms on the A53. Confidence is monotonic in the class logit, so
    nothing above the threshold is lost, and the tracker drops everything below
    its own CONF_THR anyway. Checked against `decode` on the 60-frame golden set
    at thresholds 0.25 / 0.05 / 0.001: identical cells, identical confidences,
    boxes within 3.1e-5 px (float32 summation order).
    """
    _, h, w, c = raw_nhwc.shape
    assert c == 4 * reg_max + nc, f'expected {4 * reg_max + nc} channels, got {c}'
    x = np.asarray(raw_nhwc, dtype=np.float32).reshape(h * w, c)
    k = 4 * reg_max                                          # first class channel
    cls = x[:, k] * scale[k] + bias[k]
    conf = 1.0 / (1.0 + np.exp(-cls))
    idx = np.nonzero(conf >= conf_thr)[0]

    box = x[idx, :k] * scale[:k] + bias[:k]                  # (N, 64)
    box = _softmax(box.reshape(-1, 4, reg_max), axis=2)
    box = (box * np.arange(reg_max, dtype=np.float32)).sum(2)  # (N, 4) l, t, r, b
    anchors = make_anchors(h, w, stride)[0][0].T[idx]        # (N, 2)
    xyxy = np.concatenate((anchors - box[:, :2], anchors + box[:, 2:]), 1) * stride
    return xyxy, conf[idx]


def nms(boxes, scores, iou_thr=0.45):
    """Greedy NMS. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, int)


def postprocess(raw_nhwc, scale, bias, conf_thr=0.25, iou_thr=0.45, stride=STRIDE):
    """Accelerator output -> (boxes xyxy, scores), both sorted by score."""
    boxes, scores = decode_confident(raw_nhwc, scale, bias, conf_thr, stride)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep]
