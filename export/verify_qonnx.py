"""Check an exported QONNX graph against the PyTorch model that produced it.

A .onnx that writes cleanly but computes something else is worse than no export
at all -- it would send us into FINN debugging a model mismatch. So: run
qonnx's own executor (onnxruntime cannot execute Quant nodes) on the same input
and compare.

Also runs qonnx's `cleanup_model`, which is the first thing FINN's frontend does.
If the graph does not survive cleanup it will not survive FINN.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov5'))
sys.path.insert(0, str(ROOT))

from qat.quantize import load_and_quantize  # noqa: E402
from export.export_qonnx import RawHead  # noqa: E402


def onnx_module_path(name):
    """ONNX Quant node name -> the PyTorch module path it came from.

    torch.onnx emits a scope per nesting level, and for indexed containers it
    emits BOTH the container and the indexed child:
        /model.4/m/m.0/cv1/act/act_quant/...  ->  model.4.m.0.cv1.act
        /model.0/act/act_quant/...            ->  model.0.act
    so collapse any segment whose stem repeats the previous segment.
    """
    segs = name.strip('/').split('/')
    if 'act_quant' in segs:
        segs = segs[:segs.index('act_quant')]
    out = []
    for s in segs:
        if out and '.' in s and s.rsplit('.', 1)[0] == out[-1]:
            out[-1] = s
        else:
            out.append(s)
    return '.'.join(out)


def decode(raw, detect, imgsz):
    """Detect's post-conv maths, applied to an already-convolved feature map.

    Detect.forward cannot be reused directly because its first act is to run the
    output conv, which has already happened here. This is that function minus
    that one line (v7.0, boxes-only branch).
    """
    z = []
    for i, r in enumerate(raw):
        bs, _, ny, nx = r.shape
        x = r.view(bs, detect.na, detect.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        grid, anchor_grid = detect._make_grid(nx, ny, i)
        xy, wh, conf = x.sigmoid().split((2, 2, detect.nc + 1), 4)
        xy = (xy * 2 + grid) * detect.stride[i]
        wh = (wh * 2) ** 2 * anchor_grid
        z.append(torch.cat((xy, wh, conf), 4).view(bs, -1, detect.no))
    return torch.cat(z, 1)


def detection_check(ref, onnx_model, args, n_images=32):
    """Do the two graphs actually detect the same things on real images?

    This is the criterion that matters, and it exists because the raw-logit
    tolerance is unfalsifiable: a 1% difference on a pre-sigmoid logit may move
    every box or none of them. Raw-tensor agreement is a proxy; box agreement is
    the requirement. Compares post-NMS detections at the operating threshold.
    """
    from qonnx.core.onnx_exec import execute_onnx
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression
    import cv2

    # NMS is DISCONTINUOUS in confidence: a box at 0.2531 is kept and the same box
    # at 0.2497 is dropped. Two implementations differing by ~1 LSB will therefore
    # always disagree on boxes sitting at the cutoff, and requiring identical box
    # COUNTS makes the check fail for a reason that has nothing to do with export
    # fidelity. (Observed: n_eighth, one box at conf 0.2531 vs 0.2497 -- while the
    # confident box in the same image matched to 0.7783 vs 0.7781.)
    # So: boxes at conf >= CONF + MARGIN must agree exactly; boxes inside the
    # margin are counted and reported, not failed on.
    CONF, MARGIN = 0.25, 0.05
    detect = ref._detect
    paths = [l.strip() for l in (ROOT / 'configs/val_close.txt').read_text().splitlines() if l.strip()]
    paths = paths[:n_images]
    inp = onnx_model.graph.input[0].name

    n_t = n_o = matched = n_border = 0
    dmax = dconf = 0.0
    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            continue
        im = letterbox(im0, args.imgsz, stride=32, auto=False)[0]
        x = torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]

        with torch.no_grad():
            rt = ref(x)
        rt = [rt] if not isinstance(rt, list) else rt
        oc = execute_onnx(onnx_model, {inp: x.numpy()})
        ro = [torch.from_numpy(oc[o.name]) for o in onnx_model.graph.output]

        dt = non_max_suppression(decode(rt, detect, args.imgsz), CONF, 0.45)[0]
        do = non_max_suppression(decode(ro, detect, args.imgsz), CONF, 0.45)[0]
        n_border += int((dt[:, 4] < CONF + MARGIN).sum() + (do[:, 4] < CONF + MARGIN).sum())
        dt, do = dt[dt[:, 4] >= CONF + MARGIN], do[do[:, 4] >= CONF + MARGIN]
        n_t += len(dt)
        n_o += len(do)
        for a in dt:                       # nearest-centre pairing, 2 px tolerance
            if not len(do):
                break
            ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
            cb = torch.stack([(do[:, 0] + do[:, 2]) / 2, (do[:, 1] + do[:, 3]) / 2], 1)
            d, j = torch.hypot(cb[:, 0] - ca[0], cb[:, 1] - ca[1]).min(0)
            d = d.item()
            dmax = max(dmax, d)
            if d <= 2.0:
                matched += 1
                dconf = max(dconf, abs(a[4].item() - do[j, 4].item()))

    # A model that detects nothing agrees with itself trivially, so zero boxes is
    # INCONCLUSIVE, not a pass -- that is exactly the state an uncalibrated export
    # is in, and it must not be able to certify itself.
    agree = n_t == n_o and matched == n_t
    verdict = 'OK' if agree and n_t else ('INCONCLUSIVE (no detections)' if not n_t else 'FAIL')
    print(f'\ndetection check ({len(paths)} real close-regime images, '
          f'conf >= {CONF + MARGIN:.2f})')
    print(f'  torch {n_t} boxes | onnx {n_o} boxes | centre-matched {matched}  '
          f'max centre delta {dmax:.3f} px  max conf delta {dconf:.4f}  {verdict}')
    print(f'  ({n_border} near-threshold box(es) in [{CONF:.2f}, {CONF + MARGIN:.2f}) '
          f'excluded — NMS is discontinuous there)')
    return agree and n_t > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--ckpt', default=None,
                    help='MUST match what was passed to export_qonnx.py -- otherwise the '
                         'reference is the uninitialised model and everything mismatches')
    ap.add_argument('--wbw', type=int, default=4)
    ap.add_argument('--abw', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=416)
    args = ap.parse_args()

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.util.cleanup import cleanup_model

    from qonnx.transformation.infer_shapes import InferShapes

    # Two graphs on purpose. cleanup_model renames every node to Quant_0/Quant_1/...
    # which destroys the module paths the per-layer check matches on, so the
    # layer comparison runs on the NAMED graph (shape-inferred only -- execute_onnx
    # refuses a graph with unspecified shapes). The cleaned graph is what FINN
    # actually consumes, so the detection check and the saved file use that one.
    named = ModelWrapper(args.onnx).transform(InferShapes())
    model = ModelWrapper(args.onnx)
    n_before = len(model.graph.node)
    model = cleanup_model(model)
    n_after = len(model.graph.node)
    clean = Path(args.onnx).with_name(Path(args.onnx).stem + '_clean.onnx')
    model.save(str(clean))
    print(f'cleanup    : {n_before} -> {n_after} nodes, saved {clean.name}')

    torch.manual_seed(0)
    x = torch.rand(1, 3, args.imgsz, args.imgsz)

    ref, _ = load_and_quantize(args.weights, args.wbw, args.abw, device='cpu')
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        ref.load_state_dict(ck['state_dict'])
    ref._detect = ref.model[-1]          # kept for decode() in detection_check
    ref.model[-1] = RawHead(ref.model[-1])
    ref.eval()
    with torch.no_grad():
        y = ref(x)
    y = [y] if not isinstance(y, list) else y

    inp = named.graph.input[0].name
    ctx = execute_onnx(named, {inp: x.numpy()}, return_full_exec_context=True)
    got = [ctx[o.name] for o in named.graph.output]

    # ACCEPTANCE CRITERION -- why it is not a plain allclose.
    #
    # PyTorch and the QONNX executor round in opposite directions for a value
    # sitting exactly on a quantizer boundary, so a few elements differ by
    # exactly one LSB (= one quantization step). That is inherent to comparing
    # two implementations of the same quantized graph; an atol of 1e-4 fails a
    # perfectly correct export.
    #
    # Nor is "output error in units of the input LSB" the right yardstick: the
    # output conv sums 64 channels of 4-bit weights, so it legitimately amplifies
    # a 1-LSB input flip into several LSB of output. Normalising that way just
    # invites fudging the tolerance upward until it passes.
    #
    # What a correct export actually guarantees is that error stays BOUNDED at
    # each quantizer instead of compounding down the network. A structurally
    # wrong graph diverges multiplicatively and cannot fake this. So: check every
    # quantizer output in LSBs, and check the final output in relative terms.
    # Three checks, because no one of them is sufficient alone:
    #   INTEGER  -- two quantized tensors sharing a scale can only differ by whole
    #               LSBs, so a fractional result means the SCALES disagree. Note
    #               this proves scale agreement, NOT that the maths is right: a
    #               badly wrong graph also yields integer differences.
    #   RANGE    -- magnitude as a fraction of the quantizer's OWN full scale
    #               (2^abw - 1 levels), not an absolute LSB count. An absolute cap
    #               is depth-blind: error accumulates down the network, so pico
    #               (10 layers, 0 joins) peaks at 2 LSB while yolov5n (57 layers,
    #               20 joins) reaches 16 -- and a cap that admits both would need a
    #               different value per model, i.e. it would test nothing.
    #               As a fraction of range those are 0.8% and 6.3%: both plainly
    #               "the same tensor", which is the claim being checked.
    #
    # The hard gates are INTEGER (scales agree) and the detection check (behaviour
    # agrees). RANGE is a coarse sanity bound on top -- divergence is not subtle
    # and lands orders of magnitude out, not at 6%.
    RANGE_TOL, INT_TOL = 0.10, 0.05
    import brevitas.nn as qnn

    quants = [(n, m) for n, m in ref.named_modules() if isinstance(m, qnn.QuantReLU)]
    caught = []
    for n, mod in quants:
        mod._nm = n
        mod.register_forward_hook(
            lambda s, i, o: caught.append((s._nm, float(s.act_quant.scale()), o.detach().numpy())))
    with torch.no_grad():
        ref(x)

    # Match by NAME, not shape. Shape matching worked on pico (53 nodes) but fell
    # apart on yolov5n (317 nodes): C3 blocks put cv1/cv2/cv3 and every Bottleneck
    # at identical spatial dims, so "closest same-shape tensor" routinely paired
    # against the wrong layer and reported fractional-LSB garbage. Brevitas names
    # its Quant nodes after the PyTorch module path, so the mapping is exact.
    pool = {onnx_module_path(n.name): n.output[0]
            for n in named.graph.node if n.op_type == 'Quant' and 'act_quant' in n.name}
    levels = 2 ** args.abw - 1
    print(f'\nper-quantizer error (integer LSBs, and % of the {levels}-level range; '
          f'tolerance {RANGE_TOL:.0%})')
    worst, ok = 0.0, True
    for n, lsb, arr in caught:
        if n not in pool or pool[n] not in ctx:
            print(f'  {n:<22} {str(arr.shape):>20}  NOT FOUND in graph — skipped')
            ok = False
            continue
        n_lsb = np.abs(arr - ctx[pool[n]]).max() / lsb
        worst = max(worst, n_lsb)
        # No sub-LSB exemption: now that matching is by name the correspondence is
        # exact, so a fractional result really does mean the scales disagree.
        # (An earlier exemption existed only to paper over the shape-matching
        # heuristic that this replaced.)
        frac = abs(n_lsb - round(n_lsb))
        rng = n_lsb / levels
        good = rng <= RANGE_TOL and frac <= INT_TOL
        ok &= good
        why = '' if good else (' scale mismatch' if frac > INT_TOL else ' too large')
        print(f'  {n:<22} {str(arr.shape):>20}  {n_lsb:6.2f} LSB  {rng:6.2%}  '
              f'{"OK" if good else "FAIL" + why}')

    print('\nfinal output (raw logits — diagnostic only, see detection check below)')
    for i, (t, a) in enumerate(zip(y, got)):
        t = t.numpy()
        d = np.abs(t - a)
        rel = d.max() / (np.abs(t).max() + 1e-12)
        print(f'  out[{i}] range [{t.min():.2f}, {t.max():.2f}]  max|diff| {d.max():.3e}  '
              f'rel {rel:.3%}  mean {d.mean():.3e}')

    ok &= detection_check(ref, model, args)

    note = ('bounded => rounding, not divergence' if worst / levels <= RANGE_TOL else
            'HUGE => different models, not rounding; check --ckpt matches the export')
    print(f'\nworst quantizer : {worst:.2f} LSB = {worst / levels:.2%} of range ({note})')
    print(f'verdict         : {"QONNX graph is faithful" if ok else "DIVERGENT — do not proceed"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
