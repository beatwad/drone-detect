"""Convert a trained float YOLOv5n (ReLU variant) into a Brevitas INT8 QAT model.

Phase 4. Input is `runs/train/relu_more_data_5/weights/best.pt` — the ReLU model,
NOT the SiLU one: pasting ReLU into SiLU-trained weights collapses the model to
mAP50 0.078 (measured), so the activation change must already be absorbed.

What gets quantized
  - every `Conv` block's conv  -> QuantConv2d, int8 per-tensor weights
  - every `Conv` block's act    -> QuantReLU, uint8 (unsigned: ReLU output >= 0)
  - the 3 `Detect.m` convs      -> QuantConv2d (on-chip output layer)
  - the network input           -> `input_quant` on the FIRST conv only

  - every join (7 residual adds, 13 concats) -> shared-scale quantizer, below

Left as float, deliberately
  - SPPF's MaxPool2d and the 2 nearest-neighbour Upsamples: both commute with
    monotonic quantization, so a wrapper would add nothing.
  - Detect's decode (sigmoid + grid arithmetic) and NMS: off-chip on the ARM
    side, so they are outside the accelerated graph by design.

SCALE ALIGNMENT AT JOINS — implemented 2026-08-04, and it is a PRECONDITION
  Earlier this was deferred on the grounds that FINN's requirements were not yet
  testable. That was wrong in one direction: the requirement is not a FINN
  implementation detail, it is algebra, and it is decided here in QAT rather than
  at export. FINN's `MoveLinearPastEltwiseAdd` matches

      (x * C) + (y * C)  ->  (x + y) * C

  and guards on `np.array_equal(init0, init1)`. `a*x + b*y` does not factor
  unless `a == b`, so no toolchain version avoids it. A checkpoint whose two
  branches carry independent scales exports Muls that will never streamline —
  meaning an export from an untied checkpoint tests nothing.

  Fix: every join gets ONE quantizer instance applied to all of its operands —
  see `SharedQuant`. Tying is by object identity, so the branches cannot drift
  apart during QAT. (Brevitas ships `QuantCat`/`QuantEltwiseAdd` for this, but
  0.13.0's QuantCat is broken — see SharedQuant's docstring.)

  Only 4 of the 20 joins are actual modules (`Concat`). The other 16 are inline
  `torch.cat` / `x + ...` inside `C3.forward`, `SPPF.forward` and
  `Bottleneck.forward`, so they need wrapper modules rather than substitution.
  The wrappers below keep the original child NAMES (cv1/cv2/cv3/m), so state_dict
  keys and YOLOv5's `m.f`/`m.i` routing are unchanged.

  Uint8 (unsigned) is correct for every join here: all operands are post-ReLU.

The shared-activation trap
  YOLOv5 assigns ONE `Conv.default_act` instance to all 57 Conv blocks, so the
  float model reports `ReLU: 1` in a module histogram. QuantReLU learns a
  per-instance activation scale, so converting in place would tie all 57 layers
  to a single scale. `_split_shared_acts` gives every site its own object first.
  Same trap that makes `.modules()` under-report; walk `named_children()`.
"""
import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

import sys
from pathlib import Path

# APPEND, never insert(0). yolov5/ contains its own top-level `export.py`, which
# shadows this repo's `export/` PACKAGE if yolov5 lands ahead of the repo root on
# sys.path -- and this module is imported by export/verify_qonnx.py, which then
# fails with "'export' is not a package". Appending keeps the repo root first.
_YOLOV5 = str(Path(__file__).resolve().parents[1] / 'yolov5')
if _YOLOV5 not in sys.path:
    sys.path.append(_YOLOV5)
from models.common import C3, Concat, SPPF, Bottleneck  # noqa: E402


class SharedQuant(nn.Module):
    """ONE quantizer applied to every operand of a join, then a plain cat/add.

    Not `QuantCat`/`QuantEltwiseAdd`. Two reasons:
      1. brevitas 0.13.0's QuantCat is broken -- `QuantCat.forward` calls
         `QuantTensor.cat`, which does not exist on that class (it lives on
         `IntQuantTensor`). Verified on the installed version.
      2. We do not need QuantTensor plumbing at all. In fake-quant mode the
         graph we want FINN to see is just

             Quant(a, S) -> Concat/Add <- Quant(b, S)

         i.e. one Quant node per branch carrying the SAME scale S. Mapping a
         single QuantIdentity over the operands produces exactly that, using the
         most basic Brevitas primitive and its well-trodden export handler,
         rather than depending on QuantCat's.

    The tying is by object identity: `self.q` is one module, so every operand
    shares its learned scale by construction and cannot drift apart during QAT.
    """

    def __init__(self, act_bit_width):
        super().__init__()
        self.q = qnn.QuantIdentity(
            act_quant=Uint8ActPerTensorFloat, bit_width=act_bit_width,
            return_quant_tensor=False)

    def cat(self, tensors, dim=1):
        return torch.cat([self.q(t) for t in tensors], dim)

    def add(self, a, b):
        return self.q(a) + self.q(b)


def _adopt(dst, src):
    """Carry YOLOv5's routing metadata across a module swap.

    `parse_model` stamps .f (input indices), .i (layer index), .type and .np onto
    every top-level layer, and `BaseModel._forward_once` reads .f/.i on every
    step. Dropping them silently breaks the graph.
    """
    for attr in ('f', 'i', 'type', 'np'):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


class QuantBottleneck(nn.Module):
    """Bottleneck with the residual add tied to a shared scale (7 sites)."""

    def __init__(self, b, act_bit_width):
        super().__init__()
        self.cv1, self.cv2, self.add = b.cv1, b.cv2, b.add
        self.q_add = SharedQuant(act_bit_width) if b.add else None

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return self.q_add.add(x, y) if self.add else y


class QuantC3(nn.Module):
    """C3 with its internal concat tied to a shared scale (8 sites).

    cv1/cv2 are two convs on ONE input -- a plain fork, which FINN already
    handles via duplicatestreams.py. Only the rejoin needs work.
    """

    def __init__(self, c3, act_bit_width):
        super().__init__()
        self.cv1, self.cv2, self.cv3, self.m = c3.cv1, c3.cv2, c3.cv3, c3.m
        self.q_cat = SharedQuant(act_bit_width)

    def forward(self, x):
        return self.cv3(self.q_cat.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class QuantSPPF(nn.Module):
    """SPPF with its 4-way concat tied to a shared scale (1 site)."""

    def __init__(self, sppf, act_bit_width):
        super().__init__()
        self.cv1, self.cv2, self.m = sppf.cv1, sppf.cv2, sppf.m
        self.q_cat = SharedQuant(act_bit_width)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.q_cat.cat([x, y1, y2, self.m(y2)], dim=1))


class QuantConcat(nn.Module):
    """The 4 neck Concats -- the only joins that are already modules."""

    def __init__(self, c, act_bit_width):
        super().__init__()
        self.d = c.d
        self.q_cat = SharedQuant(act_bit_width)

    def forward(self, x):
        return self.q_cat.cat(list(x), dim=self.d)


def tie_join_scales(model, act_bit_width=8):
    """Give every join one quantizer shared across its operands.

    Runs AFTER conv/act replacement: the wrappers adopt the existing children by
    reference, so the convs they hold are already quantized.
    """
    targets = [(p, n, c) for p in model.modules()
               for n, c in p.named_children()
               if isinstance(c, (Bottleneck, C3, SPPF, Concat))]
    counts = {'add': 0, 'c3_cat': 0, 'sppf_cat': 0, 'neck_cat': 0}
    for parent, name, mod in targets:                 # mutate after the walk
        if isinstance(mod, Bottleneck):
            new = QuantBottleneck(mod, act_bit_width)
            counts['add'] += int(mod.add)
        elif isinstance(mod, C3):
            new = QuantC3(mod, act_bit_width)
            counts['c3_cat'] += 1
        elif isinstance(mod, SPPF):
            new = QuantSPPF(mod, act_bit_width)
            counts['sppf_cat'] += 1
        else:
            new = QuantConcat(mod, act_bit_width)
            counts['neck_cat'] += 1
        setattr(parent, name, _adopt(new, mod))
    counts['joins_total'] = sum(counts.values())
    return counts


def _split_shared_acts(model):
    """Give every activation site its own module instance.

    Returns the number of sites that were sharing an object. Must run before
    conversion — see the module docstring.
    """
    sites = [(parent, name, child)
             for parent in model.modules()
             for name, child in parent.named_children()
             if isinstance(child, (nn.ReLU, nn.SiLU))]
    seen, n_shared = set(), 0
    for parent, name, child in sites:
        if id(child) in seen:
            setattr(parent, name, type(child)())
            n_shared += 1
        else:
            seen.add(id(child))
    return len(sites), n_shared


def _quant_conv(conv, weight_bit_width, input_quant=None):
    """QuantConv2d carrying `conv`'s weights (and bias, if any)."""
    q = qnn.QuantConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None,
        weight_quant=Int8WeightPerTensorFloat, weight_bit_width=weight_bit_width,
        input_quant=input_quant, return_quant_tensor=False)
    q.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        q.bias.data.copy_(conv.bias.data)
    return q


def quantize_yolov5(model, weight_bit_width=8, act_bit_width=8):
    """In-place: float YOLOv5n(ReLU) -> Brevitas fake-quantized QAT model.

    The module tree keeps its exact shape and layer indices, so YOLOv5's `m.f`
    routing, the loss, the dataloader and val.py all keep working unchanged.
    """
    n_sites, n_shared = _split_shared_acts(model)

    # collect first: replacing children while walking the tree re-visits the
    # freshly inserted modules (and QuantReLU contains no nn.ReLU, but
    # QuantConv2d does contain submodules -- so mutate only after the walk).
    convs = [(p, n, c) for p in model.modules()
             for n, c in p.named_children() if isinstance(c, nn.Conv2d)]
    acts = [(p, n, c) for p in model.modules()
            for n, c in p.named_children() if isinstance(c, nn.ReLU)]

    # the first conv in execution order also quantizes the network input
    first_conv = model.model[0].conv
    n_conv = 0
    for parent, name, conv in convs:
        iq = Uint8ActPerTensorFloat if conv is first_conv else None
        setattr(parent, name, _quant_conv(conv, weight_bit_width, input_quant=iq))
        n_conv += 1

    n_act = 0
    for parent, name, _ in acts:
        setattr(parent, name, qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, bit_width=act_bit_width,
            return_quant_tensor=False))
        n_act += 1

    joins = tie_join_scales(model, act_bit_width)

    return {'act_sites': n_sites, 'shared_acts_split': n_shared,
            'quant_conv': n_conv, 'quant_relu': n_act, **joins}


def load_and_quantize(weights, weight_bit_width=8, act_bit_width=8, device='cpu'):
    """Load a float YOLOv5 checkpoint and return (quant_model, stats)."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()
    stats = quantize_yolov5(model, weight_bit_width, act_bit_width)
    return model.to(device), stats
