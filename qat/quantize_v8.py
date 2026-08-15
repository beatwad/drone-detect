"""Convert a trained float YOLOv8n-P3 (ReLU6) into a Brevitas QAT model.

Target is the quantization the ZCU102 reference bitstream was actually built
with, read off `quantyolov8_4w4a_comact_tidy.onnx` rather than guessed — see
build_notes §10.8/§10.9. Input is `configs/yolov8n_p3_relu6.yaml` trained to
`runs/train/v8n_p3_relu62`, i.e. a net whose activations already live in [0, 6].

The recipe, all measured
  weights      per-CHANNEL float scale (41 per-channel Muls in the reference,
               one per conv). Note our yolov5 pipeline uses per-tensor; this is
               a real difference, not an oversight.
  W8 / A8      the stem conv and the six Detect convs (7 of 41)
  W4 / A4      the other 34 convs
  activations  unsigned with a FIXED, non-learned scale over [0, 6] — 6/15 at
               4 bits, 6/255 at 8 bits. Verified against the reference to eight
               decimals.
  input        NOT quantized here. FINN's `step_yolov8_tidy_up` merges a
               `ToTensor` preprocessing model (a Div by 255) in front of the
               graph and annotates the input UINT8 — exactly what the reference
               does, where `global_in` feeds a bare `Div` and carries no Quant
               node at all. Putting a Brevitas `input_quant` on the stem conv
               instead is FATAL: see the unsigned-Quant rule below.

THE UNSIGNED-QUANT RULE
  FINN has two activation handlers. QuantReluHandler claims predecessors
  {Relu, Selu}; QuantIdentityHandler claims {BatchNormalization, Sub, Add, Mul,
  Div, DebugMarker, None}, and `quant_act_to_multithreshold.py` falls back to it
  for anything neither claims — and it REJECTS unsigned quantizers with
  "FINN only supports signed Quant nodes for identity activations."
  So every unsigned Quant must sit immediately behind a Relu. Our 39 activation
  quantizers do (Conv -> BN -> Relu -> Quant). An input quantizer does not: its
  predecessor is the graph input, so it takes the identity path and the build
  dies in the very first step. Measured 2026-08-13.

WHY THERE IS NO SCALE TYING HERE
  `qat/quantize.py` spends most of its length on `SharedQuant`, because YOLOv5's
  joins need every branch on one scale before FINN will streamline them, and
  yolov5's scales are learned per layer. That whole problem is absent here: the
  reference uses ONE COMMON activation range for the entire network ("comact"),
  so every operand of every concat and every residual add is already at the same
  scale by construction. Nothing to tie. Do not port SharedQuant over.

  This is why the activation scale must be CONST and not merely initialised at
  6.0 — a learned scale would drift per site and silently reintroduce the
  problem.

WHAT IS DELIBERATELY LEFT ALONE
  - Residual adds and concats: not quantized. The reference has exactly one
    MultiThreshold per conv-block activation (34 + 5 = 39, and there are 39
    Conv blocks) and none on the joins, so a sum can carry twice the activation
    range into a concat. FINN handles the resulting mixed input datatypes —
    build_notes §10.2 records a live INT19+INT21 concat.
  - `Detect.dfl`: a fixed decode conv that the reference runs off-chip. It is
    not on the training forward path and must not be quantized or exported.
  - SPPF's MaxPool and the two nearest-neighbour Upsamples: monotonic, so
    quantization commutes with them.

THE SHARED-ACTIVATION TRAP, AGAIN
  Ultralytics assigns ONE `Conv.default_act` instance to every Conv block, the
  same way YOLOv5 does, so a module histogram reports `ReLU6: 1` for the whole
  net. QuantReLU carries per-instance state, so converting in place would leave
  all 39 sites sharing one module. `_split_shared_acts` gives each its own
  object first. Walk `named_children()`, never `modules()`.
"""
import types

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.inject.enum import ScalingImplType
from brevitas.quant import Int8WeightPerChannelFloat, Uint8ActPerTensorFloat

from ultralytics.nn.modules.head import Detect

ACT_MAX = 6.0     # the reference's common activation range, [0, ACT_MAX]


class ConstUintAct(Uint8ActPerTensorFloat):
    """Unsigned per-tensor activation with a FIXED scale = ACT_MAX / (2**b - 1).

    `ScalingImplType.CONST` makes the scale a constant rather than a Parameter,
    which is what keeps every site on one common scale. `Uint8ActPerTensorFloat`
    is subclassed only to inherit the solver wiring; its percentile-statistics
    attributes are unused once the scaling impl is CONST.
    """
    scaling_impl_type = ScalingImplType.CONST
    scaling_init = ACT_MAX


def _high_precision_ids(model):
    """ids of every submodule of the stem and of Detect — the W8/A8 region.

    Located by structure, not by name: `model.16` is Detect only for the P3
    config, and would be `model.22` for the full 3-head one.
    """
    layers = model.model
    high = [layers[0]] + [m for m in layers if isinstance(m, Detect)]
    return {id(sub) for top in high for sub in top.modules()}


def _split_shared_acts(model):
    """Give every activation site its own module instance.

    Returns (n_sites, n_that_were_sharing). Must run before conversion.
    """
    sites = [(parent, name, child)
             for parent in model.modules()
             for name, child in parent.named_children()
             if isinstance(child, (nn.ReLU6, nn.ReLU, nn.SiLU))]
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
        weight_quant=Int8WeightPerChannelFloat, weight_bit_width=weight_bit_width,
        input_quant=input_quant, return_quant_tensor=False)
    q.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        q.bias.data.copy_(conv.bias.data)
    return q


def quantize_yolov8(model, low_bits=4, high_bits=8):
    """In-place: float YOLOv8n-P3(ReLU6) -> Brevitas fake-quantized QAT model.

    `low_bits` is the network body, `high_bits` the stem and the Detect head.
    Pass low_bits=8 for the W8A8 baseline. The module tree keeps its exact shape
    and layer indices, so Ultralytics' `.f`/`.i` routing, the loss and the
    dataloaders all keep working.
    """
    n_sites, n_shared = _split_shared_acts(model)
    high = _high_precision_ids(model)

    dfl = {id(m) for layer in model.model if isinstance(layer, Detect)
           for m in layer.dfl.modules()}

    # Collect before mutating: replacing children mid-walk re-visits the freshly
    # inserted modules, and QuantConv2d does contain submodules.
    convs = [(p, n, c) for p in model.modules()
             for n, c in p.named_children()
             if isinstance(c, nn.Conv2d) and id(c) not in dfl]
    acts = [(p, n, c) for p in model.modules()
            for n, c in p.named_children() if isinstance(c, (nn.ReLU6, nn.ReLU))]

    counts = {'conv_w8': 0, 'conv_w4': 0, 'act_a8': 0, 'act_a4': 0}

    for parent, name, conv in convs:
        is_high = id(conv) in high
        bits = high_bits if is_high else low_bits
        setattr(parent, name, _quant_conv(conv, bits))
        counts['conv_w8' if is_high else 'conv_w4'] += 1

    for parent, name, act in acts:
        is_high = id(act) in high
        bits = high_bits if is_high else low_bits
        setattr(parent, name, qnn.QuantReLU(
            act_quant=ConstUintAct, bit_width=bits, return_quant_tensor=False))
        counts['act_a8' if is_high else 'act_a4'] += 1

    # DISABLE Conv+BN FUSION. Ultralytics fuses inside AutoBackend before it
    # validates or predicts, and `fuse_conv_and_bn` folds BN into the conv
    # weights while KEEPING the QuantConv2d type — so the per-channel weight
    # quantizer then re-quantizes a completely different distribution (BN scales
    # vary wildly per channel). Measured on this net: the output moves by ~104.
    #
    # It also measures a model we never build. Brevitas exports Quant -> Conv ->
    # BatchNormalization as separate nodes and FINN streamlines BN into the
    # thresholds AFTERWARDS — the reference graph still carries all 39 of its
    # BatchNormalization nodes against 41 Convs. The weights that reach the chip
    # are the UNFUSED quantized ones, so that is what has to be evaluated.
    model.fuse = types.MethodType(lambda self, verbose=True: self, model)

    return {'act_sites': n_sites, 'shared_acts_split': n_shared,
            'dfl_skipped': len(dfl), 'fuse_disabled': True, **counts}


def load_and_quantize(weights, low_bits=4, high_bits=8, device='cpu'):
    """Load a float Ultralytics checkpoint and return (quant_model, stats)."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()
    stats = quantize_yolov8(model, low_bits, high_bits)
    return model.to(device), stats
