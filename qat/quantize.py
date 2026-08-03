"""Convert a trained float YOLOv5n (ReLU variant) into a Brevitas INT8 QAT model.

Phase 4. Input is `runs/train/relu_more_data_5/weights/best.pt` — the ReLU model,
NOT the SiLU one: pasting ReLU into SiLU-trained weights collapses the model to
mAP50 0.078 (measured), so the activation change must already be absorbed.

What gets quantized
  - every `Conv` block's conv  -> QuantConv2d, int8 per-tensor weights
  - every `Conv` block's act    -> QuantReLU, uint8 (unsigned: ReLU output >= 0)
  - the 3 `Detect.m` convs      -> QuantConv2d (on-chip output layer)
  - the network input           -> `input_quant` on the FIRST conv only

Left as float, deliberately
  - SPPF's MaxPool2d and the 2 nearest-neighbour Upsamples: both commute with
    monotonic quantization, so a wrapper would add nothing.
  - Detect's decode (sigmoid + grid arithmetic) and NMS: off-chip on the ARM
    side, so they are outside the accelerated graph by design.
  - the C3 residual adds and the 4 Concats: see SCALE ALIGNMENT below.

SCALE ALIGNMENT (known gap, not an oversight)
  This is fake-quantization QAT: every quantizer returns a plain float tensor
  (`return_quant_tensor=False`), so `torch.cat` and the `x + ...` residual in
  Bottleneck operate on dequantized floats and need no code change. That is what
  makes the surgery tractable and keeps YOLOv5's forward untouched.
  FINN, however, needs the operands of an add/concat to share a scale. Resolving
  that means QuantEltwiseAdd / QuantCat with tied quantizers and QuantTensor
  plumbing through Bottleneck.forward and Concat.forward. Deferred to the export
  phase, where FINN's streamlining requirements are actually testable — doing it
  now would be guessing at constraints we cannot yet run.

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

    return {'act_sites': n_sites, 'shared_acts_split': n_shared,
            'quant_conv': n_conv, 'quant_relu': n_act}


def load_and_quantize(weights, weight_bit_width=8, act_bit_width=8, device='cpu'):
    """Load a float YOLOv5 checkpoint and return (quant_model, stats)."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()
    stats = quantize_yolov5(model, weight_bit_width, act_bit_width)
    return model.to(device), stats
