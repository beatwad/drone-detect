"""Drive FINN's dataflow build on an exported QONNX model. Runs INSIDE FINN's Docker.

  cd <finn-repo> && ./run-docker.sh build_custom /path/to/drone-detect/export

FINN's `build_custom` expects the entry point to be named after the directory it
is given, so this file is invoked as the build script for `export/`.

TWO MODES, and the default is the important one:

  --estimate (default)  frontend + streamlining + HW layer conversion + resource
                        ESTIMATES. Minutes, and needs no synthesis.
  --bitfile             the full flow through Vivado to a .bit + PYNQ driver.
                        Hours.

Estimate mode is the decisive experiment: everything we care about surfaces in
`step_streamline` / `step_convert_to_hw`, long before synthesis. Do NOT sit
through a bitfile build to find out.

WHAT IS ACTUALLY BEING TESTED (revised 2026-08-04). It is NOT "can FINN compile
branched topologies" -- it can; branched YOLOv8n was built through FINN on a
ZCU102 (arXiv 2503.13023). The question is whether streamlining survives OUR
joins, and that is decided by the checkpoint, not by this script:

  PRECONDITION -- the checkpoint must have TIED QUANTISATION SCALES on the
  branches entering each join. `MoveLinearPastEltwiseAdd` matches
  `(x*C) + (y*C) -> (x + y) * C` and guards on `np.array_equal(init0, init1)`;
  `a*x + b*y` does not factor unless a == b. Untied scales fail by construction,
  so exporting an old checkpoint here tests nothing.

  pico            0 joins  -> control. A failure here is NOT about joins.
  yolov5n         20 joins -> the real run (7 Add + 13 Concat)

A yolov5n failure means streamlining stalled at a join, NOT that branches are
unsupported and pico wins. Read the last surviving intermediate model: leftover
Mul nodes feeding an Add/Concat name the offending join. If both pass,
`configs/yolov5n_eighth.yaml` (446k params, full FPN, multi-scale) is the better
model -- it beats pico on every regime and was conditional on exactly this.

KNOWN RISKS, untestable until FINN is installed:
  - **Concat joins have no upstream streamlining path.** WRITTEN 2026-08-04:
    `finn_transforms.MoveMulPastJoinConcat`, spliced in after `step_streamline`
    by `with_concat_step()`. It is NOT the ~10-line subclass of
    `MoveIdenticalOpPastJoinOp` we expected -- that base is hardcoded binary
    (SPPF's Concat has 4 inputs), has no value guard, and derives the join output
    shape from an input shape, which is false for Concat. Standalone transform
    instead, self-tested against qonnx (6/6, incl. numerical equivalence and two
    must-not-fire cases). Still unverified against a real streamlined graph.
  - Keep weights ON-CHIP (`mem_mode` != external). External mode cost the ARC
    paper ~100k LUT of DMA + interconnect, and FINN v0.10.1's release notes warn
    of "unexpected behaviour for external mem mode". 32.1 Mbit is enough.
  - The exported graph ends at a float raw conv output (Detect's decode is host
    side by design). FINN may want the final layer quantized; if it complains,
    add an output QuantIdentity in qat/quantize.py rather than changing this file.
  - The input carries a Quant node from `input_quant=Uint8ActPerTensorFloat`
    (scale 1/255), so uint8 pixels feed straight in. If FINN's frontend objects,
    the fix is a preprocessing step in the graph, not a different export.
  - Step names moved across FINN versions (`step_convert_to_hls` ->
    `step_convert_to_hw`). This uses FINN's own step lists rather than hardcoding
    them, so it should track whatever version is installed.
"""
import argparse
import os
import sys
from pathlib import Path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as cfg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # finn_transforms.py sits next to this file


def step_move_mul_past_concat(model, build_cfg):
    """Custom step: unblock the 13 Concat joins, then let streamlining finish.

    `step_streamline` turns Quant nodes into Mul/Add pairs, leaving a Mul on each
    branch entering a Concat. Upstream FINN has no Mul-past-Concat rewrite (see
    finn_transforms.py), so those Muls sit there and block absorption. Run ours,
    then re-run Streamline so the freed Muls get absorbed into the neighbouring
    layers exactly as they would on a join-free path.
    """
    from finn.transformation.streamline import Streamline
    from qonnx.transformation.infer_datatypes import InferDataTypes

    from finn_transforms import MoveMulPastJoinConcat

    model = model.transform(MoveMulPastJoinConcat())
    model = model.transform(Streamline())
    model = model.transform(InferDataTypes())
    return model


def with_concat_step(base_steps):
    """Splice our step in right after step_streamline.

    Found by NAME rather than index: the step lists differ between the estimate
    and bitfile flows and have been renamed across FINN versions.
    """
    steps = list(base_steps)
    try:
        i = steps.index('step_streamline')
    except ValueError:                  # renamed upstream -- fail loudly, not silently
        raise SystemExit('step_streamline not found in FINN step list: '
                         f'{[s for s in steps if isinstance(s, str)]}')
    steps.insert(i + 1, step_move_mul_past_concat)
    return steps

# ZCU102 = XCZU9EG-2FFVB1156. Deliberately oversized for development; do not let
# its headroom drive folding decisions we cannot afford on the deployment part.
FPGA_PART = 'xczu9eg-ffvb1156-2-e'
BOARD = 'ZCU102'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default=str(HERE / 'pico_w8a8_416_calibrated.onnx'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--bitfile', action='store_true',
                    help='full synthesis (hours). Default is estimate-only (minutes).')
    ap.add_argument('--target-fps', type=int, default=30)
    ap.add_argument('--clk-ns', type=float, default=10.0, help='10.0 ns = 100 MHz')
    args = ap.parse_args()

    onnx = Path(args.onnx)
    assert onnx.exists(), f'missing {onnx} -- run export_qonnx.py first'
    out = Path(args.out or HERE / f'finn_build_{onnx.stem}{"_bit" if args.bitfile else "_est"}')

    if args.bitfile:
        steps = with_concat_step(cfg.default_build_dataflow_steps)
        outputs = [cfg.DataflowOutputType.ESTIMATE_REPORTS,
                   cfg.DataflowOutputType.STITCHED_IP,
                   cfg.DataflowOutputType.BITFILE,
                   cfg.DataflowOutputType.PYNQ_DRIVER,
                   cfg.DataflowOutputType.DEPLOYMENT_PACKAGE]
    else:
        steps = with_concat_step(cfg.estimate_only_dataflow_steps)
        outputs = [cfg.DataflowOutputType.ESTIMATE_REPORTS]

    build_cfg = cfg.DataflowBuildConfig(
        output_dir=str(out),
        target_fps=args.target_fps,
        synth_clk_period_ns=args.clk_ns,
        fpga_part=FPGA_PART,
        board=BOARD,
        shell_flow_type=cfg.ShellFlowType.VIVADO_ZYNQ,
        steps=steps,
        generate_outputs=outputs,
        # keep every intermediate .onnx: if a join blows up, the last surviving
        # graph is the diagnostic and re-running to reproduce is expensive
        save_intermediate_models=True,
    )

    print(f'model    : {onnx}')
    print(f'mode     : {"BITFILE (hours)" if args.bitfile else "ESTIMATE ONLY (minutes)"}')
    print(f'part     : {FPGA_PART}  target {args.target_fps} FPS @ {1000 / args.clk_ns:.0f} MHz')
    print(f'output   : {out}\n')

    build.build_dataflow_cfg(str(onnx), build_cfg)

    rpt = out / 'report'
    if rpt.exists():
        print(f'\nreports in {rpt}:')
        for f in sorted(rpt.glob('*.json')):
            print(f'  {f.name}')
        print('\n  estimate_layer_resources.json -> per-layer BRAM/LUT/DSP; the '
              'total is what to compare against the 32.1 Mbit ZCU102 budget')
        print('  estimate_network_performance.json -> expected FPS / latency')


if __name__ == '__main__':
    main()
