"""Drive FINN's dataflow build on an exported QONNX model. Runs INSIDE FINN's Docker.

  cd ~/Repos/finn
  FINN_XILINX_PATH=/home/alex/Xilinx FINN_XILINX_VERSION=2022.2 \
  FINN_HOST_BUILD_DIR=/home/alex/finn_build \
  ./run-docker.sh build_custom /home/alex/Repos/drone-detect/export finn_build

`build_custom` runs `python -mpdb -cc -cq $3.py` inside the mounted directory,
where $3 DEFAULTS TO `build` -- so the third argument (`finn_build`, no .py) is
what names this file. It is not derived from the directory name.

Two things run-docker.sh gets wrong for this machine:
  - FINN_HOST_BUILD_DIR defaults to /tmp/..., and /tmp here is a 31 GB RAM-backed
    tmpfs. HLS + Vivado intermediates would be written to RAM. Always override.
  - the `-mpdb` wrapper drops into a post-mortem debugger on any exception, which
    hangs a non-interactive run. Give it a TTY (`script -qec ...`) or expect it
    to stall rather than exit on failure.

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

KNOWN RISKS (FINN installed 2026-08-05, so these are now testable):
  - **Concat joins have no upstream streamlining path.** WRITTEN 2026-08-04:
    `finn_transforms.MoveMulPastJoinConcat`, spliced in after `step_streamline`
    by `with_custom_steps()`. It is NOT the ~10-line subclass of
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
  - **The input Quant node DID break the frontend. FIXED 2026-08-05.** This was
    listed as a risk; it fired on the very first run. `step_qonnx_to_finn` raised
    "FINN only supports signed Quant nodes for identity activations", because
    QuantIdentityHandler takes any Quant with no predecessor and demands signed.
    Our `input_quant=Uint8ActPerTensorFloat` is unsigned by design (pixels are).
    Fixed by `step_input_quant_to_uint8`, which rewrites it the FINN way --
    `x_uint8 -> Mul(scale)` with the input tensor annotated UINT8. NOTE this
    changes the accelerator's INPUT CONTRACT: it now consumes integer codes
    `round(x/scale)`, not floats. Good for deployment (no float preprocessing on
    the ARM side), but host preprocessing must match.
    The canonical alternative is to drop `input_quant` in qat/quantize.py and
    re-export; deferred so the already-gated exports stay untouched.
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


def step_input_quant_to_uint8(model, build_cfg):
    """Custom step: make the input Quant node something FINN's frontend accepts.

    MEASURED 2026-08-05: without this, `step_qonnx_to_finn` dies immediately with
    "FINN only supports signed Quant nodes for identity activations". Our
    `input_quant=Uint8ActPerTensorFloat` leaves a Quant node fed by nothing, and
    FINN sends any Quant with no predecessor to QuantIdentityHandler, which
    requires signed. See finn_transforms.InputQuantToUintDtype.
    """
    from finn_transforms import InputQuantToUintDtype

    return model.transform(InputQuantToUintDtype())


def step_move_linear_past_fork(model, build_cfg):
    """Custom step: resolve Mul-forks before Streamline trips over them.

    MEASURED 2026-08-05. `Streamline()` runs `MoveScalarLinearPastInvariants`,
    which moves a scalar Mul past a nearest-neighbour Resize. In FINN v0.10.1 it
    does NOT check whether the Mul is a fork node -- it rewires the Mul's output
    tensor into the Resize's output, so any OTHER consumer of that tensor
    suddenly sees the upsampled shape. Our neck hits this on both upsamples:
    yolov5's FPN feeds each Conv output to an upsample AND to a later concat, so
    the Mul has two consumers. Result:

        [ShapeInferenceError] (op_type:Relu, node name: MultiThreshold_62):
        Inferred shape and existing shape differ in dimension 2: (26) vs (13)

    This is an upstream bug, FIXED AFTER v0.10.1 -- FINN's dev branch now guards
    with `if model.is_fork_node(prod0): ... "try MoveLinearPastFork first"`.

    We take their advice rather than their patch: `MoveLinearPastFork` already
    exists in v0.10.1, it is simply absent from `Streamline()`'s transform list.
    Running it first duplicates the Mul onto each branch, so the fork is gone by
    the time MoveScalarLinearPastInvariants looks. No FINN source is modified,
    which keeps the 2022.2/v0.10.1 pin honest.
    """
    from finn.transformation.streamline.reorder import MoveLinearPastFork

    return model.transform(MoveLinearPastFork())


def step_resolve_join_muls(model, build_cfg):
    """Absorb the Muls that streamlining leaves stranded on joins and forks.

    MEASURED 2026-08-05 on n_eighth after dev's phase_optimize_model: **60 Mul
    nodes survive**, every one of them turning a UINT8 input into a FLOAT32
    output. FINN only maps INTEGER tensors to hardware, so each stranded Mul
    poisons everything downstream -- which is why phase_convert_to_hardware
    rejected the graph with 61 unconverted layers. They split three ways:

        28  feed a Concat   -> no upstream rewrite exists; MoveMulPastJoinConcat
        17  feed an Add     -> MoveLinearPastEltwiseAdd  ((x*C)+(y*C) -> (x+y)*C)
        15  are fork nodes  -> MoveLinearPastFork  (duplicate onto each branch,
                               since AbsorbMulIntoMultiThreshold cannot absorb
                               one Mul into two different consumers)

    NOTE WHAT IS NOT IN `Streamline()`: none of these join/fork transforms appear
    in its transform list, in v0.10.1 OR dev. The tied-join-scales work exists
    precisely so the join rewrites can fire -- and the default flow never calls
    them. They have to be invoked here.

    USE FINN DEV'S OWN TRANSFORMS, NOT OURS. dev grew a `*PastJoin*` family
    (commit 45b84a54 "Add MoveMulPastJoinMul subclass, drop redundant
    MoveLinearPastEltwiseAdd"). Consequences:
      - `MoveLinearPastEltwiseAdd` NO LONGER EXISTS on dev -> `MoveMulPastJoinAdd`
      - dev ships its OWN `MoveMulPastJoinConcat`, so `finn_transforms`'s version
        is redundant here and strictly weaker: dev's subclasses
        `MoveAffinePastJoinConcat`, which handles per-CHANNEL params as well as
        scalars -- exactly the generalisation ours documents as unimplemented.
        Keep ours only for the v0.10.1 path, which has no equivalent.
    Tied scales are STILL required either way: `MoveMulPastJoinAdd` compares the
    two Mul initialisers, and `MoveAffinePastJoinConcat.are_producers_identical_
    scalar_ops` rejects unequal scalars.

    Iterated, because the three interact: freeing a fork exposes a join, and
    absorbing at a join can expose the next fork upstream. Stops early once the
    Mul count stops falling.
    """
    from finn.transformation.streamline import Streamline
    from finn.transformation.streamline.reorder import (
        MoveAddPastJoinAdd,
        MoveAddPastJoinConcat,
        MakeScaleResizeNHWC,
        MoveLinearPastFork,
        MoveMulPastJoinAdd,
        MoveMulPastJoinConcat,
        MoveTransposePastJoinAdd,
        MoveTransposePastJoinConcat,
    )
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    import finn.transformation.streamline.absorb as absorb

    from finn_transforms import ConcatToNHWC, MoveScalarMulPastIm2Col

    def n_mul(m):
        return sum(1 for n in m.graph.node if n.op_type == 'Mul')

    before = n_mul(model)
    for i in range(6):
        prev = n_mul(model)
        model = model.transform(MoveLinearPastFork())
        model = model.transform(MoveMulPastJoinConcat())
        model = model.transform(MoveAddPastJoinConcat())
        model = model.transform(MoveMulPastJoinAdd())
        model = model.transform(MoveAddPastJoinAdd())
        # Layout, not datatype, is what blocks the joins themselves:
        # InferConcatLayer only converts Concat "operating on last/-1 axis"
        # (StreamingConcat is channel-last), and ours are axis=1 on NCHW. Conv
        # lowering inserts the NCHW<->NHWC transposes but they do not propagate
        # through a join on their own, so push them past and let the pairs cancel.
        model = model.transform(MoveTransposePastJoinConcat())
        model = model.transform(MoveTransposePastJoinAdd())
        # MoveTransposePastJoinConcat only fixed 4 of 13: it needs EVERY producer
        # to be the same op, and our joins are asymmetric. ConcatToNHWC does not
        # care what feeds the branches -- it rewrites the join itself, and the
        # transposes it inserts cancel against conv lowering's existing ones.
        model = model.transform(ConcatToNHWC())
        model = model.transform(MoveScalarMulPastIm2Col())
        # Same layout story for the 2 neck upsamples: InferUpsample needs NHWC
        # ("Input not NHWC. Can't infer UpsampleNearestNeighbour"), and ours come
        # out NCHW with UINT8 data. dev already ships the fix; it is simply not in
        # Streamline()'s list, exactly like MakeMaxPoolNHWC.
        model = model.transform(MakeScaleResizeNHWC())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(InferShapes())
        model = model.transform(Streamline())
        now = n_mul(model)
        print(f'  [resolve_join_muls] pass {i}: {prev} -> {now} Mul nodes')
        if now >= prev:
            break
    # UPSTREAM BUG (dev, build_dataflow_steps.py:553):
    #   apply_if_relevant(model, ["Upsample"], to_hw.InferUpsample(), "upsample layers")
    # the gate lists only op_type "Upsample", but ONNX emits "Resize" (Upsample was
    # removed after opset 9). InferUpsample itself handles BOTH -- it just never
    # gets invoked, silently, with no warning. Our 2 neck upsamples are Resize, so
    # call it ourselves. MakeScaleResizeNHWC above has already satisfied every
    # condition it checks (NHWC layout, UINT8 input, mode=nearest, scales 1,2,2,1).
    import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

    n_resize = sum(1 for n in model.graph.node if n.op_type in ('Resize', 'Upsample'))
    if n_resize:
        model = model.transform(to_hw.InferUpsample())
        left = sum(1 for n in model.graph.node if n.op_type in ('Resize', 'Upsample'))
        print(f'  [resolve_join_muls] InferUpsample: {n_resize} -> {left} Resize/Upsample')

    model = model.transform(InferDataTypes())
    print(f'  [resolve_join_muls] {before} -> {n_mul(model)} Mul nodes overall')
    return model


def with_custom_steps(base_steps):
    """Splice our steps into FINN's list. WHICH steps depends on the FINN version.

    Positions are found by NAME, never by index. The two supported layouts:

    v0.10.1 -- fine-grained `step_*` list. Needs all three of our steps:
      step_input_quant_to_uint8   frontend rejects unsigned identity Quant
      step_move_linear_past_fork  MoveScalarLinearPastInvariants has no fork guard
      step_move_mul_past_concat   no Mul-past-Concat rewrite exists

    dev -- restructured into 4 coarse `phase_*` steps, so there is no
    step_streamline to splice around. It also FIXED the other two problems
    (dropped the signed-identity-Quant check; added the fork guard), and its
    phase_convert_to_hardware handles Concat, elementwise Add and stream forks
    directly. We still insert the input step, but note what it does NOT buy:

      MEASURED 2026-08-05 -- with and without it, dev stops identically at
      "Non-contiguous dataflow block detected" with 61 unconverted layers
      (38 MatMul, 13 Concat, 4 Im2Col, 3 MaxPool, 2 Resize, 1 Transpose) and 8
      "Input is not int. Can't infer ConvInpGen" warnings. So the float-datatype
      problem is NOT caused by a missing UINT8 annotation on the graph input;
      that hypothesis was wrong. Root cause still unidentified.
    """
    steps = list(base_steps)
    names = [s for s in steps if isinstance(s, str)]

    if 'step_streamline' in names:                      # v0.10.1-style flow
        i = steps.index('step_streamline')
        steps.insert(i + 1, step_move_mul_past_concat)
        steps.insert(i, step_move_linear_past_fork)     # before, not after
        j = steps.index('step_qonnx_to_finn')
        steps.insert(j, step_input_quant_to_uint8)
        return steps

    if 'phase_prepare_model' in names:                  # dev-style phase flow
        # insert the LATER position first, so the earlier insert cannot shift it
        steps.insert(steps.index('phase_convert_to_hardware'), step_resolve_join_muls)
        steps.insert(steps.index('phase_prepare_model'), step_input_quant_to_uint8)
        return steps

    raise SystemExit(                                   # fail loudly, not silently
        f'unrecognised FINN step list, cannot splice custom steps: {names}')

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
    ap.add_argument('--stock', action='store_true',
                    help="use FINN's own step list unmodified, with none of our "
                         "custom steps. Required on FINN dev, which replaced the "
                         "named step_* functions with 4 coarse phase_* steps, so "
                         "there is no step_streamline to splice around. Also the "
                         "cleanest test of whether upstream now handles our graph "
                         "unaided -- dev dropped the signed-identity-Quant check "
                         "and added the fork guards we had to work around.")
    ap.add_argument('--target-fps', type=int, default=30)
    ap.add_argument('--clk-ns', type=float, default=10.0, help='10.0 ns = 100 MHz')
    ap.add_argument('--standalone-thresholds', action='store_true',
                    help='split thresholds into their own layers so the RTL MVAU '
                         '(which packs DSPs) can be used instead of HLS. See the '
                         'build_cfg comment.')
    ap.add_argument('--start-step', default=None,
                    help='resume from a saved checkpoint instead of rebuilding from '
                         'scratch, e.g. --start-step phase_generate_outputs to redo '
                         'only stitch+synthesis. On dev use PHASE names, not step '
                         'names. Needs save_intermediate_models (on by default) and '
                         'the SAME --out directory.')
    ap.add_argument('--folding-config', default=None,
                    help='folding config JSON from balance_folding.py, applied in '
                         'step_apply_folding_config -- overrides --target-fps. '
                         'Use this for any real build; --target-fps alone '
                         'under-folds badly (see CLAUDE.md C4).')
    ap.add_argument('--fifo-strategy', default='largefifo_rtlsim',
                    choices=['largefifo_rtlsim', 'characterize', 'none'],
                    help="bitfile mode only. 'largefifo_rtlsim' is FINN's default "
                         "and does not scale to n_eighth; 'characterize' rejects "
                         "residual joins; 'none' uses shallow default depths and "
                         "makes utilisation a floor. See the build_cfg comment.")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    assert onnx.exists(), f'missing {onnx} -- run export_qonnx.py first'
    out = Path(args.out or HERE / f'finn_build_{onnx.stem}{"_bit" if args.bitfile else "_est"}')

    if args.bitfile:
        steps = (list(cfg.default_build_dataflow_steps) if args.stock
                 else with_custom_steps(cfg.default_build_dataflow_steps))
        outputs = [cfg.DataflowOutputType.ESTIMATE_REPORTS,
                   cfg.DataflowOutputType.STITCHED_IP,
                   cfg.DataflowOutputType.BITFILE,
                   cfg.DataflowOutputType.PYNQ_DRIVER,
                   cfg.DataflowOutputType.DEPLOYMENT_PACKAGE]
    else:
        steps = (list(cfg.estimate_only_dataflow_steps) if args.stock
                 else with_custom_steps(cfg.estimate_only_dataflow_steps))
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
        # FINN's default is False, and it silently costs us the entire DSP array.
        # With False, MatMul+MultiThreshold fuse into one MVAU *with output
        # activation*, which the RTL MVAU cannot express -- so SpecializeLayers
        # falls back to MVAU_hls everywhere, and HLS does NO DSP packing (1
        # MAC/DSP; see Electronics 2025 14:3993 Table 4). Measured result: 14 DSPs
        # used out of 2,520 while LUT was the binding resource. True splits the
        # thresholds into standalone layers so MVAU_rtl becomes available (2
        # MAC/DSP at 8-bit on DSP48E2, 4 at 4-bit), and additionally routes
        # high-bitwidth thresholds to the cheaper Requant. Both reference designs
        # set it. Costs ~1 extra node per MVAU, which matters for the block-design
        # stitch (C2) since that scales quadratically.
        standalone_thresholds=args.standalone_thresholds,
        folding_config_file=args.folding_config,
        start_step=args.start_step,
        # keep every intermediate .onnx: if a join blows up, the last surviving
        # graph is the diagnostic and re-running to reproduce is expensive
        save_intermediate_models=True,
        # FIFO sizing -- see --fifo-strategy. Neither auto strategy works on
        # n_eighth (measured 2026-08-05/06):
        #   largefifo_rtlsim (FINN's DEFAULT): inserts a large FIFO between every
        #     layer pair, then stitches and rtlsims. 245 layers + 291 FIFOs =
        #     ~536 block-design cells, and Vivado's create_bd_cell degrades
        #     super-linearly -- one cell took 3m23s at cell ~255 and was still
        #     slowing (peak mem 9.7 GB). Killed after 10 h, never reached rtlsim.
        #   characterize: refuses outright, and the guard is deliberate --
        #     derive_characteristic.py:_assert_no_reconvergent_residuals. It sizes
        #     each consumer from io_chrc_in[0] only, so a two-stream
        #     ElementwiseAdd (our 7 residual joins) cannot be sized. Fine for
        #     join-free nets like pico.
        # 'none' (auto_fifo_depths=False) leaves InsertFIFO's shallow defaults:
        # reaches synthesis fastest and Fmax stays valid, but FIFO BRAM is
        # understated (utilisation becomes a FLOOR) and the bitstream may
        # deadlock or miss target throughput. Measurement vehicle only.
        auto_fifo_depths=(args.fifo_strategy != 'none'),
        auto_fifo_strategy=(cfg.AutoFIFOSizingMethod.CHARACTERIZE
                            if args.fifo_strategy == 'characterize'
                            else cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM),
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
