"""Find the most aggressive folding that still fits the device, and emit it as a
FINN folding config JSON.

WHY THIS EXISTS
---------------
FINN's `SetFolding` (driven by `target_fps`) only guarantees every layer is
*under* a cycle target. It never equalises layers and it never looks at
resources at all. On n_eighth at `--target-fps 30` that produced a pipeline where
the first conv is 6x slower than the layers feeding it:

    FMPadding_rtl_0                     524,172 cycles
    ConvolutionInputGenerator_rtl_0   1,170,645 cycles
    MVAU_hls_0                        3,115,008 cycles   <- bottleneck

FINN's FIFO sizer then correctly demanded ~1.5 FRAMES of buffering to absorb the
mismatch (depths 289,032 and 243,514 -- 94% of all buffering in the design), and
the build died because Vivado's max FIFO depth is 32,768.

Calì/Falaschetti/Biagetti (Electronics 2025, 14, 3993) -- the Z7020 YOLOv3-Tiny
paper pico's architecture is modelled on -- solved this in section 3.6.1 by doing
the opposite: start FULLY FOLDED (every PE/SIMD = 1) and iteratively unfold only
the SLOWEST layer, until the cycles-per-layer profile is FLAT ("Figure 10. The
optimal Flat cycles-per-layer graph achieved") or the bottleneck cannot be
unfolded further. A flat pipeline needs almost no buffering.

The cheap layers here are already at their floor (PE=SIMD=1), so the only way to
flatten the profile is to speed the bottleneck UP -- which is what lowering the
cycle target does. So rather than reimplement per-layer unfolding (SetFolding
already encodes every divisibility / parallel_window / depthwise-SWG constraint,
and getting those wrong produces silently invalid hardware), this searches for
the lowest target that still fits the part.

SEARCH METHOD: a geometric SWEEP, deliberately not a binary search. It is
tempting to assume resources fall monotonically as the target rises (less
unfolding = less hardware), which would justify bisection. **That assumption is
false**, measured on n_eighth 2026-08-06: fully folded reports 252,349 LUT
(92.1%) while the same model at target_fps=30 needs only 77,927. At SIMD=PE=1
each weight memory is 8 bits wide and very deep, which FINN's estimator declines
to place in BRAM (244 BRAM fully folded vs 348 at target_fps=30), so the weights
land in LUTRAM and LUT usage explodes. Resource usage is therefore U-shaped in the
target, and bisection can converge on a false infeasibility. pico happens to be
monotonic, which is exactly why the bug hid there.

So: evaluate every target on a halving ladder and pick the feasible point with
the lowest achieved max_cycles. ~25 evaluations, no synthesis, minutes.

USAGE (inside FINN's Docker; see finn_build.py for the run-docker invocation)

    ./run-docker.sh build_custom /home/alex/Repos/drone-detect/export \
        balance_folding --onnx <build>/intermediate_models/step_specialize_layers.onnx

Feed the result back into a build with `--folding-config <json>`, which FINN
applies in `step_apply_folding_config`, overriding the target_fps folding. This
is the same route both reference designs took (their `final_hw_config_90fps.json`
and `final_hw_config.json` are hand-tuned files fed to ApplyConfig).
"""

import argparse
import json
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.config import (
    extract_model_config_consolidate_shuffles,
    extract_model_config_to_json,
)
from functools import partial

# ZCU102 = XCZU9EG-2FFVB1156. NOTE BRAM: the part has 912 blocks of *36* Kb,
# which is 1824 in the BRAM_18K units FINN reports. Dividing by 912 double-counts
# utilisation -- we did exactly that once and reported 38.2% for what is 19.1%.
FPGA_PART = "xczu9eg-ffvb1156-2-e"
BUDGET = {"LUT": 274080, "BRAM_18K": 1824, "DSP": 2520, "URAM": 0}

# The estimate covers the dataflow partition only: the Zynq shell, DMA engines,
# AXI interconnect and (crucially) the FIFOs sized later are all extra. FINN also
# UNDER-estimates LUT -- the Z7020 paper measured 26,694 estimated vs 41,605
# actual, a factor of 1.56 -- so leaving a quarter of the device unclaimed is not
# conservatism, it is the known error bar.
DEFAULT_HEADROOM = 0.75

HW_ATTRS = [
    "PE", "SIMD", "parallel_window", "ram_style", "depth", "impl_style",
    "resType", "mem_mode", "runtime_writeable_weights", "inFIFODepths",
    "outFIFODepths", "depth_trigger_uram", "depth_trigger_bram",
]


def evaluate(onnx_path, target_cycles, fpga_part, mvau_wwidth_max):
    """Fold a fresh copy of the model at `target_cycles` and measure it.

    Reloads from disk rather than deep-copying: SetFolding mutates node
    attributes in place, so a stale model would silently carry folding from the
    previous iteration and corrupt the search.
    """
    model = ModelWrapper(str(onnx_path))
    model = model.transform(
        SetFolding(target_cycles_per_frame=target_cycles, mvau_wwidth_max=mvau_wwidth_max)
    )
    # Replicate step_minimize_bit_width, which the build runs immediately AFTER
    # folding. It is not optional bookkeeping: accumulator width depends on SIMD,
    # so it can only run once folding is fixed, and it is worth a factor of ~3 in
    # LUT. Measured on n_eighth -- without it, res_estimation reports 252,349 LUT
    # where the real build reports 77,927, i.e. the whole design looks infeasible.
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataTypes())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(InferDataTypes())
    model = model.transform(AnnotateCycles())
    perf = model.analysis(dataflow_performance)
    res = model.analysis(partial(res_estimation, fpgapart=fpga_part))
    total = aggregate_dict_keys(res)
    return model, perf, total


def fits(total, headroom):
    for key, cap in BUDGET.items():
        used = total.get(key, 0)
        if cap == 0:
            if used > 0:
                return False
        elif used > cap * headroom:
            return False
    return True


def util_str(total):
    parts = []
    for key, cap in BUDGET.items():
        used = total.get(key, 0)
        parts.append(f"{key} {used:,.0f}" + (f" ({used / cap * 100:.1f}%)" if cap else ""))
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True,
                    help="a post-SpecializeLayers intermediate model, e.g. "
                         "<build>/intermediate_models/step_specialize_layers.onnx")
    ap.add_argument("--out", default=None, help="output folding config JSON")
    ap.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM,
                    help="fraction of the device the dataflow partition may claim")
    ap.add_argument("--fpga-part", default=FPGA_PART)
    ap.add_argument("--mvau-wwidth-max", type=int, default=36,
                    help="SetFolding stops raising an MVAU's SIMD once "
                         "weight_bits*SIMD exceeds this. FINN's default of 36 caps "
                         "SIMD at 4 for our 8-bit weights (the W4A4 reference designs "
                         "get 9), and it is what pins MVAU_hls_0 at SIMD=3. Raising it "
                         "widens the weight stream/memory per PE.")
    ap.add_argument("--clk-ns", type=float, default=10.0)
    args = ap.parse_args()

    onnx = Path(args.onnx)
    assert onnx.exists(), f"missing {onnx}"
    out = Path(args.out or onnx.parent / "balanced_folding_config.json")

    # Fully folded (PE=SIMD=1): a huge target makes SetFolding stop immediately on
    # every layer, which is the paper's starting point. NOT necessarily the
    # resource minimum -- see SEARCH METHOD in the module docstring -- so this is
    # reported for reference and then swept past, never used as a feasibility gate.
    print("=== baseline: fully folded ===")
    _, perf, total = evaluate(onnx, 10**12, args.fpga_part, args.mvau_wwidth_max)
    top = perf["max_cycles"]
    print(f"max_cycles {top:,}  ({top * args.clk_ns / 1e6:.1f} ms/frame)  {util_str(total)}"
          f"  {'fits' if fits(total, args.headroom) else 'TOO BIG (expected; sweeping anyway)'}")

    print(f"\n=== sweeping targets (headroom {args.headroom:.0%}) ===")
    best, target, seen_floor = None, top, None
    while target >= 1:
        model, perf, total = evaluate(onnx, target, args.fpga_part, args.mvau_wwidth_max)
        ok = fits(total, args.headroom)
        achieved = perf["max_cycles"]
        print(f"target {target:>12,} -> max_cycles {achieved:>10,}  "
              f"{'fits  ' if ok else 'TOO BIG'}  {util_str(total)}")
        if ok and (best is None or achieved < best[2]["max_cycles"]):
            best = (target, model, perf, total)
        # Once the achieved cycles stop tracking the target, some layer is fully
        # unfolded and nothing below this can go faster -- the paper's
        # termination condition ("the slowest layer being fully unfolded").
        if achieved > target:
            if seen_floor == achieved:
                print(f"  floor reached: {perf['max_cycles_node_name']} fully unfolded "
                      f"at {achieved:,} cycles -- stopping")
                break
            seen_floor = achieved
        target //= 2

    if best is None:
        print("\nNo feasible target found. Every folding either exceeds the "
              "headroom or cannot be represented -- consider raising --headroom, "
              "lowering --mvau-wwidth-max, or a smaller model.")
        raise SystemExit(2)

    target, model, perf, total = best
    nonzero = sorted(
        (c for c in model.analysis(exp_cycles_per_layer).values() if c > 0), reverse=True
    )

    print(f"\n=== chosen: target {target:,} (achieved {perf['max_cycles']:,}) ===")
    print(f"max_cycles      {perf['max_cycles']:,}  ({perf['max_cycles_node_name']})")
    print(f"frame time      {perf['max_cycles'] * args.clk_ns / 1e6:.2f} ms  "
          f"-> {1e9 / (perf['max_cycles'] * args.clk_ns):.1f} FPS @ {1000 / args.clk_ns:.0f} MHz")
    print(f"resources       {util_str(total)}")
    if nonzero:
        # Flatness is the whole point: the ratio between the bottleneck and the
        # rest is what the FIFOs have to absorb.
        median = nonzero[len(nonzero) // 2]
        print(f"flatness        max/median = {nonzero[0] / median:.1f}x "
              f"(max {nonzero[0]:,}, median {median:,}) -- lower is better")
        print(f"top 8 layers    {', '.join(f'{c:,}' for c in nonzero[:8])}")

    if model.get_nodes_by_op_type("InnerShuffle_rtl") or model.get_nodes_by_op_type(
        "OuterShuffle_hls"
    ):
        extract_model_config_consolidate_shuffles(model, str(out), HW_ATTRS)
    else:
        extract_model_config_to_json(model, str(out), HW_ATTRS)
    print(f"\nwrote {out}")
    print(f"feed back with:  finn_build.py --folding-config {out}")


if __name__ == "__main__":
    main()
