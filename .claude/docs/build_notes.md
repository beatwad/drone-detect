# Build Notes — QAT → QONNX → FINN → bitstream

Everything measured on the way from a float checkpoint to an FPGA accelerator.
**This is what the tools did, not what the docs claim.** Where a fact came from a
measurement, the date is given; where it came from reading source, the file and
line are given.

Companion docs: [project_brief.md](project_brief.md) (hardware, datasets, roadmap),
[host_pipeline_plan.md](host_pipeline_plan.md) (the phased plan these notes execute).
Project-level status, locked decisions and open questions live in
[CLAUDE.md](../../CLAUDE.md).

**Section lettering (A, B, C0, C, C2–C5, D) is referenced from elsewhere — keep it
stable when editing.**

---

## 1. Brevitas QAT

Code: `qat/` (`quantize.py`, `ptq_baseline.py`, `train_qat.py`, `evaluate.py`).

### Results — INT8 needs no fine-tuning at all
W8A8 PTQ mAP50 **0.9628** vs float **0.9629**. 4-bit weights collapse under PTQ
(**0.197**) but QAT recovers close range to **0.9856**. Gate met.

So the cheap path is real: for W8A8 the calibration pass alone is enough, and the
QAT fine-tune is only needed if we ever drop to 4-bit weights. See memory
`qat-w4a8-results` and `pico-w8a8-beats-w4a8` — on pico, 8-bit calibration beats
4-bit QAT on *every* regime with zero training.

### SiLU→ReLU: train float in ReLU, never paste
Locked decision (see CLAUDE.md). Retrained ReLU costs 2.1 pt mAP50-95 and **zero**
mAP50 / centre error. Pasting ReLU into SiLU-trained weights destroys the model
(mAP50 0.963 → **0.078**) because ~49% of pre-activations are negative and ReLU
zeroes them across 57 layers. QAT starts from
`runs/train/relu_more_data_5/weights/best.pt`, never the SiLU model.

**Gotcha — shared SiLU instance.** YOLOv5 uses one class-level
`Conv.default_act = nn.SiLU()` shared across ALL Conv layers, so `model.modules()`
reports a *single* SiLU. The swap is done by rebinding `Conv.default_act` (or
walking `.act` attributes) before instantiation — **not** per-layer object
replacement, which silently no-ops.

### Tied join scales — the FINN precondition, set here
FINN's join streamlining is algebra: `a·x + b·y` does not factor unless `a == b`.
Verified in upstream FINN `streamline/reorder.py` — `MoveLinearPastEltwiseAdd`
matches `(x*C) + (y*C) -> (x + y) * C` and guards on
`np.array_equal(init0, init1)`. **No toolchain change avoids this**, so it must be
set during QAT, not fixed at export.

**DONE 2026-08-04** in `qat/quantize.py`: `SharedQuant` +
`QuantC3` / `QuantSPPF` / `QuantBottleneck` / `QuantConcat`. All **20 joins**
(13 concats + 7 residual adds) share one quantizer instance.

- Gate: `export/check_join_scales.py` → **20/20 tied**.
- **Free**: n_eighth W8A8 close **0.9830** vs 0.9845 untied, long **0.8605** vs
  0.8604.
- Brevitas' own `QuantCat` is broken — hence the hand-written `QuantConcat`
  (memory `tie-join-scales-implemented`).

Still required on FINN dev too: `MoveMulPastJoinAdd` compares the two Mul
initialisers and `are_producers_identical_scalar_ops` rejects unequal scalars.

---

## 2. QONNX export

Code: `export/export_qonnx.py`, `export/verify_qonnx.py`, `export/check_join_scales.py`.
Artifact: **`export/n_eighth_tied_w8a8_416.onnx`**.

Both gates pass:
- `check_join_scales.py` → 20/20 tied
- `verify_qonnx.py` → faithful, **32/32 boxes, 0.367 px** max centre error

pico exports with 0 joins / 0 unsupported ops; yolov5n with 20 joins
(memory `qonnx-export-works`).

---

## 3. FINN compiles branched YOLO — the blocker was wrong

> **RESOLVED 2026-08-04.** Was: *"FINN 0.10 cannot compile branched graphs, so
> yolov5n is blocked on any board."* The Z7020 papers deleted their FPNs for
> **resource** reasons that got retconned into a toolchain limit.

Danilowicz & Kryjak, ARC 2025 ([arXiv 2503.13023](https://arxiv.org/abs/2503.13023))
built branched YOLOv8n — C2f split+concat, residual Bottlenecks, SPPF, full FPN —
through FINN on a **ZCU102**, 195.3 fps @300 MHz, W4A4, 320×192.

yolov5n's **13 concats + 7 adds** (4 neck + 8 C3 + 1 SPPF; adds are the 7 shortcut
Bottlenecks — head C3s use `shortcut=False`) are compilable. Two real costs
replace the blocker:

1. **Joins require branches to share a quantisation scale** — set during QAT, not
   fixable at export. See §1 above. **DONE.**
2. **Concat joins have no upstream streamlining path in v0.10.1.** `reorder.py`
   has no `Move{Mul,Add}PastJoinConcat`. Written as
   `export/finn_transforms.MoveMulPastJoinConcat` (NOT the ~10-line subclass we
   expected). **Moot on FINN dev**, which converts Concat natively.

We need **less** patching than the reference did: they added a `Split` node for
C2f; YOLOv5's C3 has no split (cv1/cv2 are two convs on one input = a plain fork,
already handled by `duplicatestreams.py`). Every other primitive is upstream
(`concat.py`, `streamingeltwise.py`, `upsampler.py`, `streamingmaxpool.py`).

### The reference fork — read it, don't clone it
Corrected 2026-08-06: `mdanilow/finn` branch `yolov8_dev` **IS public**, 32 commits
ahead of `Xilinx/finn` dev and **1999 behind**, last touched 2025-08-28. We don't
need it — the substance (the `*PastJoin*` family, concat/upsample/addstreams/split,
`set_folding` join support) all landed upstream, which is why n_eighth compiled on
plain dev.

Its `notebooks/experiments/yolov8/build_yolov8.py` is worth reading as independent
confirmation of our build: it also calls `to_hw.InferUpsample()` explicitly, also
iterates a hand-ordered streamline list around `MoveMulPastJoinAdd` +
`MoveMulPastJoinConcat`, and also needs `MoveLinearPastFork` + `MoveMulPastMaxPool`
**twice, commented "SPPF"**.

Differences: it sets `standalone_thresholds=True` from the start, and gets its
UINT8 input the canonical way (`MergeONNXModels` of a `ToTensor()` preproc graph)
rather than our hand-written `InputQuantToUintDtype`.

Its estimate report calibrates ours almost exactly — yolov8n@320×192 W4A4 gives
`max_cycles` 3,072,000 → 32.55 FPS (vs our 3,115,008 → 32.10) and the same
pessimistic `critical_path_cycles` latency artefact. Note its `target_fps=90` did
**not** produce 90; the paper's 195 FPS came from a separate hand-tuned
`final_hw_config_90fps.json`.

Also: `UviDTE-FPSoC/daiedge-fpga` (thin student repo, no published results, but
real bitfiles) for the `auto_fifo_depths=False` precedent and a board-side
inference notebook. Memory `finn-yolo-reference-repos`.

### Keep weights on-chip
The reference streamed weights from DDR, which cost ~100k LUT of DMA +
interconnect (detector 105k / 38%, full design **205k / 75%** of the ZCU102).
Upstream external mem mode also carries a v0.10.1 release-note warning of
"unexpected behaviour". ZCU102's 32.1 Mbit makes it avoidable.

### Vitis AI DPU as a fallback
DPUCZDX8G officially supports ZCU102 and remains available as a de-risking
baseline — one bitstream serves any model — but it is DDR/PS-centric and
4.8–8.8 W, so it is **not** the deployment architecture. See "Power shape" in
CLAUDE.md.

---

## 4. Running a JOINED network through FINN — the actual sequence

*Established by measurement 2026-08-05, the first day FINN was runnable.
**Estimate-only build is COMPLETE** — see §C for the result.*

**Control result first:** `pico` (0 joins) compiles **end-to-end on v0.10.1**, all
12 steps, in **6 seconds** — 19,472 LUT (7.1%), 193 BRAM_18K (10.6%), 1 DSP,
32.1 FPS est. So the toolchain, the part, the licence and our export format are
all sound; everything below is specifically about joins.

### A. Export-side rules (these are OUR responsibility, all verified)

1. **Tie the quantiser at every join** — `SharedQuant` in `qat/quantize.py`, one
   instance per join applied to all operands. Algebra, not a FINN quirk. Gate:
   `export/check_join_scales.py` → 20/20. (Full detail in §1.)

2. **Put a plain `nn.ReLU()` in front of every shared quantiser.** FINN has only
   two activation handlers — `QuantReluHandler` claims `{Relu, Selu}`,
   `QuantIdentityHandler` claims `{BatchNormalization, Sub, Add, Mul, Div,
   DebugMarker, None}` — and `quant_act_to_multithreshold.py` **falls back to
   QuantIdentityHandler for any predecessor neither claims**, which then rejects
   unsigned. So the real rule is: *an unsigned Quant must sit immediately behind a
   Relu.* Without it, 36 nodes fail (30 preceded by another `Quant`, 3 by SPPF's
   MaxPool, 2 by the neck's Resize, 1 the graph input).

   **Free**: every join operand here is already non-negative (post-ReLU conv
   outputs, or MaxPool/nearest-Resize of such, or a sum of two), so `ReLU(t)==t`.
   Verified: min pre-ReLU value at all 7 residual adds = `+0.000000`, and both
   gates unchanged (20/20 tied; 32/32 boxes @ 0.367 px).

   *Corollary:* never leave a bare `Relu` that is NOT followed by a `Quant` — FINN
   has no hardware node for one. Check it; we do.

3. **The input `Quant`** (`input_quant=Uint8ActPerTensorFloat`) is unsigned with no
   predecessor → same fallback → rejected on v0.10.1. Rewritten by
   `finn_transforms.InputQuantToUintDtype` to `x_uint8 -> Mul(scale)`.

   **This changes the accelerator's input contract**: it consumes integer codes
   `round(x/scale)`, not floats. Good for deployment (no float preprocessing on the
   ARM side) but **host preprocessing must match**.

### B. Which FINN — dev, not v0.10.1

v0.10.1's streamlining assumes single-consumer chains; yolov5's FPN and SPPF are
forks everywhere. Two distinct bugs, both fatal:

- `MoveScalarLinearPastInvariants` has **no fork guard** — it moves a scalar Mul
  past a nearest-neighbour Resize by rewiring the Mul's output tensor, so the
  fork's *other* consumer sees the upsampled shape
  (`differ in dimension 2: (26) vs (13)`). Workaround that WORKS: run
  `MoveLinearPastFork()` first (already in v0.10.1, just absent from
  `Streamline()`'s list). Upstream dev added the guard and the same advice.
- SPPF's MaxPool fork loses its NHWC conversion on one branch, leaving
  `MultiThreshold_56` with an NCHW output shape. **Unsolved on v0.10.1** —
  repairing that one annotation just moves the error to the next node, so the whole
  branch is stale, not one entry.

**FINN dev clears both**, and requires the same Vivado 2022.2, so the pin that
matters is untouched. Its most recent merge is literally
`feature/unsigned_identity_quant`, which deletes the `signed` check in rule A3.
Two consequences:

- dev **restructured the build flow**: the ~12 `step_*` functions became 4
  `phase_*` steps, so nothing can be spliced by step name. `with_custom_steps()` in
  `export/finn_build.py` detects which layout it is looking at and fails loudly on
  a third.
- `MoveMulPastJoinConcat` and `MoveLinearPastFork` are **not needed on dev** — its
  `phase_convert_to_hardware` logs `Converting concat layers`, `Converting
  elementwise binary operations`, `Checking for graph forks (duplicate streams)`.
  Keep them only for the v0.10.1 path.

### C0. The join Muls — the main lever, and it works (2026-08-05)

`Streamline()` leaves **60 Mul nodes** stranded, each turning `UINT8 -> FLOAT32`.
FINN only maps INTEGER tensors to hardware, so every stranded Mul poisons
everything downstream — that, not the input annotation, is why nothing converted.
They split 28 feeding a Concat / 17 feeding an Add / 15 fork nodes.

**None of the transforms that fix this are in `Streamline()`'s list, in either FINN
version.** The tied-join-scales work exists so `MoveMulPastJoinAdd` can fire, and
the default flow never calls it. `export/finn_build.py:step_resolve_join_muls`
invokes them explicitly and iterates (freeing a fork exposes a join and vice
versa): **60 → 17 → 8 → 7**, converged; unconverted layers **61 → 24**, MatMul
38 → 4, MaxPool 3 → 0.

**Use dev's own transforms, not ours.** dev grew a `*PastJoin*` family (commit
`45b84a54`) and **deleted `MoveLinearPastEltwiseAdd`** → use `MoveMulPastJoinAdd`.
dev also ships its own `MoveMulPastJoinConcat`, subclassing
`MoveAffinePastJoinConcat`, which handles per-CHANNEL params as well as scalars —
strictly better than `finn_transforms`'s, which is now only for the v0.10.1 path.
**Tied scales are still required either way** (see §1).

### C. RESULT — n_eighth compiles end-to-end on dev (2026-08-05)

All 6 steps, exit 0, **0 errors**, 466 s. Build `n_eighth_fix6`. Every layer
converts:

    Dataflow conversion validation: Fpgadataflow layers form contiguous block
    245 layers: 57 MVAU_hls, 42 Thresholding_rtl, 24 DuplicateStreams_hls,
    23 InnerShuffle_rtl, 23 OuterShuffle_hls, 21 FMPadding_rtl,
    21 ConvolutionInputGenerator_rtl, 13 StreamingConcat_hls,
    10 ElementwiseAdd_hls, 3 Pool_hls, 3 MVAU_rtl, 3 ElementwiseMul_hls,
    2 UpsampleNearestNeighbour_hls

All **13 concats** → StreamingConcat, all **7 residual adds** (+3 head biases) →
ElementwiseAdd, SPPF's 3 MaxPools → Pool_hls, both neck upsamples →
UpsampleNearestNeighbour. Unconverted layers went **61 → 24 → 20 → 3 → 0**. The
FINN-compiles-branched-YOLO claim is now verified on our own network.

**Estimates @ 100 MHz, ZCU102 (xczu9eg), `--target-fps 30`:**

| | LUT | BRAM_18K | DSP | URAM |
|---|---|---|---|---|
| pico (0 joins) | 19,472 (7.1%) | 193 (10.6%) | 1 | 0 |
| **n_eighth (20 joins)** | **77,927 (28.4%)** | **348 (19.1%)** | **3** | 0 |

Comfortable on a ZCU102. 348 BRAM_18K = **6.3 Mbit**, which by the portability rule
in CLAUDE.md is *under* the ~7 Mbit line — so this net plausibly ports below a K26.
Verify against real synthesis before believing it. 441k 8-bit weights = 3.5 Mbit,
so BRAM is ~1.8× the raw weight footprint, close to the ×2 rule of thumb. DSP≈0
because 8-bit MVAUs map to LUTs.

**The FPS number is a config knob, not a result.** `--target-fps` defaults to 30 and
`set_folding` folds each net just far enough to clear it, so *both* nets stop at
the identical `max_cycles = 3,115,008` → 32.1026 FPS. Do not read that as "n_eighth
is as fast as pico" — it means both were folded to the same target and n_eighth
paid 4× the LUTs to get there. See §C4.

**Estimate mode is CLOCK-BLIND, and its latency number is meaningless.** Both
verified 2026-08-05 by re-running at `--clk-ns 3.33 --target-fps 96`
(`n_eighth_300mhz`, matched folding):

- Resources, cycles and `auto_folding_config.json` came back **byte-identical** to
  the 100 MHz run. `--clk-ns` changes *nothing* in estimate mode except the final
  `cycles × period` multiplication (96.40 FPS, 401.8 ms). The extra LUT/FF Vitis
  HLS spends pipelining at a tighter period is real but **invisible until HLS
  synthesis runs**. Do not re-run estimates at other clocks expecting an area
  answer.
- `estimated_latency_ns` is `critical_path_cycles × period`, and
  `dataflow_performance`'s own docstring calls it "very pessimistic, it assumes **no
  overlap** between executions" — it sums whole-frame cycles of all ~245 layers in
  series, the opposite of what a streaming dataflow design does. Treat it as a hard
  upper bound only. The floor is one bottleneck frame (31.2 ms @100 MHz / 10.4 ms
  @300 MHz) plus pipeline fill; 32 layers are rate-matched at exactly `max_cycles`
  and 21 conv line-buffers sit in series. **True latency needs
  `RTLSIM_PERFORMANCE`, which requires `STITCHED_IP`** — hours, same cost class as
  a bitfile.

So achieved Fmax, real area and real latency are all **synthesis facts**. Estimate
mode has given all it can. Note FINN's own notebooks/tests target 10.0 ns almost
exclusively (a few at 5.0 ns, **none** at 3.33 ns), and the 300 MHz ZCU102 result
in arXiv 2503.13023 was W4A4, not our W8A8 — so treat 300 MHz as aspirational and
200 MHz as the trodden path. If chasing it, first flip
`standalone_thresholds=True` (but see §C5 — at our operating point it loses).

**Two fixes were needed beyond C0, both upstream bugs:**

1. **`InferUpsample` never ran.** `build_dataflow_steps.py:553` gates on op_type
   `["Upsample"]`, but ONNX opset ≥11 emits **`Resize`** — the deprecated name is
   the only one checked. Silent: no warning, the node just stays. Worked around by
   calling `to_hw.InferUpsample()` directly in `step_resolve_join_muls`.
2. **`set_folding.py` crashes on joins.** `TypeError: Attribute SIMD expects int,
   got numpy.int64` — `common_divisors()` yields numpy ints, `set_nodeattr` demands
   a Python int. Only reachable via StreamingConcat/Split, i.e. only for joined
   graphs, which is why upstream CI misses it. **LOCAL PATCH** at
   `~/Repos/finn-dev/.../set_folding.py:261` (`int(simd_val)`); original at
   `~/finn_build_dev/set_folding.py.orig`. **Re-apply after any `git pull`.**

Two remaining config warnings, both benign: `shell_no_bitfile` (expected for an
estimate-only run) and `standalone_thresholds=False`. The INT32→INT20 dtype
warnings are *narrowing*, i.e. good.

**Resolved, was open:** the 3 detection heads needed no output `QuantIdentity`.
dev's `ElementwiseBinary` converts `MVAU_rtl_{0,1,2}` → `ElementwiseMul_hls` →
`ElementwiseAdd_hls` directly, so `RawHead` stays as designed and the block stays
contiguous through the heads.

**Dead hypothesis, do not retry:** it is NOT a missing UINT8 annotation on the graph
input. Measured byte-identical (61 layers, same breakdown, same 8 warnings) with
and without `step_input_quant_to_uint8`.

---

## 5. Bitfile builds

### C2. NEVER use the default FIFO sizing strategy

Attempt 1 (2026-08-05, 100 MHz) died after **10 h** in `phase_optimize_hardware`,
having never reached synthesis. Cause: `step_set_fifo_depths` defaults to
`auto_fifo_strategy = LARGEFIFO_RTLSIM`, which inserts a large FIFO between *every*
layer pair, stitches the whole thing, then rtlsims 2 inferences. On n_eighth that
is 245 layers + **291 FIFOs ≈ 536 block-design cells**, and Vivado's
`create_bd_cell` degrades super-linearly:

    create_bd_cell: Time (s): cpu = 00:08:02 ; elapsed = 00:03:23   # cell ~255

3m23s for one cell and still slowing, peak memory 9.7 GB, host down to 2 GB free.
It was still assembling the block design — the rtlsim it was building toward had
not started.

**`CHARACTERIZE` is NOT the fix — it refuses our network outright** (attempt 2,
2026-08-06, failed in ~30 min):

    RuntimeError: Characterize FIFO sizing is not supported for models with
    reconvergent residual paths (two-stream ElementwiseAdd 'ElementwiseAdd_hls_0'
    fed by a stream fork). Use a different auto_fifo_strategy.

A deliberate guard (`derive_characteristic.py:_assert_no_reconvergent_residuals`),
added precisely so you fail fast instead of after an hours-long rtlsim. The method
sizes each consumer from `io_chrc_in[0]` **only**, so a node joining two live
streams cannot be sized. Our **7 residual adds** are exactly that. Not fixable by
config; fine for join-free nets like pico.

**So both automatic strategies are dead ends for n_eighth.** `--fifo-strategy` now
selects between them; the remaining option is `none` (`auto_fifo_depths=False`),
which leaves `InsertFIFO`'s shallow defaults — reaches synthesis fastest and keeps
Fmax valid, but understates FIFO BRAM (utilisation becomes a **floor**) and may
deadlock or miss target throughput. A measurement vehicle, not a deployable
accelerator.

**`none` has third-party precedent.** `UviDTE-FPSoC/daiedge-fpga` builds TinyYOLOv3
for PYNQ-Z1/Z2 and ZCU104 with exactly `auto_fifo_depths=False` +
`split_large_fifos=True` + `target_fps=10000`. So the path reaches a working
accelerator — but TinyYOLOv3 is far shallower than our 245 layers, so it does
**not** prove our stitch cost is survivable. **We do not currently set
`split_large_fifos`; try it.**

**The way out of paying FIFO sizing every build: size once, freeze, replay.**
mdanilow's `final_hw_config.json` is a 712-entry dump — 374 explicit
`StreamingFIFO_rtl_*` depths (mostly 2, one 771) alongside PE/SIMD/`ram_style`/
`impl_style`/`mem_mode` — produced by `extract_model_config_to_json` after a
successful sizing run and replayed via `folding_config_file`. **If any strategy
ever completes on n_eighth, dump it immediately and never re-run it.**

**Vivado IPI stitching is QUADRATIC in cell count.** Measured on the pico control
run: `create_bd_cell(i) ≈ 5.5 + 0.32·i` seconds, i.e. per-cell cost grows linearly,
so total cost grows as N². Fit from 54 timed cells, and it predicted n_eighth's
attempt-1 death almost exactly:

| cells | predicted stitch | note |
|---|---|---|
| 114 | ~45 min | pico, measured, matches |
| 536 | **~13.8 h** | n_eighth; attempt 1 killed at 10 h, still going |

FINN emits ~2.2 cells per layer (layer + FIFO + DWC), so n_eighth's 245 layers give
~536 cells. **`largefifo_rtlsim` stitches TWICE** (once in
`set_fifo_depths.py:463` for the sizing rtlsim, once in `step_create_stitched_ip`)
→ ~27 h for n_eighth. `--fifo-strategy none` stitches once → ~13.8 h. That halving
is the real benefit of `none`; it does NOT reduce cells per stitch
(`RemoveShallowFIFOs` defaults to `shallow_threshold=0`, and the depth-2 FIFOs
`InsertFIFO` creates are deliberately kept for decoupling).

### C3. THE REAL BLOCKER: auto-folding produces a wildly unbalanced pipeline

The pico control bitfile build (2026-08-06, `largefifo_rtlsim`) ran **all 8
phases** — FIFO sizing, HLS synth of every layer — in ~3 h and died only at final
stitching:

    ERROR: [IP_Flow 19-3461] Value '524288' is out of the range for parameter
    'FIFO depth(FIFO_DEPTH)' ... Valid values are - 16, 32, ... 32768

So the flow itself is sound. The cause is folding, from `final_hw_config.json`:

    FMPadding_rtl_0                     524,172 cycles
    StreamingFIFO_rtl_1              -> depth 289,032   (1.67 frames!)
    ConvolutionInputGenerator_rtl_0   1,170,645 cycles
    StreamingFIFO_rtl_3              -> depth 243,514
    MVAU_hls_0                        3,115,008 cycles   <- bottleneck, 6x slower

44 of 46 FIFOs are ≤4,149; those two outliers are **94% of all buffering**. The
first conv is 6× slower than the layers feeding it, so upstream races ahead,
back-pressures, and the sizer correctly demands ~1.5 frames of buffer.

**This is `set_folding` working as designed, not a bug:** it only guarantees each
layer is *under* `target_cycles_per_frame`; it never equalises them. At
`target_fps=30` the cheap early layers stay massively over-provisioned relative to
the first conv. It is exactly why the reference yolov8 design shipped a hand-tuned
`final_hw_config_90fps.json` instead of trusting `target_fps` — and why its
`target_fps=90` did not produce 90 FPS.

`split_large_fifos=True` only makes it *build*: a 289k-deep FIFO costs ~128–386
BRAM_18K depending on stream width, one FIFO exceeding pico's entire 193-BRAM
budget. Not viable. **The fix is to balance folding** (§C4).

**Do not assume omitting `RTLSIM_PERFORMANCE` avoids rtlsim** — it does not.
`step_set_fifo_depths` runs its own, in `phase_optimize_hardware` (step 6),
*before* synthesis. `RTLSIM_PERFORMANCE` is a separate, later step
(`phase_generate_outputs`, after `step_create_stitched_ip` and before
`step_synthesize_bitfile`) and needs `STITCHED_IP`.

Also: `NUM_DEFAULT_WORKERS` defaults to **4** in `run-docker.sh`; this box has 12
cores and there are ~272 HLS syntheses. **Set it to 10.** HLS ipgen then takes
~15 min, which is NOT the bottleneck — FIFO sizing is.

### C4. FOLDING IS THE WHOLE GAME — `export/balance_folding.py` (2026-08-06)

Fixes C3. `SetFolding` only caps cycles; it never equalises layers and never looks
at resources. `balance_folding.py` searches for the lowest
`target_cycles_per_frame` that still fits the part, reusing `SetFolding` so every
divisibility / `parallel_window` / depthwise-SWG constraint stays correct. Emits a
folding config JSON; feed it back via `folding_config_file`
(`step_apply_folding_config`) — the same route both reference designs used.

Measured on **pico**:

| config | max_cycles | FPS @100 MHz | LUT | flatness (max/median) |
|---|---|---|---|---|
| `--target-fps 30` | 3,115,008 | 32 | 19,472 (7.1%) | ~6× |
| balanced, wwidth 36 | 389,376 | 257 | 101,418 (37%) | 2.0× |
| **balanced, wwidth 144** | **174,726** | **572** | 119,128 (43.5%) | **1.8×** |

**18× throughput for 6× LUT.** `--target-fps 30` was leaving almost everything on
the table; it is a floor, not a target, and **should never be used for a real
build**.

**`mvau_wwidth_max` is a second, nearly free lever.** `SetFolding` stops raising an
MVAU's SIMD once `weight_bits * SIMD > mvau_wwidth_max` (default **36**). At **W8
that caps SIMD at 4** — and pinned `MVAU_hls_0` at SIMD=3, since its MW=27 has
divisors 1/3/9/27 and 9 would need 72 bits. The W4A4 reference designs get SIMD 9
from the same default, which is why they never hit this. Raising it to 144 bought
**2.2× for +17% LUT**.

**pico is now at the pixel-rate floor.** Six layers sit at exactly **173,056 =
416²** cycles — one pixel per clock, the hard limit for a 416×416 input. The
bottleneck moved off the MVAUs to `ConvolutionInputGenerator_rtl_0`. Further
folding cannot help; only a smaller input or >1 pixel/clock would. (Reference
design for calibration: 462 FPS est. at 216,320 cycles, same regime.)

**n_eighth, the net we actually care about** (all post-`minimize_bit_width`):

| config | max_cycles | FPS @100 MHz | LUT (est) | BRAM_18K |
|---|---|---|---|---|
| `--target-fps 30` | 3,115,008 | 32.1 | 77,927 (28.4%) | 348 (19.1%) |
| balanced, wwidth 36 | 1,168,128 | 85.6 | 99,076 (36.1%) | 372 (20.4%) |
| **balanced, wwidth 144 — PREFERRED** | **529,200** | **189** | **129,462 (47.2%)** | 381 (20.9%) |
| balanced, wwidth 144 (fastest) | 275,072 | 363.5 | 205,268 (74.9%) | 474 (26.0%) |

11.3× throughput is *available*; **take the 189 FPS point, not the 363**. At 74.9%
estimated LUT the fastest config becomes ~320k after the ×1.56 correction below —
past the ZCU102's 274,080 before the Zynq shell, DMA or FIFOs. 189 FPS is 5.3 ms a
frame, far inside the 50–100 ms budget, so the extra throughput buys nothing we
need. Flatness 13.5× → 3.2×, top 8 layers within 6%, bottleneck moved off the MVAUs
to `ConvolutionInputGenerator_rtl_12`.

**Trust the LUT number less than the others.** The Z7020 paper measured 26,694
estimated vs **41,605 actual** (**×1.56**) — though note their BRAM was exhausted
and Vivado spilled BRAM into LUTs, so it may be pessimistic. It is the only measured
point we have, so budget with it. **75% estimated is NOT a safe ceiling.**

**Two traps in this script, both of which produced confidently wrong answers:**

1. **Estimate the right snapshot.** `step_specialize_layers.onnx` has not been
   through `step_minimize_bit_width`, which the build runs *immediately after*
   folding and which is worth **~3× in LUT** (n_eighth: 252,349 vs 77,927 for the
   identical model). It must run after folding — accumulator width depends on SIMD.
   `evaluate()` replicates the whole sequence; without it every n_eighth config
   looks infeasible at >92% LUT.
2. **Resources are NOT monotonic in the cycle target, so do not bisect.** Fully
   folded n_eighth costs *more* LUT than partially unfolded: at SIMD=PE=1 the weight
   memories are 8 bits wide and very deep, FINN declines to put them in BRAM, and
   they spill to LUTRAM. Usage is U-shaped and bisection can converge on a false
   infeasibility. pico happens to be monotonic, which is exactly why this hid there
   first.

**`max/median` is a weak flatness metric on a deep net.** n_eighth reports 13.5×
unbalanced when its top 8 layers are within 12% of each other — the median is
dragged down by the many intrinsically-cheap 13×13 FPN layers, which *should* be
fast. What actually sets FIFO depth is producer→consumer mismatch **per edge**. If
the FIFOs still misbehave, measure adjacent pairs instead.

### C5. `standalone_thresholds` / DSP packing — MEASURED, and NOT worth it for us

Tested 2026-08-06 (`--standalone-thresholds`, build `n_eighth_dsp`). It works and
does what the Z7020 paper's Table 4 says — it is simply the wrong trade at our
operating point.

With `standalone_thresholds=False` (FINN's default) MatMul+MultiThreshold fuse into
one MVAU *with output activation*, which the RTL MVAU cannot express, so
`SpecializeLayers` falls back to `MVAU_hls` everywhere and HLS does **no DSP
packing**. Setting it True splits the thresholds out: all **60 MVAUs become
`MVAU_rtl`**, +57 `Thresholding_rtl` layers (245 → **302** nodes), and DSP packing
engages (2 MAC/DSP at 8-bit on DSP48E2). The join work survives it — 6 steps,
0 errors, block still contiguous.

**LUT goes FLAT at ~144,700 across the whole folding range while DSP scales
60 → 822.** That is packing working: extra compute lands in DSPs, not LUTs. But the
57 extra threshold layers are a large *fixed* cost, so it only wins once folding is
aggressive enough to amortise them:

| max_cycles | FPS | fused LUT / DSP | standalone LUT / DSP |
|---|---|---|---|
| 1,038,336 | 96 | **97,512** / 4 | 144,540 / 245 |
| 529,200 | 189 | **129,462** / 7 | 144,540 / 471 |
| 275,072 | 363 | 205,268 / 14 | **144,664** / 822 |

**Crossover ≈ 450k cycles (~220 FPS).** Above it standalone is decisive: at 363 FPS
it cuts LUT 29%, which after the ×1.56 correction is 226k (82%, tight but
buildable) versus 320k (>274,080, impossible). Below it, fused is cheaper.
**We chose 189 FPS, which is below the crossover, so keep the default `False`.**
Revisit only if the throughput requirement rises above ~220 FPS.

**Blocker if we ever do use it: URAM.** The standalone build estimates **URAM 2**,
and the XCZU9EG has **no URAM at all** — some large threshold memory is being placed
in a resource the part does not have. `balance_folding.py` correctly rejects every
such config (`BUDGET["URAM"] = 0`), which is why that sweep found nothing feasible
despite LUT sitting at 52.8%. Would need `ram_style` / `depth_trigger_uram` forced
to BRAM first.

---

## 6. D. Practical invocation notes (each cost real time to learn)

- `run-docker.sh` runs `docker build` on **every** invocation unless
  `FINN_DOCKER_PREBUILT=1`. Normally a cached no-op — but `docker builder prune`
  makes every later run rebuild ~45 layers from scratch.
- Pass `--network=host` (via `FINN_DOCKER_BUILD_EXTRA` / `FINN_DOCKER_EXTRA`).
  Container networking here times out on pypi constantly; host networking took image
  builds from dozens of 15 s timeouts to **zero**, at 19 MB/s.
- `FINN_HOST_BUILD_DIR` defaults to `/tmp/...`, and `/tmp` is a **31 GB RAM-backed
  tmpfs** on this box. **Always override to real disk.**
- FINN's own `build_dataflow_cfg` calls `pdb.post_mortem()` on any exception, so a
  failed build **hangs at a `(Pdb)` prompt** instead of exiting. A "slow" build may
  be a dead one. (`build_custom` additionally wraps in `python -mpdb`.)
- Estimate-only is the decisive mode: the compile is seconds; only
  `phase_convert_to_hardware` takes minutes.

---

## 7. Build environment

- **FPGA toolchain, installed 2026-08-05:** Vivado + Vitis + Vitis HLS **2022.2** at
  `/home/alex/Xilinx` (47 GB, Zynq UltraScale+ / Kria devices only — the default
  all-device selection needs >200 GB). `FINN_XILINX_PATH=/home/alex/Xilinx`,
  `FINN_XILINX_VERSION=2022.2`. Licence at `~/.Xilinx/Xilinx.lic`, verified by
  running `synth_design` + `place`/`route_design` on a real xczu9eg design.
  - **The tools do not start without a `libtinfo.so.5` shim** — 2022.2 links
    ncurses5, this OS ships 6 only; symlinked inside each tool's own
    `lib/lnx64.o/`, nothing system-wide.
  - ZCU102 board files already ship with Vivado at
    `data/xhub/boards/XilinxBoardStore/` (NOT the empty legacy
    `data/boards/board_files`); FINN hardcodes rev **3.3**.
  - Desktop launchers for Vivado / Vitis / Vitis HLS live in
    `~/.local/share/applications/*-2022.2.desktop`.
- **FINN: two clones, both usable.** `~/Repos/finn` (v0.10.1, the release) and
  `~/Repos/finn-dev` (dev, ~881 commits ahead). Separate Docker images, separate
  `FINN_HOST_BUILD_DIR`s (`~/finn_build`, `~/finn_build_dev`). Use **dev** for
  branched nets — see §B.
- Xilinx installer ISOs moved to `/run/media/alex/Локальный диск/Xilinx_2022.2`
  (106 GB) — `/` filled to 0 bytes and killed a build. Needed only to re-add device
  families via `xsetup -b Add`.
- **Version pin:** Vivado/Vitis **2022.2 exactly**, not latest; hold the 2022.2.2
  update. PetaLinux will be needed since ZCU102 has no stock PYNQ image
  (memory `finn-toolchain-pin`).

---

## 8. What's next

**Bitstream.** Estimate mode has given all it can (§C). The open path is:
balanced folding config from §C4 (189 FPS point, `mvau_wwidth_max=144`,
`standalone_thresholds=False`) + `--fifo-strategy none` + `NUM_DEFAULT_WORKERS=10`,
accepting ~13.8 h of stitching. Dump `final_hw_config.json` the moment anything
completes. The numbers to check against reality: LUT (expect ~×1.56 the estimate),
achieved Fmax, and true latency via `RTLSIM_PERFORMANCE`.
