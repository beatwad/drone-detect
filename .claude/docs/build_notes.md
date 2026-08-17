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
**CORRECTED 2026-08-07 — the reference design is FULLY ON-CHIP, not DDR.** This
section previously claimed "the reference streamed weights from DDR, which cost
~100k LUT of DMA + interconnect (detector 105k / 38%, full design 205k / 75%)".
Checked against their actual artifacts, that is wrong:

- **All 63 MVAUs are `mem_mode: internal_decoupled`** in every one of their config
  files (`final_hw_config_90fps.json`, `final_hw_config.json`,
  `yolov8_output_dir/*`). FINN's own source
  (`matrixvectoractivation.py:86-93`) defines this as *"streaming weights with
  streamer packaged **inside** the IP"*. The off-chip modes are `external` /
  `external_mem`, which appear **nowhere** in their configs or build script.
- **Their BRAM budget confirms it independently:** weights are
  2,250,240 × 4b + 895,920 × 8b = **16.17 Mbit**; they allocate **1,064 BRAM_18K
  = 18.7 Mbit** (58% of the part). That is the whole weight set held on-chip plus
  activation buffers. A DDR-streaming design would not need 58% of block RAM.

The 105k/205k LUT figures may come from the ARC paper itself (**which we do not
have a copy of** — only the Electronics/Z7020 one), but the published repo build
is unambiguously on-chip.

**This is good news:** the reference reached 195.3 FPS @300 MHz on a ZCU102
**without** spending ~100k LUT on DMA + interconnect, so that overhead is not a
cost we must plan around. Keep weights on-chip — ZCU102's 32.1 Mbit makes it easy,
and upstream external mem mode also carries a v0.10.1 release-note warning of
"unexpected behaviour".

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

**SUPERSEDED 2026-08-06 — `largefifo_rtlsim` was not the problem; unbalanced
folding was.** This section originally concluded "both automatic strategies are
dead ends, use `none`". The first half is wrong. Re-run on pico with a balanced
folding config (§C4) and the *default* `largefifo_rtlsim` sized every FIFO
happily:

| | unbalanced | balanced |
|---|---|---|
| deepest FIFO | 289,032 | **1,627** |
| 2nd deepest | 243,514 | 1,515 |
| sum of all depths | 565,670 | **11,255** |
| over Vivado's 32,768 max | 2 | **none** |

178× shallower. Those two giant buffers existed only to absorb the 6× rate
mismatch `--target-fps 30` created; flatten the pipeline and the need disappears.
The build walked straight through the step that had killed the previous attempt.

**What is still true:**
- `characterize` remains a genuine dead end on **joined** nets — the guard above
  is structural, not a tuning issue. Fine for join-free nets like pico.
- The **stitch cost is unchanged**. Balancing does *not* shrink the block design:
  balanced pico stitched 118 cells vs 114 unbalanced, because the FIFO *count* is
  the same (57) — `RemoveShallowFIFOs` uses `shallow_threshold=0`, so none are
  dropped; only depths shrink. `largefifo_rtlsim` still stitches TWICE.
- So `--fifo-strategy none` is still worth choosing on n_eighth, but for **time**
  (one stitch instead of two, ~13.8 h vs ~27 h) — not because the default is
  broken. Its cost stands: FIFO BRAM excluded, so utilisation is a floor.

**Caveat on the evidence:** pico has **0 joins**. Balanced folding + the default
strategy is proven on pico, NOT yet on n_eighth's 20 joins.

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
accepting ~13.8 h of stitching. (`none` for the one-stitch time saving — the
default `largefifo_rtlsim` now *works* once folding is balanced, it just stitches
twice.) Dump `final_hw_config.json` the moment anything completes. The numbers to
check against reality: LUT (expect ~×1.56 the estimate), achieved Fmax, and true
latency via `RTLSIM_PERFORMANCE`.

**Two things learned the hard way on 2026-08-06, both now handled in
`finn_build.py`:**
- **The licence does not reach the container.** `run-docker.sh` mounts
  `FINN_XILINX_PATH`, the build dir and `.ssh` — but **not `~/.Xilinx`**, and it
  never sets `XILINXD_LICENSE_FILE`. Estimates, HLS and stitching need no
  Synthesis feature, so this stays invisible until the first in-container
  `synth_design`, which then dies with
  `ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis'`.
  Pass `-v $HOME/.Xilinx:$HOME/.Xilinx -e XILINXD_LICENSE_FILE=$HOME/.Xilinx/Xilinx.lic`
  via `FINN_DOCKER_EXTRA`.
- **Resume instead of rebuilding.** `--start-step phase_generate_outputs` (dev
  takes PHASE names) re-runs only stitch + synthesis from
  `intermediate_models/phase_build_hardware.onnx`, turning a 4.5 h redo into
  ~45 min. Needs the same `--out` directory.

---

## 9. GROUND TRUTH: FINN's LUT estimate is ×2.73 low (measured 2026-08-06)

First design ever taken through real Vivado synthesis: **pico, balanced folding,
572 FPS config** (`mvau_wwidth_max=144`, 174,726 cycles). Synthesis completed;
**place-and-route refused the design.**

| | FINN estimate | Vivado post-synth | ratio |
|---|---|---|---|
| **LUT as Logic** | 118,916 | **324,702 (118.5%)** | **×2.73** |
| CLB LUTs (incl. LUT-as-memory) | — | 344,766 (125.8%) | — |
| CARRY8 | — | 35,715 (104.3%) | — |
| CLB Registers | — | 205,935 (37.6%) | — |
| BRAM (18K equiv) | 255 | 372 | ×1.46 |
| DSP | 2 | 7 | — |

    ERROR: [DRC UTLZ-1] LUT as Logic over-utilized: requires 332,652,
    only 274,080 available.  ERROR: [Vivado_Tcl 4-23] Placer not run.

**The ×1.56 from Electronics 2025 14:3993 does not transfer, and it is obvious in
hindsight why:** their design put its MACs in **DSPs** (204 of 220, RTL MVAU with
packing). Ours puts every MAC in **LUTs** (`MVAU_hls`, `resType: lut`, 7 DSPs of
2,520). We are measuring the cost of doing 8-bit multiply-accumulate in fabric.

**Budget rule: keep estimated LUT under ~70,000** for a ~70% real fit. Search with
`balance_folding.py --headroom 0.27` (0.75 / 2.73) rather than the 0.75 default.

**What actually fits on a ZCU102 at ×2.73:**

| config | est LUT | real LUT | verdict |
|---|---|---|---|
| pico balanced 572 FPS | 118,916 | 324,702 (118%) | **NO — measured** |
| pico balanced 257 FPS | 101,418 | 276,923 (101%) | no |
| pico @283 FPS | 66,130 | 180,569 (66%) | yes |
| pico @144 FPS | 41,401 | 113,046 (41%) | comfortable |
| n_eighth `target_fps 30` (32 FPS) | 77,927 | 212,781 (78%) | marginal |
| n_eighth balanced 85.6 FPS | 99,076 | 270,529 (99%) | no |
| **n_eighth balanced 189 FPS** | 129,462 | **353,498 (129%)** | **NO** |
| n_eighth @24 FPS | 77,434 | 211,435 (77%) | marginal |
| n_eighth @16 FPS | 73,959 | 201,946 (74%) | yes |

**So n_eighth tops out at roughly 24–32 FPS on a ZCU102 with this architecture** —
not the 189 FPS §C4 recommended. That recommendation is withdrawn.

**This REOPENS §C5 (DSP packing), which was closed on uncalibrated numbers.** The
conclusion "standalone_thresholds is not worth it below ~220 FPS" compared two LUT
*estimates*, and we now know the LUT estimate is the unreliable one — precisely
because our MACs live in LUTs. Moving compute to DSPs attacks the exact resource
that is over budget, and the standalone-thresholds sweep showed LUT going **flat
at ~144.7k** while DSP scaled 60 → 822. Whether that flat number carries the same
×2.73 is unknown and worth measuring: if its LUT is fixed overhead rather than
compute, the multiplier may be much smaller. **Do not re-close §C5 without a
synthesis run.**

**Also learned:** `CARRY8` hit 104% — a resource FINN does not model at all.
Watch it; adders and accumulators generate carry chains that no estimate predicts.

### 9.1 The ZCU102 is not the problem — our arithmetic is

Direct comparison against the reference YOLOv8n ZCU102 build (`mdanilow/finn`
`yolov8_dev`, `notebooks/experiments/yolov8/yolov8_output_dir/`):

| | reference YOLOv8n | our n_eighth |
|---|---|---|
| input | 320×192 | 416×416 |
| **total MACs** | **2,622,873,600** | 238,081,792 (**11× less**) |
| layers | 259 | 245 |
| quantisation | W4A4 mixed | W8A8 |
| MVAU impl | **63 × `MVAU_rtl`** | 57 × `MVAU_hls`, `resType: lut` |
| **DSP** | **603** | **3–7** |
| LUT (est) | 140,751 | 77,927–129,462 |
| clock | 300 MHz | 100 MHz |
| FPS | 195.3 | 32 |

**They run 11× our compute for the same LUT.** It is not architecture — their layer
count is *higher*. It is that their multiplies live in 603 DSPs at 4 bits and ours
live in LUT fabric at 8 bits.

Their hand-tuned `final_hw_config_90fps.json`: all 63 MVAUs `resType: 'auto'`,
`mem_mode: internal_decoupled`, and **PE = 1–4, SIMD = 9**. They barely unfold —
throughput comes from 300 MHz + DSP packing, not parallelism. Our
`balance_folding.py` was buying FPS with the most expensive resource on the chip.

**SIMD=9 is `mvau_wwidth_max / weight_bits` = 36/4** — independent confirmation of
the §C4 finding. W8 gets 4, W4 gets 9.

**DSPs are not 4-bit-only** (a DSP48E2 is a 27×18 multiplier). Bit width sets
*packing density*, per the Z7020 paper Table 4 — RTL DSP48E2: **4 MAC/DSP at W4A4,
2 MAC/DSP at W8A8**, HLS: 1 MAC/DSP with no packing either way. Our 7 DSPs are
caused by `MVAU_hls` + `resType: lut`, **not** by being 8-bit.

**Measured ceiling of the current architecture:** `balance_folding --headroom 0.27`
(i.e. against the real ×2.73 LUT) picks **16.1 FPS** for n_eighth at 73,959 est /
~202k real LUT (74%). Anything faster overflows the part. **16 FPS is below
requirement, so W4 + RTL MVAU is not an optimisation — it is the only route to a
usable frame rate on this board.**

**Plan (each factor is multiplicative, all four are currently set the expensive
way):**
1. **W4 weights** — SIMD 4→9 free; 2→4 MAC/DSP if activations are also 4-bit.
2. **`standalone_thresholds=True` + `resType auto`** → RTL MVAU → multiplies leave
   the LUTs. Reopens §C5, which was closed on uncalibrated W8A8 numbers.
3. **200–300 MHz** — the reference ran 300 on this exact part; we assumed 100.
4. **Modest folding (PE 1–4)** — stop buying FPS with LUTs.

**Accuracy gate first.** n_eighth **W4A8** is measured (QAT close-range 0.9856).
Full **W4A4** — which is what unlocks 4 MAC/DSP — is **unmeasured on n_eighth**,
and on pico 8-bit beat 4-bit on every regime (memory `pico-w8a8-beats-w4a8`). So
run W4A4 QAT before banking the 4 MAC/DSP figure; W4A8 alone still gives SIMD 9
and 2 MAC/DSP.

### 9.2 W4A4 QAT on n_eighth — MEASURED 2026-08-07, and it passes

Run: `qat/train_qat.py --weights runs/train/n_eighth_416/weights/best.pt
--weight-bits 4 --act-bits 4 --imgsz 416`, 30 epochs, best at epoch 27 →
`runs/qat/n_eighth_qat_w4a4/best.pt`. All **20 joins stayed tied** through the
switch, so the FINN precondition survives at 4 bits.

**W4A4 PTQ collapses** (close 0.558 vs 0.983) — 4-bit needs QAT, as expected.

Per-regime, W8A8 vs W4A4 QAT (both `--imgsz 416`):

| regime | W8A8 mAP50 | W4A4 mAP50 | Δ | W8A8 mAP50-95 | W4A4 mAP50-95 |
|---|---|---|---|---|---|
| **close** | 0.9845 | **0.9738** | **−1.1 pt** | 0.6793 | 0.5812 |
| **mid** | 0.9777 | **0.9295** | −4.8 pt | 0.6352 | 0.5060 |
| long | 0.8608 | 0.6859 | −17.5 pt | 0.4727 | 0.2874 |
| overall | 0.9294 | 0.8450 | −8.4 pt | 0.5752 | 0.4402 |

Loss scales inversely with target size — 4-bit activations wash out the fine
spatial gradients small objects depend on. The long-range collapse lands in the
regime the optics plan already designs away.

**Centre error (`scripts/center_error.py`) — the metric that actually matters,
since mAP50-95 is dominated by box extent, not centre:**

| model | regime | mean px | p95 px | mean frac | n_TP |
|---|---|---|---|---|---|
| W8A8 | close | 15.30 | 49.93 | 0.0151 | 1610 |
| **W4A4** | close | **17.71** | 52.73 | 0.0176 | 1554 |
| W8A8 | mid | 6.44 | 21.42 | 0.0061 | 1755 |
| **W4A4** | mid | **7.73** | 24.07 | 0.0074 | 1676 |

+2.4 px at close (~16%), but only **+0.0025 of frame width** — about **+0.04° of
aim error** at the 10–14° effective FOV, ~+0.11° at 45°. CLAUDE.md already
accepted +0.26° choosing between pico and yolov5n. `n_TP` differs by 3–5%, so the
comparison is fair.

**VERDICT: W4A4 is good enough.** The mAP50-95 drop is box-extent sloppiness, not
centre displacement. Costs ~1 pt close-range mAP50 and <0.1° aim error; buys
**SIMD 9 (vs 4)** and **4 MAC/DSP (vs 1)** — the two levers that lift n_eighth off
its measured 16 FPS ceiling (§9).

**Not measured: n_eighth W4A8.** The 0.9856 figure elsewhere in this repo is from
the *yolov5n* phase-4 work, not n_eighth. W4A8 remains the conservative fallback
(8-bit activations, still SIMD 9, but 2 MAC/DSP not 4) if W4A4 ever proves
insufficient in the field.

**Tooling:** `scripts/center_error.py` gained a `load_any()` branch — QAT
checkpoints hold a state_dict, not a pickled model, so `attempt_load` inside
`DetectMultiBackend` dies with `KeyError: 'model'`. It now rebuilds the quantized
graph via `load_and_quantize` when it sees `{state_dict, weight_bits}`.

---

## 10. Reference YOLOv8n on ZCU102 — use the AUTHORS' fork, 2026-08-10

**The single most important fact: neither FINN checkout works alone.**
`~/Repos/finn-dev` builds the HLS fine but dies assembling the Zynq shell;
`~/Repos/finn-mdanilow` (their fork) does the reverse. The working combination is
**their fork + one file from dev's finn-hlslib**.

### 10.1 finn-dev is ~2000 commits past what the reference validated
Their branch `yolov8_dev` bases on FINN `60ccf026` (2025-08-28). Running their
`build_yolov8.py` on our finn-dev needed three hand patches, and then failed
anyway:
- `ApplyConfig` moved `qonnx.transformation.general` → `finn.transformation.general`
- `step_make_pynq_driver` renamed `step_make_driver`
- **`InferAddStreamsLayer` was generalised** and now absorbs the tail `Mul`/`Add`
  into `ElementwiseMul`/`ElementwiseAdd`. That strands each output `Transpose` as
  a lone non-FINN node *between* hw nodes, and
  `step_create_dataflow_partition` asserts *"cycle-free graph violated: partition
  depends on itself"*. On their pin the tail stays float and partitions out
  cleanly with the Transposes. Dev also has `step_transpose_decomposition`, which
  is probably the supported fix.
- **Then: `ERROR: [BD 5-336] ... locked IPs: top_StreamingDataflowPartition_1_0`
  at `make_zynq_proj.py:284`, 36 h in.** This was **NOT** a dev-drift bug — it
  reproduced identically on the authors' fork. Root cause and fix in §10.6. Note
  `report_ip_status` on a re-opened project is **useless** — the `.xpr` persists
  *zero* `ip_repo_paths` (all 700+ are set in-memory by `ip_config.tcl`), so a
  re-open reports "IP definition not found" regardless.

### 10.2 Their fork's finn-hlslib predates a fix it needs — ALL concats crash
`ipgen` fails on **every one of the 10 `StreamingConcat` layers** (33 other HLS
layers and 136 RTL layers are fine), with
`clang: error: unable to execute command: Segmentation fault`. Not datatype- or
size-related: it hits `ap_uint<4>` 3-input concats and `ap_int<21>` 2-input alike.

**Cause:** their pin `HLSLIB_COMMIT=a9d64bc0` predates upstream `7ab5ad9`
(2025-07-14) *"Explicit inlining to help Vitis HLS 2022.2 to process code"*,
which adds `#pragma HLS inline` to the two `PackReader::read_nb` methods in
`concat.hpp`. Without them the recursive variadic template segfaults the frontend.

**Fix (REQUIRED, and fragile):**
```
cp ~/Repos/finn-dev/deps/finn-hlslib/concat.hpp \
   ~/Repos/finn-mdanilow/deps/finn-hlslib/concat.hpp
```
Signatures are identical; only the two pragmas differ. `fetch_repo` compares the
commit hash and skips, so the edit survives — but a **pin change or fresh clone
silently loses it**, exactly like the `set_folding.py` patch. Original kept at
`concat.hpp.orig`. Result after patching: **43/43 HLS layers, 10/10 concats, 0
failures.**

Two dead ends, recorded so they are not retried:
- **Dropping `step_minimize_bit_width`** (it creates an INT19+INT21 concat) —
  it also runs `RoundAndClipThresholds`; without it `thresholding_rtl` codegen
  asserts *"This value is not permitted by chosen dtype"*. Costs 1.48× LUT anyway.
- **Equalising the concat input datatypes** — pointless. FINN emits separate
  stream arguments, and the uniform overload takes `hls::stream<TI> (&src)[N]`,
  an *array*, so the variadic overload binds regardless of type uniformity.
  (If you ever do need it: `StreamingConcat` reads codegen types from its own
  `inputDataTypes` **node attribute**, not from the tensors.)

### 10.3 The shipped ONNX is 32% dead logic, and is a stride-8-only COCO detector
`quantyolov8_4w4a_comact_tidy.onnx` wires only the stride-8 head to `global_out`;
the P4/P5 head tails dangle. **86 of 269 nodes are unreachable** — 22 of 63 MVAUs,
18 of 57 Thresholding, 14 CIG, 14 FMPadding. Their `final_hw_config_90fps.json`
lists all 63 MVAUs, so **the paper's 140,751 LUT / 603 DSP include dead logic.**
The FPN itself is intact (2 Upsample, 3 SPPF Pool survive) — only the extra
prediction heads are orphaned. Output is `[1,144,24,40]` = 64 DFL + **80 COCO
classes**: it does not detect drones.

Pruning by reachability is behaviour-preserving (verified: live subgraph of the
original vs pruned graph identical in op type *and* weight bytes across all 183
nodes) and roughly halves the design:

| | unpruned | pruned | ZCU102 % |
|---|---|---|---|
| LUT | 126,200 | **58,609** | 21.4% |
| BRAM_18K | 1,090 | **507** | 27.8% |
| DSP | 426 | **315** | 12.5% |
| FPS | 90.42 | 90.42 | — |

Do it: BRAM is the binding constraint (60% *before* FIFOs are sized), and cell
count drives the quadratic BD stitch — 567 cells pruned vs 902 unpruned.

**FINN's `estimate_layer_resources` excludes FIFOs.** Both times I quoted a BRAM
figure from it, it was wrong.

### 10.4 Environment gotchas for their fork
- `FINN_DOCKER_GPU=0` — their older `run-docker.sh` enables `--gpus all` whenever
  `docker info | grep nvidia` matches; the nvidia runtime is unusable here and
  FINN needs no GPU. Without it: *"could not select device driver"*.
- It has **no `exec` subcommand**, and its bare-args path re-expands `"$@"`
  unquoted, so `run-docker.sh bash -c "cd X && cmd"` is shredded. Put commands in
  a script file and run `run-docker.sh bash /path/script.sh`.
- `-t --tty` is hardcoded, so any backgrounded run dies with *"the input device is
  not a TTY"*. Wrap in `script -qec "..." /dev/null`.
- Separate `FINN_HOST_BUILD_DIR=/home/alex/finn_build_mdanilow` and its own image
  tag, so the finn-dev setup is untouched.
- Their fork uses **PyVerilator** for rtlsim (dev dropped it for `finn_xsi`): it
  compiles the design to a native multi-threaded binary, so `step_set_fifo_depths`
  shows a long `g++` link then a `Vfinn_design_wrapper` process at ~400% CPU with
  **no log output for a long time**. That is normal, not a hang.

### 10.5 Where the build stopped, 2026-08-11
`/home/alex/finn_build_mdanilow/yolov8/`, scripts `build_zcu102_bit5.py` +
`run_bit5.sh`, log `bit5.log`, output `yolov8_zcu102_bit5/`. Deviations from the
authors' script: `BOARD="ZCU102"`, model path, output dir, `step_yolov8_prune_dead`
inserted after `convert_to_hw_layers`, and `step_measure_rtlsim_performance` /
`step_out_of_context_synthesis` dropped.

Reached **`step_set_fifo_depths [13/17]`, 0 failures**: ipgen complete (43/43 HLS
IP), stitch complete (567/567 cells), Verilator FIFO sim running ~40 min when
stopped by request. Remaining: finish rtlsim → second stitch → `synthesize_bitfile`.
Verify the `concat.hpp` patch before any run —
`grep -c "pragma HLS inline" ~/Repos/finn-mdanilow/deps/finn-hlslib/concat.hpp`
must be **3**.

**RESUME, don't restart.** Their fork *does* support `start_step`
(`build_dataflow.py:81-112` overrides the input model from the saved
intermediate) — an earlier note here claimed otherwise and cost 35 min of
redundant re-run. Use `build_zcu102_resume.py` + `run_resume.sh`, setting
`RESUME_FROM` to the first step not yet completed:
- `"step_set_fifo_depths"` after ipgen — needs `intermediate_models/step_hw_ipgen.onnx`
- `"step_create_stitched_ip"` after FIFO sizing — skips the rtlsim *and* its stitch
- `"step_synthesize_bitfile"` after stitching

It must write to the **same** `output_dir`, and the `code_gen_ipgen_*` dirs (1682
of them) must survive — the model references generated IP by path. Resume is
**step-granular only**: a crash mid-stitch loses that whole stitch, and the
`ZynqBuild` re-stitch inside `step_synthesize_bitfile` happens regardless.

**Stitch cost, measured — the dominant term and consistently under-estimated.**
567 cells took **~11 h** (148 cells in the first 30 min, the remaining 419 over
10.6 h): strongly quadratic in cell index, per §6. The flow stitches up to
**three** times — `step_set_fifo_depths` (rtlsim characterisation),
`step_create_stitched_ip` (final depths), and `ZynqBuild` per partition. Budget
**20–30 h** for a full bitfile at this size, which is why the finn-dev attempt
ran 36 h. Cutting cell count is the only real lever.

### 10.6 `BD 5-336` — root-caused and fixed, 2026-08-12
Not a finn-dev drift bug (it reproduced on the authors' fork). **FINN registers
one `ip_repo_paths` entry per generated IP** — 700 separate repository
directories, each set in-memory by `ip_config.tcl` immediately before its
`create_bd_cell`. Vivado's IP catalog does not survive that: by the time the
partition wrapper is instantiated, its own definition has been evicted, so
`create_bd_cell` "succeeds" but yields a **locked** cell, and `validate_bd_design`
then fails with `BD 5-336` / `BD 5-390`.

**Fix — one consolidated repository.** Symlink every IP directory into a single
folder and register that folder *once*, before the first `create_bd_cell`:

```
mkdir -p /home/alex/finn_build_mdanilow/ip_repo_all      # 701 symlinks
# 700 from the absolute paths in ip_config.tcl, PLUS:
ln -s ~/Repos/finn-mdanilow/finn-rtllib/memstream /home/alex/finn_build_mdanilow/ip_repo_all/memstream
```

**`memstream` is the one that matters and the easy one to miss** — it is written
as `$::env(FINN_ROOT)/finn-rtllib/memstream`, not an absolute path, so any filter
keying on a leading `/` drops it. Without it, instantiation succeeds but child-IP
generation fails on `MVAU_rtl_*_wstrm` (the weight-stream source). That single
symlink was the last blocker.

Harness: `/home/alex/finn_build_mdanilow/zynq_retry/` (`ip_config_v4.tcl` is the
working one, `run.sh` + `inner.sh` drive it). **Iterate here, not through FINN** —
the zynq shell assembly is minutes while the 24 h of stitches persist on disk, so
five diagnostic attempts fit in under an hour.

### 10.7 GROUND TRUTH #2: a DSP-based design estimates ×1.44, not ×2.73
First real bitstream in this project, `zynq_retry/finn_zynq_link.runs/impl_1/top_wrapper.bit`
(26.5 MB, 2026-08-12), pruned reference YOLOv8n W4A4 @ 90 FPS target, XCZU9EG.

| resource | **real (post-route)** | % ZCU102 | FINN estimate | ratio |
|---|---|---|---|---|
| CLB LUT | **84,364** | 30.8% | 58,609 | **×1.44** |
| — as logic | 54,830 | 20.0% | | |
| — as memory | 29,534 | 20.5% | | |
| CLB Registers | 63,132 | 11.5% | — | |
| **CARRY8** | 1,583 | **4.6%** | not modelled | |
| **Block RAM Tile** | **725.5** | **79.6%** | 253.5 (=507 BRAM_18K) | **×2.86** |
| DSP48E2 | 359 | 14.3% | 315 | ×1.14 |
| CLB | 16,867 | 49.2% | | |

Timing closed with room: **WNS +3.282 ns** at a 10.0 ns period, TNS 0.000, **0 of
392,283 endpoints failing**; hold met (WHS +0.009). Critical path ≈ 6.72 ns, i.e.
**~149 MHz achievable** → ~134 FPS rather than the 90.42 quoted at 100 MHz.

Three corrections to §9 that follow from this:
1. **The ×2.73 LUT multiplier is specific to LUT-based arithmetic.** With MACs in
   DSPs (`MVAU_rtl`, W4A4) it drops to **×1.44**. The "keep estimated LUT under
   ~70,000" rule was derived from ×2.73 and is far too conservative for this class.
2. **CARRY8 stops being the wall.** 4.6% here vs **104%** on n_eighth W8A8 — that
   was what made placement refuse. Same cause: adder trees in fabric.
3. **BRAM is now the binding constraint, and the estimate is ×2.86 low** because
   `estimate_layer_resources` excludes FIFOs. 79.6% used means the *unpruned*
   design (×2.15 the BRAM estimate) would not have fitted. Pruning was a
   precondition for building, not an optimisation.

**Power, post-route: 5.230 W total — of which PS8 is 2.736 W.** The whole fabric
(CLB 0.382 + signals 0.344 + BRAM 0.609 + DSP 0.196 + clocks 0.224) is **1.755 W**.
That is the Z7020 power shape again on a much bigger part, and it confirms the
locked "Power shape" decision: **PS involvement, not fabric size, dominates.**

### 10.8 The compiled topology, recovered exactly — 2026-08-12
To retrain the reference for drones we need the architecture the bitstream was
actually built from, not stock yolov8n. Recovered by reachability analysis on
`quantyolov8_4w4a_comact_tidy.onnx` and reproduced as
**`configs/yolov8n_p3_relu.yaml`**.

**It is stock yolov8n layers 0–15 plus a single stride-8 head.** §10.3 said "the
FPN is intact, only the extra prediction heads are orphaned" — that was
imprecise. The whole **bottom-up PAN path is dead too** (stock layers 16–21:
Conv/Concat/C2f ×2), because 18 and 21 feed nothing but the P4/P5 head branches.
What runs is backbone + top-down FPN + `Detect([15])`.

- 41 live Conv of 63. Verified by `scripts/check_v8_topology.py`: **identical
  multiset of 41 conv weight shapes** at nc=80. Compare on the multiset, not the
  sequence — `named_modules()` walks C2f in declaration order while ONNX is
  topological, and the exporter interleaves Detect's independent cv2/cv3 branches.
- The head must be **legacy** (`Conv 3×3 → Conv 3×3 → Conv2d 1×1`). Ultralytics
  8.3.253 defaults `Detect.cv3` to a DWConv head; `parse_model` sets
  `legacy=True` for v8 yamls, so a v8 yaml is right and a v11 one would not be.
- At **nc=1 the head narrows**: `c3 = max(ch[0], min(nc, 100))` is 64 for one
  class vs 80 for COCO, so the three cv3 convs shrink and the output goes
  `[1,144,H,W]` → `[1,65,H,W]`. Everything else is shape-identical. This is the
  only unavoidable deviation from "one-to-one".
- Output is the **raw** `[1, 4·reg_max + nc, 24, 40]` head tensor. DFL, decode
  and NMS run off-chip (`simple_yolov8_driver.py`) — do not export them.

**Precision map, read off the graph — it is NOT uniformly 4-bit.**

| | weights | activation |
|---|---|---|
| stem conv (`model.0`) | INT8 | UINT8 |
| the six `Detect` convs | INT8 | UINT8 |
| the other 34 convs | INT4 | UINT4 |

All activations are **unsigned** (UINT4/UINT8), which is what pins ReLU. Input is
UINT8 raw pixels. This is the recipe QAT has to reproduce; the authors' Brevitas
source (`models.finn_models.QuantC2f`, `QuantV8Detect`) is **not** in their public
fork — only the exported ONNX — so it has to be rebuilt from this table.

**Footprint: 1,610,337 params at nc=1** (vs 3,011,043 for full 3-head yolov8n),
1,603,568 conv weights = 152,048 at W8 + 1,451,520 at W4 = **7.02 Mbit
theoretical → 14.04 Mbit at FINN ×2**. That is the **7–23 Mbit band**, i.e. a
K26-class part — so the built network *does* port off the devkit. An earlier note
calling YOLOv8n devkit-only was costing the full 3.2M-param net, not this one.

### 10.9 "comact" = ONE common activation range, and it must be trained in
Measured on the reference graph 2026-08-12. Every live `Mul`-by-constant in it is
either a per-channel weight scale (41, one per conv) or a **per-tensor activation
scale — and there are only two distinct values in the whole network**:

| activations | scale | = | range |
|---|---|---|---|
| 34 × UINT4 | 0.40000004 | 6/15 | **[0, 6]** |
| 5 × UINT8 | 0.02352941 | 6/255 | **[0, 6]** |

Same range everywhere, differing only by bit width. That is what `comact` in
`quantyolov8_4w4a_comact` means: **common activation**. Note also that weights are
**per-channel** quantized, not per-tensor as in our yolov5 pipeline.

**This removes the hardest part of the yolov5 recipe.** With one common activation
range every join is at the same scale by construction, so none of
`qat/quantize.py`'s `SharedQuant` machinery — the thing §4 and CLAUDE.md call the
FINN precondition — is needed. There is nothing to tie.

**But the network has to be trained into [0, 6], and ours was not.** Measured on
`runs/train/v8n_p3_relu` over 256 close-regime images: **all 39 Conv blocks exceed
6**, median site max **21.7**, worst 210 (`model.1`), and **10–13% of values in the
early layers sit above 6**. Clamping that to a fixed [0, 6] at QAT time throws most
of the early signal away.

Fix: train the float model with **`nn.ReLU6()`** —
`configs/yolov8n_p3_relu6.yaml`, verified 41/41 against the compiled graph. Do not
try to reach the reference's activation quantization from a plain-ReLU checkpoint;
it is the same class of mistake as pasting ReLU into SiLU-trained weights (§1).

**The clamp is free — measured 2026-08-13.** `runs/train/v8n_p3_relu62` (ReLU6)
vs `runs/train/v8n_p3_relu` (plain ReLU), identical recipe otherwise: close mAP50
0.9855 vs 0.9878, mid 0.9893 vs 0.9929, **long 0.8691 vs 0.8643**, close centre
error 0.0139 vs 0.0141 at -0.9% n_TP. A few tenths of a point on close/mid, half a
point *gained* on long, aim error unchanged. Train into the range; do not clamp
afterwards.

**One trap the fix for §10.5's output path created.** Ultralytics reuses
`args.project` as the **W&B project name**
(`wb.init(project=str(trainer.args.project).replace("/", "-"))`), so making
`project` absolute — required for the output dir to land correctly — sends the run
to a project called `-home-alex-Repos-drone-detect-runs-train`. Its callback is
guarded by `if not wb.run:`, so the fix is to call `wandb.init()` yourself before
`model.train()` and let Ultralytics adopt the run.

### 10.10 W4A4 QAT on the drone net — PASSES, 2026-08-13
`qat/quantize_v8.py` + `qat/train_qat_v8.py`, 30 epochs at lr0 0.002 from the
ReLU6 float run, ~50 min. Best at epoch 28, `runs/qat/v8n_p3_w4a4`.

| | close mAP50 | close mAP50-95 | mid mAP50 | long mAP50 | long mAP50-95 | close centre err |
|---|---|---|---|---|---|---|
| float ReLU6 | 0.9855 | 0.7128 | 0.9893 | 0.8691 | 0.4848 | 0.0139 |
| W4A4 PTQ | 0.9880 | 0.6737 | — | 0.7663 | 0.3540 | — |
| **W4A4 QAT** | 0.9835 | 0.7099 | 0.9879 | 0.8514 | 0.4733 | **0.0139** |

PTQ alone loses 3.9 pt of close mAP50-95 and 10.3 pt of long-range mAP50; QAT
gives back 96% and 86% of that. **Aim error is unchanged** — 14.03 px vs 14.10 at
0.4% fewer matched TPs. Note close mAP50 *rises* under PTQ (0.9880) and falls
slightly after QAT (0.9835): at IoU 0.5 the metric is blind to what quantization
actually damages. Judge on mAP50-95 and centre error.

Three Brevitas/Ultralytics traps, all now handled in `qat/train_qat_v8.py`:
- **`load_state_dict` strands 120 const-scale buffers on CPU.** Build and load on
  CPU, then `.to(device)` — the same rule already recorded for yolov5. Verified
  bit-exact.
- **`final_eval` reloads `best.pt` through `load_checkpoint`** and raises
  `KeyError: 'model'` on a state_dict checkpoint. Override it to validate the
  in-memory EMA.
- **Ultralytics' W&B callback keys off global SETTINGS**, not the trainer, so a
  `--no-wandb` flag must strip the callbacks rather than skip its own init.

### 10.11 QONNX export of the drone net — matches the reference graph, 2026-08-13
`export/export_qonnx_v8.py` → `export/v8n_p3_w4a4_192x320_clean.onnx`, 227 live
nodes, input `(1,3,192,320)`, output `(1,65,24,40)` raw. **Zero unsupported ops.**

| op | ours | reference |
|---|---|---|
| Conv | 41 | 41 |
| BatchNormalization | 39 | 39 |
| Concat | 10 | 10 |
| Split | 6 | 6 |
| MaxPool | 3 | 3 |
| Resize | 2 | 2 |
| Add | 6 | 8 |
| Relu / MultiThreshold | 39 | 39 |
| Quant | 81 | 0 (streamlined) |

Quant histogram is the reference precision map exactly: **34 signed-4 (weights) +
34 unsigned-4 (acts) + 7 signed-8 (weights) + 6 unsigned-8 (5 acts + the input)**.
The two remaining differences are streamlining artefacts, not discrepancies: the
reference's extra 2 Adds are the head bias terms that streamlining splits out of
the Conv nodes, and its 80 Muls / 1 Div are the dequant scales our graph still
carries as Quant nodes.

**Two export-only patches, both required:**
- `RawV8Head` — Detect reduced to cv2/cv3 plus the box|cls concat. DFL, decode
  and NMS stay off-chip.
- **`C2f.forward` must become `forward_split`.** `Tensor.chunk` unrolls into
  Shape → Gather → Div/Mul → Slice (verified on opset 11 and 13), leaving 12
  Slice nodes with no FINN hardware op; `torch.split` traces to one `Split`,
  which is what the reference carries. Numerically identical — a tracing concern
  only, so patch at export, not in training. Ultralytics ships `forward_split`.

Then `qonnx.util.cleanup`: 269 → 227 nodes, and it constant-folds the 12 Adds
down to **exactly the 6 Bottleneck residuals**.

### 10.12 Build resolution: 192×320, decided 2026-08-13
Measured with `scripts/center_error.py --imgsz 192,320` (only `predict` honours a
rectangle — `val()` silently rounds `[h, w]` to an int):

| input | close n_TP | close err | long n_TP | long err |
|---|---|---|---|---|
| 320×320 | 1645 | 0.0139 | **1664** | 0.0020 |
| 192×320 | 1644 | **0.0131** | **896** | 0.0027 |

**Close is untouched** — 1 detection difference and a slightly better centre
error. **Long loses 46% of its detections.** That is the vertical squeeze pushing
small targets under the detection floor, and it is an artefact of letterboxing
*our val images* into 5:3 — in deployment the input is a centre **crop**, which
keeps angular resolution and only narrows the field. So it overstates the cost.

Decision: build at **192×320**. It is the geometry already proven to fit (79.6%
BRAM), it costs nothing in the close/mid regime we actually target, and 320×320
is 1.67× the pixels against a BRAM budget that is already the binding constraint.
Revisit once there is real footage through the chosen lens.

### 10.13 Reclaiming disk from a FINN build dir
A `FINN_HOST_BUILD_DIR` is ~95% regenerable scratch. Measured on
`~/finn_build_dev` (20 GB, 2026-08-14):

| | size | keep? |
|---|---|---|
| `code_gen_*` | 13 GB / 1515 dirs | no — HLS/RTL codegen scratch |
| `rtlsim_*` | 2.6 GB / 269 dirs | no |
| `vivado_*` | 3.2 GB / 18 dirs | no — stitch projects |
| named outputs (`n_eighth_*`, `pico_*`, `yolov8_ref*`) | **933 MB** | **YES** |

`rm -rf code_gen_* rtlsim_* vivado_*` frees 19 GB and leaves every `report/`
that build_notes cites. **Do not delete the named output dirs** — they carry the
`estimate_layer_resources.json` behind the §9 and §C5 numbers. The cost is that
those builds stop being resumable (their saved `.onnx` reference generated IP by
absolute path), which is fine for finished or abandoned ones.

### 10.14 A single failed HLS IP passes silently and kills the stitch hours later
Hit 2026-08-14 on the drone build. `step_set_fifo_depths` died with

    ERROR: [BD 5-390] IP definition not found for VLNV: xilinx.com:hls:DuplicateStreams_hls_12:1.0
    Exception: CreateStitchedIP failed, no wrapper HDL found

**This is NOT the §10.6 catalog-eviction bug.** `CreateStitchedIP` already
registers every repo in one `set_property ip_repo_paths [list ...]`. The real
cause was one layer whose IP had never been packaged: its `vitis_hls.log` held

    ERROR: [Common 17-685] Unable to load Tcl app xilinx::questa
    ERROR: [IMPL 213-28] Failed to generate IP.

**Exactly 1 of 41 HLS layers**, transient — it did not recur on re-run, so it is
most likely a race between the 10 parallel Vitis HLS processes, not a
configuration fault. **FINN does not check.** `HLSSynthIP` then logged
*"Using pre-existing IP"* for it (82 such lines in that run) and the flow carried
on for hours before the stitch tripped over the missing VLNV.

**Detect it right after ipgen** — one incomplete dir is enough to waste a day:
```
for d in $FINN_HOST_BUILD_DIR/code_gen_ipgen_*_hls_*; do
  find "$d" -name component.xml -print -quit | grep -q . || echo "NOT PACKAGED: $d"
done
```
Only test `*_hls_*`: **RTL layers never produce a `component.xml`**, so a naive
sweep reports hundreds of false positives (437 of them here, nearly all
`Thresholding_rtl_*`).

**Recovery, without rebuilding:** delete the failed dir and resume from
**`step_hw_codegen`**, not `step_hw_ipgen`. The saved `step_hw_codegen.onnx`
records the old dir in each node's `code_gen_dir` attribute, and `HLSSynthIP`
only synthesises — it is `PrepareIP`, in the codegen step, that creates the
directory. Resuming at ipgen just dies with
`FileNotFoundError: .../ipgen.sh`.

### 10.15 `BD 5-336` recurs on every build — patch it BEFORE launching, 2026-08-15
§10.6 root-caused it and built a working harness, but **nothing in FINN was
fixed**, so `step_synthesize_bitfile` walks into the same wall on every new
design. On the drone build it did so after **23 h 32 min** of codegen, FIFO
sizing and stitching, two minutes into `MakeZYNQProject`:

    ERROR: [BD 5-336] ... locked IPs: top_StreamingDataflowPartition_1_0
    Exception: Synthesis failed, no bitfile found

The failure costs almost no compute — everything before it is on disk and gets
reused — but it costs however long it takes someone to notice. **Treat the
harness as part of the recipe, not as a recovery step.**

Recipe, ~5 minutes, from the `ip_config.tcl` FINN already wrote into
`vivado_zynq_proj_*`:
1. Extract every `/home/.../` path out of the `set_property ip_repo_paths` lines
   (690 for this design) and symlink each into **one** fresh folder.
2. Add the `memstream` symlink (§10.6 — it is written via `$::env(FINN_ROOT)`,
   so any absolute-path filter silently drops it).
3. Replace all `set_property ip_repo_paths` lines with a single
   `set_property ip_repo_paths [list <folder>] [current_project]` before the
   first `create_bd_cell`, and keep exactly one `update_ip_catalog`.
4. Run it under `run-docker.sh bash <inner.sh>`; needs a pty, so wrap in
   `script -qec`.

**Use a separate folder per design.** `ip_repo_all` (reference) and
`ip_repo_drone` (ours) must not be merged: both define
`xilinx_finn:finn:StreamingDataflowPartition_1:1.0`, `MVAU_rtl_0` and so on.
Harness: `/home/alex/finn_build_mdanilow/zynq_drone/`.

FINN stays parked at its pdb prompt after the exception, so **steps 6/7 and 7/7
(`step_make_pynq_driver`, `step_deployment_package`) never run**. Generate the
driver separately from `intermediate_models/step_create_stitched_ip.onnx` — it
reads data layouts from the model and does not need the bitfile.

### 10.16 GROUND TRUTH #3: our own detector, on hardware — 2026-08-15
`zynq_drone/finn_zynq_link.runs/impl_1/top_wrapper.bit` (26.5 MB), yolov8n-P3
ReLU6 W4A4 @ 192×320, XCZU9EG, 100 MHz. Whole design incl. the PS shell.

| resource | **real (post-route)** | % ZCU102 | predicted | error | reference §10.7 |
|---|---|---|---|---|---|
| CLB LUT | **82,222** | 30.0% | ~82,900 | **0.8%** | 84,364 |
| — as logic | 53,291 | 19.4% | | | 54,830 |
| — as memory | 28,931 | 20.1% | | | 29,534 |
| CLB Registers | 60,820 | 11.1% | | | 63,132 |
| CARRY8 | 1,534 | 4.5% | | | 1,583 |
| **Block RAM Tile** | **691** | **75.8%** | ~714 | **3.2%** | 725.5 |
| DSP48E2 | 334 | 13.3% | ~342 | **2.3%** | 359 |
| CLB | 16,471 | 48.1% | | | 16,867 |

**The §10.7 calibration transferred to a different network with no adjustment**
(LUT ×1.44, BRAM ×2.86, DSP ×1.14 over FINN's estimate of 57,600 LUT / 499
BRAM_18K / 300 DSP). One measurement was a data point; two make it a rule —
budget new designs with these multipliers and expect a few percent.

Timing: **WNS +1.765 ns** at 10.0 ns, TNS 0.000, **0 of 379,557 endpoints
failing**, hold met (WHS +0.006, tight but positive). Critical path 8.235 ns →
**~121 MHz achievable**, vs the reference's 6.72 ns / ~149 MHz. **We are smaller
than the reference on every resource yet have half its slack**, so the long path
is ours, not the flow's — look at the P3 head or a wide MVAU if clock ever
matters. It does not yet: 90.4 FPS at the built 100 MHz (≈110 if clocked up)
against a 50–100 ms end-to-end budget.

Power, post-route: **5.104 W total, PS8 2.736 W**, static 0.738 W, whole fabric
1.630 W (CLB 0.359 + signals 0.315 + BRAM 0.572 + DSP 0.168 + clocks 0.216).
Third confirmation of the locked "Power shape" decision: the PS dominates.

**BRAM remains the go/no-go for a smaller part.** 691 tiles = 24.9 Mbit on-chip;
this design does not fit a ZU3EG and needs a K26-class part at minimum.
