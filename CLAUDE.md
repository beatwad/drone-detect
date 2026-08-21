# Drone-Detect — Project Memory

Close-range (≤10 m) drone detection for precision aiming, targeting an
**FPGA-accelerated** deployment. A vision system detects a large-in-frame drone
and outputs angular/positional data to a downstream kinetic aiming subsystem.
Latency budget end-to-end ~50–100 ms. Status: **Proof of Concept**.

Reproduction instructions for the whole chain are in [README.md](README.md).
Full source docs live in [.claude/docs/](.claude/docs/):
- [build_notes.md](.claude/docs/build_notes.md) — **everything measured about QAT →
  QONNX → FINN → bitstream.** Read this before touching `qat/`, `export/`, or
  running a FINN build; it is where all the hard-won toolchain detail lives.
- [research/](.claude/docs/research/) — findings from primary sources, cited.
  **Kept apart from build_notes on purpose:** build_notes is what we measured,
  research is what the docs say. A claim moves across only once we measure it.

## Pipeline (target)
`yolov8n-P3 (float, ReLU6)` → `Brevitas QAT (W4A4)` → `QONNX export` →
`FINN → bitstream` → `PYNQ deployment`. Everything up to and including QONNX
export is **board-agnostic** and done on the host GPU.

**Where we are: the whole host side is done and verified; we are waiting on
hardware.** As of 2026-08-19 the detector is a bitstream, the driver exists, the
numeric path from PyTorch to the built graph is checked end to end, and a ZCU102
Linux image is built and inspected. The board itself is **not here yet** (on its
way), so exactly one junction in the chain is untested: real hardware against the
simulation. `deploy/run_on_board.py` performs that check in one command when it
arrives.

2026-08-15, yolov8n-P3
ReLU6 W4A4 @ 192×320 built to `top_wrapper.bit` on ZCU102 — timing closed with
+1.77 ns slack at 100 MHz (~121 MHz achievable), **30.0% LUT, 75.8% BRAM, 13.3%
DSP**, 5.10 W of which PS8 is 2.74 W. The reference build that proved the path
(2026-08-12) is at 30.8% / 79.6% / 14.3%, +3.28 ns. **Prediction error was
0.8% on LUT, 3.2% on BRAM, 2.3% on DSP** — the §10.7 multipliers transferred to
a new network unchanged, so they are now a budgeting rule, not a data point.
Key facts, all detailed in [build_notes.md](.claude/docs/build_notes.md):

- **FINN compiles branched YOLO** — the old "no joins" blocker was wrong
  (refuted 2026-08-04). Joins need their branches to share one quantisation
  scale, which is **set during QAT** and is already done (20/20 tied, free).
- **Use FINN dev, not v0.10.1**, for anything with joins.
- **Folding is the whole game.** `--target-fps 30` under-folds ~11× and produces
  unbuildable FIFOs; `export/balance_folding.py` is the fix. But see the LUT
  calibration below — the old "preferred 189 FPS" config **does not fit**; run
  `balance_folding.py --headroom 0.27` to search against reality.
- **FIFO sizing: the dead end was the imbalance, not the strategy.** Balance
  folding first and the default `largefifo_rtlsim` works — verified on pico
  2026-08-06: deepest FIFO **289,032 → 1,627**, nothing near Vivado's 32,768
  limit. `characterize` *is* a real dead end on joined nets (it refuses
  two-stream adds). `--fifo-strategy none` remains useful only to halve the
  stitch time, which is unchanged and quadratic.
- **The LUT multiplier depends on where the MACs live — two measurements now.**
  LUT-based arithmetic (pico W8A8, 2026-08-06): FINN said 118,916 LUT, Vivado
  said **324,702** = **×2.73**, 118% of the device, place-and-route refused it,
  and **CARRY8 hit 104%** — the adder trees, not the LUTs, were the real wall.
  DSP-based (`MVAU_rtl`, W4A4, 2026-08-12): **×1.44**, and CARRY8 falls to 4.6%.
  So `standalone_thresholds` + W4 to unlock DSP packing is not a micro-opt, it
  moves the design into a different cost class. See build_notes §9 and §10.7.
- **BRAM is the binding constraint, and FINN under-estimates it ×2.86** because
  `estimate_layer_resources` **excludes FIFOs**. The real build sits at 79.6% of
  the ZCU102's 912 tiles. Budget BRAM at ×2.9 the estimate; it, not LUT, is the
  go/no-go for a smaller deployment part.
- **Consequence: n_eighth's 24–32 FPS ceiling was a LUT-arithmetic result**, and
  should be re-derived at W4A4 with DSP packing before it is treated as final.

> **Target framing is a lens × crop question — and "mid" is good enough.**
> A 30 cm drone at 10 m subtends only **1.72°**. *Resizing* a Full-HD frame to
> 416 puts it at ~12 px at 60° FOV — at or below the measured detection floor.
> *Centre-cropping* 416×416 out of 1920×1080 keeps native angular resolution and
> puts it at **55 px**. Same lens, same network: undetectable → comfortable.
> We do **not** need the *close* regime — measured **mid ≈ close** (yolov5n
> 0.992 vs 0.989; pico 0.975 vs 0.975). Only **long** is bad (0.904 / 0.778).
> The goal is simply to stay out of long. See open question 1.

## Status — host-side phases
0. Scaffolding — **DONE**
1. Dataset acquisition + merge — **DONE** (see `scripts/merge_dataset.py`,
   `configs/drone.yaml`, `data/drone/` + `manifest.csv`)
2. Augmentation (close-range framing) — **DONE** (in-loop via hyp; `mosaic: 0.5`)
3. Float training — gate mAP50 > 0.85 — **DONE**, mAP50 **0.963** on the
   YOLOv5n line (since removed, see below); superseded by phase 7. Historical:
   (`runs/train/more_data_5`, SiLU) and **0.963** (`relu_more_data_5`, ReLU)
4. Brevitas QAT — **DONE** (`qat/`): **INT8 needs no fine-tuning at all**
   (W8A8 PTQ mAP50 0.9628 vs float 0.9629). 4-bit weights collapse under PTQ
   (0.197) but QAT recovers close range to 0.9856. Gate met.
5. QONNX export (handoff to FINN) — **DONE**. `export/n_eighth_tied_w8a8_416.onnx`
   passes both gates: `check_join_scales.py` 20/20 tied, `verify_qonnx.py`
   faithful (32/32 boxes, 0.367 px).
5b. FINN build (estimates) — **DONE 2026-08-05**. pico on v0.10.1 (19,472 LUT /
   193 BRAM), n_eighth on FINN dev (77,927 LUT / 348 BRAM / 3 DSP at the
   `--target-fps 30` floor; 129,462 LUT at the preferred 189 FPS point).
5c. FINN build (bitstream) — **DONE 2026-08-15 for our own drone net.**
   `zynq_drone/.../top_wrapper.bit`, timing closed, real numbers in build_notes
   §10.16; the reference build that proved the path is §10.7. Recipe: the
   authors' fork + dev's `concat.hpp` + dead-logic pruning + one consolidated IP
   repository (§10.2/§10.3/§10.6). **`BD 5-336` is unfixed in FINN and will hit
   every build** — prepare the harness up front, §10.15.
7. Retarget to drones — **DONE.** yolov8n-P3 ReLU6 retrained on drones (close
   0.9855 / mid 0.9893 / long 0.8691 float; W4A4 QAT 0.9835 / 0.9879 / 0.8514
   with centre error unchanged at 0.0139), exported, built.
8. Board bring-up, host side — **DONE 2026-08-19**, build_notes §11.
   - PYNQ driver recovered (FINN never generated it — §10.17) and the deployment
     package assembled: UINT8 (1,192,320,3) NHWC in, INT21 (1,24,40,65) out.
   - The whole numeric path verified (§10.18): export vs torch = ±1 activation
     step, FINN frontend exact, **convert_to_hw bit-exact**, and the
     MultiThreshold convention costs nothing measurable on aim error.
   - PetaLinux 2022.2 image built for ZCU102 — `deploy/petalinux/` reproduces it.
     PetaLinux runs in a container (the host is Ubuntu 26.04, far past what
     kirkstone tolerates). Three settings are load-bearing and none is default:
     `MACHINE_NAME=zcu102-rev1.0`, the **GTR mux hogs the DTG's board dtsi omits**
     (without them SEL=0000 and USB 3.0 does not exist), and
     `CONFIG_USB_DWC3_DUAL_ROLE` (host-only does not link on Xilinx 5.15).
   - `deploy/run_on_board.py` compares real INT21 against a 60-frame golden set
     from the simulation, in LSB units. Tolerance 0.05 LSB — a real error is 1.0,
     float32 noise at these magnitudes is ~0.008. Positive and negative controls
     both pass on the host.
9. **NEXT, blocked on hardware arriving:** `pip install pynq` over the built XRT
   (no official PYNQ image exists for ZCU102 — the one genuinely unknown step),
   write the card with `deploy/petalinux/mksd.sh`, boot, then run
   `run_on_board.py`. After that: real FPS and power under load.

6. **The whole YOLOv5 line was removed 2026-08-20** — vendored `yolov5/`, its
   QAT and export scripts, and the `pico` / `n_eighth` / `relu` configs. It had
   not been used since the retarget to yolov8n-P3 (phase 7), and its rationale
   was already void: pico existed as insurance against the "FINN cannot compile
   branches" blocker, which was refuted. Everything it measured is recorded here
   and in build_notes; the code is in git history. `scripts/center_error.py` was
   ported off it first, and re-measured **identical to the last digit** on the
   shipping checkpoint (close 13.264 px / mid 6.908 / long 2.684, same n_TP).

## Open questions (settle these before more model work)
1. **Deployment optics = lens FOV × crop.** Much less demanding than the earlier
   "you need a ~5° telephoto" framing: since **mid ≈ close**, the only goal is to
   stay out of the **long** regime (<1% box area). Still urgent — it picks the
   C-mount lens bought with the camera.
   - **Working answer: a 30–45° full-frame lens + a 416×416 centre crop** puts a
     30 cm drone at 10 m in **mid** (1.6–3.5% area, 73–110 px). At 60° the crop
     yields 0.87% — marginally long — so 45° or narrower. Never *resize* the
     full frame: at 60° that gives ~12 px, at/below the detection floor.
   - **A 416 crop beats a 640 crop.** Smaller crop = narrower effective FOV =
     larger target fraction (at 45° full FOV: 1.55% at 416 vs 0.66% at 640).
     Smaller input is better here, and it suits a single low-stride head, which
     needs targets comfortably above its stride.
   - Prefer **camera-side ROI** to a host-side crop: ~2.6× faster sensor readout
     (416 vs 1080 rows), 12× less USB traffic (173 KB vs 2.07 MB), zero CPU work.
     Cost: the window is fixed. A *steerable* crop (centred on the Kalman
     prediction) needs full-frame transfer instead. **Unresolved fork:** fixed
     ROI = minimum latency; moving crop = wider effective capture area.
   - **Still open:** (a) **acquisition** — a 10–14° effective FOV is a narrow
     search cone, and full-frame downscaled acquisition does not rescue it
     (~12 px, below the floor); needs platform slew, an external cue, or a second
     wide sensor. (b) exact FOV depends on sensor size *and* focal length —
     compute once the camera is chosen. (c) **domain gap** — our val "mid"
     images are diverse web photos, not centre crops through one fixed lens;
     collect a few hundred real frames before trusting the numbers.
   - Bonus: a centre crop is boresight-aligned, so the centre offset the aiming
     subsystem consumes needs no coordinate transform.
2. ~~**What does pico cost once quantized?**~~ **CLOSED 2026-08-20, not by
   measurement but by removal.** pico was a YOLOv5 config and went with that
   line. It was insurance against a blocker that turned out not to exist, and
   yolov8n-P3 W4A4 is measured, built and better on every regime. Reopening it
   would mean restoring the line from git history.

## Locked decisions
- **Model source: the `ultralytics` package, pinned, nothing vendored.**
  (Was: classic `ultralytics/yolov5` v7.0 vendored at `yolov5/`. That served the
  anchor-based YOLOv5n line and was removed with it, 2026-08-20.) The topology
  lives in `configs/yolov8n_p3_relu6.yaml` and Ultralytics builds it. **The pin
  is load-bearing:** `qat/quantize_v8.py` hooks `C2f` and `Detect` internals,
  which move between releases — `non_max_suppression` already migrated from
  `ultralytics.utils.ops` to `ultralytics.utils.nms` inside the 8.3 line. After
  any version bump, re-run the export gates before trusting a build.
- **SiLU→ReLU swap timing: REVISED — train float in ReLU from the start.**
  (Was: "swap at the QAT stage, not before float training." Measurement
  contradicted it.) A retrained ReLU model costs 2.1 pt mAP50-95 and **zero**
  mAP50 / centre error vs SiLU. But *pasting* ReLU into SiLU-trained weights
  destroys the model — mAP50 0.963 → **0.078**, close range → 0.0007 — because
  ~49% of pre-activations are negative and ReLU zeroes them across 57 layers.
  So QAT must start from `runs/train/relu_more_data_5/weights/best.pt`, never
  from the SiLU model. For the shipping net this is settled by construction:
  `configs/yolov8n_p3_relu6.yaml` trains in ReLU6 from scratch.
- **Devkit (DEVELOPMENT ONLY):** **ZCU102** (XCZU9EG-2FFVB1156), chosen
  2026-08-03. 274,080 LUT / 32.1 Mbit BRAM (912×36Kb, **no URAM**) / 2,520
  DSP48E2; PS = 4× Cortex-A53 @1.2 GHz + 2× Cortex-R5F @500 MHz; 2× FMC HPC and
  **no native MIPI** (MIPI would need a camera FMC module — but see the camera
  decision below). This board is deliberately oversized — experiment freely, but
  do NOT let its headroom drive architecture decisions.
  *PS = Processing System (hard ARM cores + hard peripherals: DDR, USB, GEM,
  I2C…). PL = Programmable Logic (the fabric). They talk over AXI: HP ports for
  bandwidth, GP ports for control.*
- **Camera: USB 3.0 global-shutter machine-vision camera** (development choice).
  USB is a **PS** peripheral, so frames land in PS DDR and are then DMA'd to PL.
  The "sensor streams into PL, preprocessing fused with acquisition" path is
  therefore **NOT available** — that needs MIPI/parallel pixels on PL pins.
  Accepted: est. ~10–25 ms end-to-end on ZCU102 with C preprocessing, well
  inside the 50–100 ms budget.
  - **Trigger DEFERRED 2026-08-19** — decided to buy a camera without one for
    now (candidate: ELP AR0234 USB3, 1920×1200 global shutter, 120 fps). At
    120 fps the inter-frame interval is 8.3 ms, which bounds timestamp error
    without a trigger; that is tolerable for bring-up. The reasoning below still
    stands and the requirement returns as soon as aim error is being measured
    rather than the pipeline being proven.
  - **Hardware trigger is a REQUIREMENT, not a nice-to-have.** Drive the camera
    trigger from PL and you get an exact exposure timestamp t₀; the Kalman then
    predicts forward from t₀, so variable USB/Linux delay becomes a *measured*
    quantity instead of an error source. Without it every ms of jitter lands
    directly in aim error.
  - Also require: **ROI windowing** (sensor readout is the dominant latency
    term), **C/CS-mount** (FOV unresolved — see the lens callout), modest
    resolution (we feed 416 px; a 4K sensor just costs transfer time to discard
    pixels), maintained ARM64 SDK or plain UVC.
  - **Tension with the power target:** USB 3.0 keeps PS + Linux + DDR awake,
    which fights the ~2–5 W deployment goal (Z7020 reference: 1.9 W of 2.55 W
    was PS+DDR idle). Deployment may need MIPI-into-PL instead — so keep
    preprocessing behind an interface and do NOT let software assume a
    USB-shaped frame source.
- **Deployment target (undecided, but it constrains design):** a smaller,
  cheaper, **~2–5 W** UltraScale+ part — bigger than a Z7020, far smaller than a
  ZU9EG. Candidates: ZU3EG (Ultra96-V2, 7.6 Mbit BRAM, ~$250), ZU5EV / Kria K26
  (~23 Mbit incl. URAM), Kria K24. Portability rule of thumb, using
  **theoretical on-chip footprint × 2** for FINN's real BRAM allocation
  (multiplier measured from Electronics 2025 14:3993):
  - **< ~7 Mbit** → ports to anything in the class (pico W4A8 = 3.5 Mbit)
  - **7–23 Mbit** → needs a K26-class part (yolov5n@416 W4A8 = 21 Mbit)
  - **> 23 Mbit** → ZU7EV/ZU9EG only, i.e. it does not ship
  Record the footprint of anything we train so we know what ports.
- **Post-processing runs on the A53s for now — decided 2026-08-20.** The tracking
  and aim-output flow (README §11: seed, IoU cluster, WBF, Kalman, centring gate)
  is implemented in `deploy/track.py` and runs on Linux, next to the capture and
  the driver. Exercised on synthetic tracks only — **never on a real detection**,
  and every threshold in it is a placeholder awaiting footage through the real
  lens (open question 1). Rationale: the
  task is a **baseline** — prove the chain end to end — and the arithmetic is
  ~10k operations per frame, microseconds on anything. R5F was considered and
  dropped: it buys determinism but costs an OpenAMP/RPMsg hop and a second
  firmware to maintain, which is not what a baseline needs.
  **This is temporary and known to be wrong for deployment.** The endpoint is the
  whole chain in PL — preprocessing, decode, tracker — fed by MIPI straight into
  the fabric, so nothing between exposure and aim command is scheduled by Linux.
  Every millisecond of scheduler jitter lands directly in aim error, exactly like
  the missing camera trigger. Do not let baseline code assume a Linux-shaped
  world any more than it must.
- **Power shape:** on the measured Z7020 reference, **1.9 W of 2.55 W total was
  PS + DDR idle**; the fabric drew only 0.22–0.65 W. PS involvement, not fabric
  size, dominates the power budget. So the full-fabric architecture that
  minimises latency and jitter (preprocessing, decode, NMS, tracker in PL; A53s
  parked; control on R5F) is *also* the low-power one. These goals converge —
  don't trade one against the other.

## Environment
- **uv-managed** venv at `.venv/`, Python 3.11, locked in `uv.lock`. Run
  everything via `uv run ...`.
- GPU: **RTX 4090, 24 GB** (Ada, capability 8.9), 12 CPU cores. torch
  **2.4.1+cu121**, CUDA available and verified. Note the torch build has no
  `sm_89` cubin (arch list stops at `sm_86`, plus `sm_90`); `sm_86` is
  binary-compatible with Ada, so this is harmless — don't "fix" it.
  (Earlier work was done on a GTX 1070, 8 GB — old batch sizes reflect that.)
- Key libs: brevitas 0.13.0, qonnx 1.0.0, onnx 1.22, onnxruntime 1.27,
  opencv 4.11, albumentations 2.0.8, kaggle 2.2.3, numpy 1.26 (pinned <2).
- **FPGA toolchain:** Vivado/Vitis/Vitis HLS **2022.2** at `/home/alex/Xilinx`;
  FINN cloned twice at `~/Repos/finn` (v0.10.1) and `~/Repos/finn-dev` (dev, use
  this one for joined nets). Both have setup gotchas that will waste hours if
  rediscovered — see [build_notes.md](.claude/docs/build_notes.md) §7.
- **PetaLinux 2022.2** is NOT installed on the host and must not be: Ubuntu 26.04
  has gcc 15 and python 3.12, and PetaLinux is Yocto kirkstone. It lives at
  `/home/alex/petalinux/` (11 GB, aarch64 only) and runs in the container built
  from `deploy/petalinux/Dockerfile`; `plnx.sh` is the entry point. Note that
  **`petalinux-*` commands exit 0 when they fail** — the real error is in
  `build/config.log`. See build_notes §11.
- Repo layout: `scripts/` (merge + cleaning + `center_error.py`), `configs/`,
  `training/`, `qat/` (`quantize.py` `ptq_baseline.py` `train_qat.py`
  `evaluate.py`), `export/` (`export_qonnx.py` `verify_qonnx.py`
  `check_join_scales.py` `finn_transforms.py` `finn_build.py`
  `balance_folding.py` `verify_qonnx_v8.py` `verify_finn_steps.py`),
  `deploy/` (`postprocess.py` `run_on_board.py` + `petalinux/` — the board's
  Linux image). No vendored model code — `ultralytics` comes from `.venv/`.
  `data/` + `runs/` are gitignored.

## Gotchas / notes
- **conda shell warning:** the user's shell has `VIRTUAL_ENV=~/anaconda3`
  active; uv ignores it and correctly uses `.venv`. Harmless. Prefer `uv run`.
- **onnxruntime `/sys/class/drm/card0` warning:** cosmetic GPU-discovery noise;
  ORT falls back to CPU fine.
- **Dataset regimes (brief §5 is WRONG):** measured bbox areas show source A
  (muki2003, `data/raw/drone_dataset`, 1339 imgs) is already **close-range**
  (median box 33% of frame) — our target regime — while source B (sshikamaru,
  `Database1`, 4007 imgs) is **long-range** video frames (median 0.55%). So we
  DO have real close-range data. Track per-regime mAP separately (see
  `manifest.csv` `regime` column). Merged set: 5345 pairs but only ~2891 unique
  pHash-clusters (46% near-dup video frames); split is cluster-safe (no near-dup
  train/val leakage). See memory `dataset-regimes`.
- **Quantization/FINN gotchas** (shared SiLU instance, tied join scales, the
  ReLU-before-Quant rule, the `set_folding` patch that must survive `git pull`)
  are all in [build_notes.md](.claude/docs/build_notes.md).

## Conventions
- Commit/push only when the user asks.
- Change only what's necessary; don't add tests/examples unless asked.
- Kaggle dataset pull needs `~/.kaggle/kaggle.json` (manual credential step).
- Research output goes to `.claude/docs/research/<question>.md`, one file per
  question, every claim cited to a primary source. Not into build_notes — see
  above for why.
