# Drone-Detect — Project Memory

Close-range (≤10 m) drone detection for precision aiming, targeting an
**FPGA-accelerated** deployment. A vision system detects a large-in-frame drone
and outputs angular/positional data to a downstream kinetic aiming subsystem.
Latency budget end-to-end ~50–100 ms. Status: **Proof of Concept**.

Full source docs live in [.claude/docs/](.claude/docs/):
- [project_brief.md](.claude/docs/project_brief.md) — hardware, toolchain, datasets, roadmap, risks.
- [host_pipeline_plan.md](.claude/docs/host_pipeline_plan.md) — the phased host-side plan we're executing.
- [build_notes.md](.claude/docs/build_notes.md) — **everything measured about QAT →
  QONNX → FINN → bitstream.** Read this before touching `qat/`, `export/`, or
  running a FINN build; it is where all the hard-won toolchain detail lives.

## Pipeline (target)
`YOLOv5n (float)` → `Brevitas QAT (INT8, SiLU→ReLU)` → `QONNX export` →
`FINN → bitstream` → `PYNQ deployment`. Everything up to and including QONNX
export is **board-agnostic** and done on the host GPU.

**Where we are:** the whole chain compiles. n_eighth (20 joins) goes end-to-end
through FINN dev to estimates — 245 layers, all converted, contiguous dataflow
block. **Bitstream is the open work**, and it is the first step that checks the
estimates against reality. Key facts, all detailed in
[build_notes.md](.claude/docs/build_notes.md):

- **FINN compiles branched YOLO** — the old "no joins" blocker was wrong
  (refuted 2026-08-04). Joins need their branches to share one quantisation
  scale, which is **set during QAT** and is already done (20/20 tied, free).
- **Use FINN dev, not v0.10.1**, for anything with joins.
- **Folding is the whole game.** `--target-fps 30` under-folds ~11× and produces
  unbuildable FIFOs; `export/balance_folding.py` is the fix. Preferred n_eighth
  config: **189 FPS @100 MHz, 129k LUT est** (`mvau_wwidth_max=144`).
- **Never use the default FIFO sizing strategy** on a joined net — both
  automatic strategies are dead ends; use `--fifo-strategy none`.
- Budget LUT at **×1.56 the estimate**. 75% estimated is not a safe ceiling.

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
3. Float training YOLOv5n — gate mAP50 > 0.85 — **DONE**, mAP50 **0.963**
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
5c. FINN build (bitstream) — **OPEN.** Three failed attempts so far, all in FIFO
   sizing or stitching; the path forward is balanced folding + `--fifo-strategy
   none`. See build_notes §5.
6. `configs/yolov5_pico.yaml` — branch-free, single-scale, 357k-param net. Float
   close mAP50 **0.975** vs yolov5n's 0.989; long range 0.778 vs 0.904. Not yet
   quantized. Sized for a Z7020 budget, so ~10% of a ZCU102. **Its rationale is
   now largely void**: it was insurance against the branch blocker, and branches
   compile. `configs/yolov5n_eighth.yaml` (446k params, full FPN, multi-scale)
   beats it on every regime and was explicitly conditional on this question.

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
     Smaller input is better here, and it suits pico's **stride-32** single head,
     which needs targets well above 32 px.
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
2. **What does pico cost once quantized?** Float close mAP50 is 0.975; W4A4 and
   W4A8 are unmeasured. ~1 h — the existing `qat/` pipeline runs on pico
   unchanged. Closes the last accuracy unknown on the FINN path. Expect
   ~0.96–0.97 by analogy with yolov5n (INT8 free; W4A8 cost close range 0.5 pt).

## Locked decisions
- **YOLOv5 source:** classic `ultralytics/yolov5` **v7.0**, vendored at
  [yolov5/](yolov5/) (pin: commit `915bbf2`, see `yolov5/VENDORED_PIN.txt`).
  NOT master (which pulls in the `ultralytics` pkg + anchor-free code) and NOT
  the pip package. Rationale: matches the brief's anchor-based/C3 assumptions,
  keeps QAT graph surgery tractable, has known Brevitas/FINN precedent.
- **SiLU→ReLU swap timing: REVISED — train float in ReLU from the start.**
  (Was: "swap at the QAT stage, not before float training." Measurement
  contradicted it.) A retrained ReLU model costs 2.1 pt mAP50-95 and **zero**
  mAP50 / centre error vs SiLU. But *pasting* ReLU into SiLU-trained weights
  destroys the model — mAP50 0.963 → **0.078**, close range → 0.0007 — because
  ~49% of pre-activations are negative and ReLU zeroes them across 57 layers.
  So QAT must start from `runs/train/relu_more_data_5/weights/best.pt`, never
  from the SiLU model. Use `configs/yolov5n_relu.yaml` (`activation: nn.ReLU()`).
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
- Repo layout: `scripts/` (merge + cleaning + `center_error.py`), `configs/`,
  `training/`, `qat/` (`quantize.py` `ptq_baseline.py` `train_qat.py`
  `evaluate.py`), `export/` (`export_qonnx.py` `verify_qonnx.py`
  `check_join_scales.py` `finn_transforms.py` `finn_build.py`
  `balance_folding.py`), `yolov5/` (vendored).
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
