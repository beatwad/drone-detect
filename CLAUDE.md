# Drone-Detect — Project Memory

Close-range (≤10 m) drone detection for precision aiming, targeting an
**FPGA-accelerated** deployment. A vision system detects a large-in-frame drone
and outputs angular/positional data to a downstream kinetic aiming subsystem.
Latency budget end-to-end ~50–100 ms. Status: **Proof of Concept**.

Full source docs live in [.claude/docs/](.claude/docs/):
- [project_brief.md](.claude/docs/project_brief.md) — hardware, toolchain, datasets, roadmap, risks.
- [host_pipeline_plan.md](.claude/docs/host_pipeline_plan.md) — the phased host-side plan we're executing.

## Pipeline (target)
`YOLOv5n (float)` → `Brevitas QAT (INT8, SiLU→ReLU)` → `QONNX export` →
`FINN → bitstream` → `PYNQ deployment`. Everything up to and including QONNX
export is **board-agnostic** and done on the host GPU; FINN synthesis and
everything downstream (camera/UVC, on-ARM NMS, latency validation) is deferred
until a devkit is chosen.

### Host-side phases (what we're building)
0. Scaffolding — **DONE**
1. Dataset acquisition + merge — **DONE** (see `scripts/merge_dataset.py`,
   `configs/drone.yaml`, `data/drone/` + `manifest.csv`)
2. Augmentation (close-range framing from long-range data)
3. Float training YOLOv5n — gate: **mAP50 > 0.85** on val
4. Brevitas QAT (INT8, SiLU→ReLU) — gate: quant mAP within budget of float
5. QONNX export (board-agnostic handoff to FINN)

## Locked decisions
- **YOLOv5 source:** classic `ultralytics/yolov5` **v7.0**, vendored at
  [yolov5/](yolov5/) (pin: commit `915bbf2`, see `yolov5/VENDORED_PIN.txt`).
  NOT master (which pulls in the `ultralytics` pkg + anchor-free code) and NOT
  the pip package. Rationale: matches the brief's anchor-based/C3 assumptions,
  keeps QAT graph surgery tractable, has known Brevitas/FINN precedent.
- **SiLU→ReLU swap timing:** at the **QAT stage**, not before float training.
  Float model trains with stock SiLU for the strongest baseline; ReLU
  substitution + fine-tune happens during Brevitas QAT.
- **Devkit:** undecided. Do NOT do devkit-specific work (PYNQ image, FINN
  part-targeting, camera, actuator) until the board is chosen.

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
- Repo layout: `scripts/ configs/ training/ qat/ export/` (skeleton),
  `yolov5/` (vendored), `data/` + `runs/` are gitignored.

## Gotchas / notes
- **Shared SiLU instance:** YOLOv5 uses one class-level `Conv.default_act =
  nn.SiLU()` shared across ALL Conv layers, so `model.modules()` reports a
  single SiLU. The QAT SiLU→ReLU swap is done by rebinding `Conv.default_act`
  (or walking `.act` attributes) before instantiation — not per-layer object
  replacement.
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

## Conventions
- Commit/push only when the user asks.
- Change only what's necessary; don't add tests/examples unless asked.
- Kaggle dataset pull needs `~/.kaggle/kaggle.json` (manual credential step).
