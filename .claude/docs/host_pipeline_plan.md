# Host-Side Pipeline Plan (Devkit-Independent)

> **STATUS 2026-08-06: all five phases are DONE.** This document is kept as the
> record of *what we set out to do and why*; where measurement contradicted the
> plan, the correction is marked inline. For what actually happened — the
> measured QAT/QONNX/FINN detail — see [build_notes.md](build_notes.md).
> The devkit is no longer undetermined: **ZCU102, chosen 2026-08-03**, so the
> "devkit-independent" framing below is historical. It served its purpose: none
> of phases 0–5 had to be redone once the board was picked.

**Scope:** Everything that can be built and validated on a host GPU workstation
**without** committing to a specific FPGA devkit — from an empty repo up to a
board-agnostic **QONNX** export ready to hand to FINN.

**Out of scope (deferred until a board is chosen):** PYNQ image, FINN synthesis
to a specific part, camera/UVC bring-up, on-ARM NMS, end-to-end latency
validation. See brief §7 steps 1, 5, 6, 7 and the second half of step 4.
*(FINN synthesis is now in progress — see build_notes §4–5.)*

**Key upfront decision — REVERSED by measurement.** The plan was: SiLU→ReLU swap
happens **at the QAT stage**, float trains with stock SiLU. That is wrong.
*Pasting* ReLU into SiLU-trained weights destroys the model (mAP50 0.963 →
**0.078**) — ~49% of pre-activations are negative and ReLU zeroes them across 57
layers. Retraining float **in ReLU from the start** costs 2.1 pt mAP50-95 and
**zero** mAP50 / centre error. So: train float in ReLU
(`configs/yolov5n_relu.yaml`), and start QAT from
`runs/train/relu_more_data_5/weights/best.pt`. See CLAUDE.md → Locked decisions.

---

## The devkit boundary

```
  HOST-SIDE (this plan)                        │  DEVKIT-SIDE (deferred)
  ─────────────────────────────────────────── │ ────────────────────────
  0. Scaffolding                               │  FINN synthesis → .bit
  1. Dataset acquisition + merge               │  PYNQ deployment
  2. Augmentation (close-range framing)        │  Camera / UVC capture
  3. Float training (YOLOv5n, ReLU*)           │  On-ARM NMS
  4. Brevitas QAT (INT8, tied join scales*)    │  Latency validation
  5. QONNX export (board-agnostic)  ───────────┼──▶ handoff artifact
                                               │
```

`*` = differs from the original plan. Float trains in ReLU (see above), and QAT
additionally has to **tie the quantiser at every join** — a FINN precondition
discovered later that is unfixable at export time. See build_notes §1.

The QONNX file is the clean handoff. It carries no board/part assumptions —
FINN decides folding/parallelism per target at synthesis time. So all work
below is safe to do now, whatever board wins.

---

## Phase 0 — Scaffolding — **DONE**

**Goal:** reproducible Python env + repo layout + git hygiene.

*Outcome: classic `ultralytics/yolov5` v7.0 vendored at commit `915bbf2`; env is
uv-managed (`.venv/`, `uv.lock`), not pip/conda as sketched below.*

- **YOLOv5 source decision (blocking everything downstream).** Two realistic
  options:
  - `ultralytics/yolov5` (the classic v5 repo) — closest to the brief's
    assumptions (anchor-based, C3 blocks, native `yolov5n.yaml`), well-trodden
    Brevitas/FINN examples exist for it.
  - `ultralytics` pip package (v8+ API) — can still train a v5n config but the
    codebase and export path diverge from most FINN references.
  - **Recommendation:** classic `ultralytics/yolov5` repo, pinned to a specific
    commit/tag. It matches the brief and keeps the QAT graph surgery tractable.
- Env: conda or venv, pinned `torch` + CUDA matching the host GPU, plus
  `brevitas`, `qonnx`, `onnx`. Pin versions — Brevitas/QONNX/FINN are
  version-sensitive as a trio; check the FINN release's compatibility matrix
  even though we don't run FINN yet, so we don't export an incompatible QONNX.
- Repo layout (proposed):
  ```
  data/            # datasets (gitignored, populated by scripts)
  scripts/         # dataset prep, merge, split
  configs/         # yolov5n model yaml, dataset yaml, hyp yaml
  training/        # train/val entrypoints, logs
  qat/             # brevitas QAT + SiLU→ReLU surgery
  export/          # QONNX export
  runs/            # checkpoints, metrics (gitignored)
  ```
- `.gitignore` for `data/`, `runs/`, weights, `__pycache__`, env dirs.
- `git init` + initial commit (repo is currently not a git repo... actually
  `.git/` exists — confirm state before first commit).

**Deliverable:** `pip install -r requirements.txt` reproduces the env; empty
but structured repo committed.

---

## Phase 1 — Dataset acquisition + merge — **DONE**

**Goal:** one clean YOLO-format dataset from the two Kaggle sources.

*Outcome: `scripts/merge_dataset.py`, `configs/drone.yaml`, `data/drone/` +
`manifest.csv`. Dedup needed more than the image hashing planned below — flat sky
makes pHash degenerate, so it takes pixel NCC + gradient NCC on top (memory
`phash-alone-fails-on-sky`). Split is cluster-safe: 5345 pairs, ~2891 unique
pHash-clusters.*

- Sources (brief §5):
  - `muki2003/yolo-drone-detection-dataset` (~1359 imgs, YOLO format)
  - `sshikamaru/drone-yolo-detection` (secondary, YOLO format)
- Pull via `kaggle` CLI (needs `~/.kaggle/kaggle.json` API token — **flag to
  user**, this is a manual credential step).
- **Class-map reconciliation (critical, must verify by inspection):** confirm
  both sets are single-class "drone". If class indices differ (e.g. one uses
  `0`, the other embeds drone among multiple classes), remap to a unified
  single-class `0: drone`. Do not assume — read the `.txt` label files and any
  `data.yaml`/`classes.txt`.
- Normalize directory structure to a single `images/` + `labels/` layout.
- **Dedup:** the two sets may overlap or share source imagery — hash images and
  drop duplicates to avoid train/val leakage.
- **Split:** deterministic train/val (e.g. 80/20), seeded. Ensure the split is
  by image, and that no near-duplicate lands on both sides.
- Emit a `configs/drone.yaml` (dataset yaml: paths, `nc: 1`, `names: [drone]`).
- Sanity: a small script to render a handful of images with boxes overlaid, to
  confirm annotations are correct after remap.

**Deliverable:** `data/drone/{train,val}/{images,labels}` + `drone.yaml`;
verification renders.

**Risk (brief §5) — WRONG, corrected by measurement.** The brief claims both
datasets are long/medium range. Measured bbox areas say otherwise: source A
(muki2003, 1339 imgs) is already **close-range** (median box 33% of frame) — our
target regime — while only source B (sshikamaru `Database1`, 4007 imgs) is
long-range video frames (median 0.55%). **We do have real close-range data.**
Track per-regime mAP separately (`manifest.csv` `regime` column); memory
`dataset-regimes`. This weakens, but does not remove, the motivation for Phase 2.

---

## Phase 2 — Augmentation (close-range framing) — **DONE**

**Goal:** simulate close-range (large-in-frame) drones from long-range data.

*Outcome: in-loop via hyp, no custom pipeline. The one non-obvious result is
`mosaic: 0.5` rather than 1.0 — it cuts the empty-frame false-alarm rate ~12 pt
and val mAP cannot see the difference (memory `mosaic-reduces-false-alarms`).*

- Core trick from brief §5: `RandomCrop(scale=(0.3, 1.0))` — cropping into the
  image enlarges the drone relative to the frame, approximating close range.
- YOLOv5 has built-in augmentation via its `hyp.*.yaml` (mosaic, scale,
  translate, hsv, flip). Map the brief's intent onto these:
  - `scale` hyperparameter (image scale gain) + high `mosaic` gives
    close-range-like framing without a custom pipeline.
  - `hsv_h/s/v` covers the ColorJitter intent.
  - Motion blur / Gaussian noise are **not** native to YOLOv5 hyp — if we want
    them (brief lists `MotionBlur(p=0.3)`, `GaussianNoise(p=0.2)`), add via an
    Albumentations hook (YOLOv5 supports an Albumentations transform block).
- **Decision to make:** rely on YOLOv5 native hyp only (simplest) vs. add
  Albumentations for motion blur + noise (closer to brief, more code). Suggest
  starting native-only, add Albumentations if val performance on hand-cropped
  close-range test images is weak.
- Build a tiny held-out "close-range-ish" eval set (manually cropped from val)
  to measure whether augmentation actually helps the target regime — the
  standard mAP on long-range val will *not* tell us this.

**Deliverable:** tuned `hyp.yaml`; optional Albumentations block; close-range
mini-eval set.

---

## Phase 3 — Float training (YOLOv5n) — **DONE, mAP50 0.963**

**Goal:** a float checkpoint clearing the sanity bar.

- Train `yolov5n` from COCO-pretrained weights (transfer) on the merged set.
- ~~Keep **stock SiLU** here (per locked decision).~~ **Superseded:** train
  **two** float models, SiLU and ReLU, and carry the **ReLU** one forward.
  Measured: `runs/train/more_data_5` (SiLU) mAP50 **0.963**,
  `runs/train/relu_more_data_5` (ReLU) mAP50 **0.963** — identical on the gate
  metric, 2.1 pt apart on mAP50-95. The ReLU model is what QAT consumes.
- Target: **mAP50 > 0.85** on val (brief §7 step 2) as a go/no-go gate. **Met.**
- Log to `runs/`; keep `best.pt`.
- Evaluate on the close-range mini-eval set separately — expect lower numbers;
  this is the honest metric for the actual use case.

**Deliverable:** `best.pt` float model + metrics report (standard val +
close-range mini-eval).

**Gate:** if mAP50 < 0.85 on standard val, iterate on data/augmentation before
proceeding to QAT — no point quantizing a weak model.

---

## Phase 4 — Brevitas QAT (INT8) — **DONE**

**Goal:** INT8 quantization-aware model, FINN-compatible activations.

- ~~**SiLU→ReLU graph surgery** happens *here*.~~ **Superseded** — it happens
  before float training (see the reversal at the top). Note the mechanism when
  swapping: YOLOv5 shares **one class-level `Conv.default_act` instance** across
  all Convs, so rebind that, not per-layer `.act` objects.
- Convert conv/act layers to Brevitas quantized equivalents (`QuantConv2d`,
  `QuantReLU`, quant identity on inputs), INT8 weights + activations (brief §4).
- ~~**Fine-tune from float weights** ~20–30 epochs to recover accuracy.~~
  **Not needed at INT8.** W8A8 PTQ scores mAP50 **0.9628** vs float 0.9629 — the
  calibration pass alone suffices, and the fine-tune budget was never spent.
  4-bit weights *do* collapse under PTQ (0.197); QAT recovers close range to
  0.9856, so the fine-tune only earns its keep if we drop to W4.
- **Unplanned, and the load-bearing part: tie the quantiser at every join.**
  FINN can only streamline a join whose branches share one scale — `a·x + b·y`
  does not factor unless `a == b`. This is a QAT-time decision that cannot be
  repaired at export. `SharedQuant` in `qat/quantize.py`, all 20 joins, gate
  `export/check_join_scales.py` → 20/20, and **free** in accuracy. Also: every
  shared quantiser needs a plain `nn.ReLU()` in front of it. build_notes §1, §A.
- **Known risk (brief §8):** C3 blocks and YOLOv5-specific constructs may need
  simplification for clean FINN compilation. **Did not materialise** — C3, SPPF
  and the full FPN all compile; see build_notes §3–4. The architecture needed no
  simplification, only the join-scale tying above.

**Deliverable:** quantized YOLOv5n checkpoint + QAT metrics. **Delivered.**

---

## Phase 5 — QONNX export (board-agnostic handoff) — **DONE**

**Goal:** the clean artifact to hand FINN, with no board assumptions.

- Export via `brevitas.export.export_qonnx` (brief §4).
- Validate the QONNX graph: load with `qonnx`, run shape inference, sanity-check
  op set is within what FINN's frontend expects (Quant nodes, standard conv,
  ReLU, no leftover SiLU, no unsupported ops).
- Numerical check: run the QONNX model (via `onnxruntime`/`qonnx` executor) on a
  few val images and confirm outputs match the Brevitas model within tolerance —
  catches export bugs before they become confusing FINN failures.
- Document the tensor layout / output decoding (anchors, box format) — this is
  the spec the future on-ARM NMS (deferred) will implement.

**Deliverable:** **`export/n_eighth_tied_w8a8_416.onnx`** + export-vs-runtime
numerical parity check. Both gates pass: `check_join_scales.py` 20/20 tied,
`verify_qonnx.py` faithful (32/32 boxes, 0.367 px max centre error).

**Caveat on "no board assumptions" — one leaked in.** The graph input `Quant` is
unsigned with no predecessor, which FINN v0.10.1 rejects, so
`finn_transforms.InputQuantToUintDtype` rewrites it to `x_uint8 -> Mul(scale)`.
That **changes the accelerator's input contract**: it consumes integer codes
`round(x/scale)`, not floats. Good for deployment (no float preprocessing on the
ARM side), but host preprocessing must match. build_notes §A3.

---

## Sequencing & gates summary

```
Phase 0  Scaffold ──▶ 1 Dataset ──▶ 2 Augment ──▶ 3 Float train (ReLU)
   DONE                 DONE           DONE            DONE, mAP50 0.963
                                                      │ gate: mAP50 > 0.85  ✓
                                                      ▼
                                          4 QAT (INT8, tied join scales)
                                                      │ DONE, W8A8 0.9628
                                                      │ gate: within budget  ✓
                                                      ▼
                                          5 QONNX export ──▶ [FINN handoff]
                                             DONE, both gates pass
```

Everything downstream of the handoff — FINN compilation, folding, bitstream —
is tracked in [build_notes.md](build_notes.md), not here.

## Open items needing a user decision along the way — **ALL SETTLED**

1. ~~**YOLOv5 source:**~~ **Settled:** classic `ultralytics/yolov5` **v7.0**,
   vendored at commit `915bbf2`. Not master, not the pip package.
2. ~~**Kaggle credentials:**~~ **Settled:** `~/.kaggle/kaggle.json` supplied;
   remains a manual step for anyone reproducing the pull.
3. ~~**Augmentation depth:**~~ **Settled:** YOLOv5-native hyp only. Albumentations
   is in the env but the in-loop hyp path was sufficient; `mosaic: 0.5` mattered
   more than adding transforms.
4. ~~**Version pinning:**~~ **Settled and validated by actually running FINN:**
   brevitas 0.13.0 / qonnx 1.0.0 / onnx 1.22, against FINN **dev** (not v0.10.1)
   with Vivado **2022.2**. The exported QONNX was not rejected. Note the real
   version risk turned out to be FINN-side, not Brevitas-side — v0.10.1 cannot
   compile our joined graph at all (build_notes §B).

## What this plan got wrong (worth remembering)

- **The SiLU→ReLU ordering**, reversed above — the single costliest planning
  error, caught only because the pasted-ReLU model scored 0.078.
- **"Fine-tune 20–30 epochs to recover accuracy"** — at INT8 there was nothing to
  recover. The plan budgeted training time for a problem that did not exist.
- **The dataset regime assumption** inherited from brief §5 — we already had
  close-range data and did not know it.
- **What the plan never anticipated at all:** that the hard part would be neither
  accuracy nor quantization, but **join scales and folding**. Phases 0–5 landed
  close to schedule; everything expensive has been downstream of the handoff.
