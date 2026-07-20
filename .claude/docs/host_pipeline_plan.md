# Host-Side Pipeline Plan (Devkit-Independent)

**Scope:** Everything that can be built and validated on a host GPU workstation
**without** committing to a specific FPGA devkit — from an empty repo up to a
board-agnostic **QONNX** export ready to hand to FINN.

**Out of scope (deferred until a board is chosen):** PYNQ image, FINN synthesis
to a specific part, camera/UVC bring-up, on-ARM NMS, end-to-end latency
validation. See brief §7 steps 1, 5, 6, 7 and the second half of step 4.

**Key upfront decision (locked):** SiLU→ReLU swap happens **at the QAT stage**,
not before float training. Float model trains with stock SiLU; ReLU substitution
is a QAT-time graph edit + fine-tune. Rationale: get a strong float baseline
first, matching brief §7 ordering.

---

## The devkit boundary

```
  HOST-SIDE (this plan)                        │  DEVKIT-SIDE (deferred)
  ─────────────────────────────────────────── │ ────────────────────────
  0. Scaffolding                               │  FINN synthesis → .bit
  1. Dataset acquisition + merge               │  PYNQ deployment
  2. Augmentation (close-range framing)        │  Camera / UVC capture
  3. Float training (YOLOv5n, SiLU)            │  On-ARM NMS
  4. Brevitas QAT (SiLU→ReLU, INT8)            │  Latency validation
  5. QONNX export (board-agnostic)  ───────────┼──▶ handoff artifact
                                               │
```

The QONNX file is the clean handoff. It carries no board/part assumptions —
FINN decides folding/parallelism per target at synthesis time. So all work
below is safe to do now, whatever board wins.

---

## Phase 0 — Scaffolding

**Goal:** reproducible Python env + repo layout + git hygiene.

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

## Phase 1 — Dataset acquisition + merge

**Goal:** one clean YOLO-format dataset from the two Kaggle sources.

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

**Risk (brief §5):** both datasets are long/medium range (drone small in
frame). The whole point is close range. This is *not* fixed here — it's the
motivation for Phase 2, and ultimately for the "real close-range footage" step
that's out of scope for now.

---

## Phase 2 — Augmentation (close-range framing)

**Goal:** simulate close-range (large-in-frame) drones from long-range data.

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

## Phase 3 — Float training (YOLOv5n, stock SiLU)

**Goal:** a float checkpoint clearing the sanity bar.

- Train `yolov5n` from COCO-pretrained weights (transfer) on the merged set.
- Keep **stock SiLU** here (per locked decision). This is the strongest-baseline
  float model; ReLU swap comes later.
- Target: **mAP50 > 0.85** on val (brief §7 step 2) as a go/no-go gate.
- Log to `runs/`; keep `best.pt`.
- Evaluate on the close-range mini-eval set separately — expect lower numbers;
  this is the honest metric for the actual use case.

**Deliverable:** `best.pt` float model + metrics report (standard val +
close-range mini-eval).

**Gate:** if mAP50 < 0.85 on standard val, iterate on data/augmentation before
proceeding to QAT — no point quantizing a weak model.

---

## Phase 4 — Brevitas QAT (SiLU→ReLU, INT8)

**Goal:** INT8 quantization-aware model, FINN-compatible activations.

- **SiLU→ReLU graph surgery** happens *here* (the locked decision). Replace all
  SiLU with ReLU in the model definition, then load the float `best.pt` weights
  into the modified architecture.
  - Expect an accuracy drop from the activation swap — this is exactly why we
    QAT-fine-tune afterward rather than just doing it and stopping.
- Convert conv/act layers to Brevitas quantized equivalents (`QuantConv2d`,
  `QuantReLU`, quant identity on inputs), INT8 weights + activations (brief §4).
- **Fine-tune from float weights** ~20–30 epochs (brief §7 step 3) to recover
  accuracy lost to (a) ReLU swap and (b) quantization.
- Validate quantized mAP; compare against float baseline. Define an acceptable
  degradation budget (e.g. keep mAP50 within a few points of float).
- **Known risk (brief §8):** C3 blocks and YOLOv5-specific constructs may need
  simplification for clean FINN compilation. Watch for ops that Brevitas
  quantizes fine but FINN later rejects. We can't fully validate FINN support
  without running FINN, but we can keep the architecture conservative (avoid
  exotic ops) to de-risk.

**Deliverable:** quantized YOLOv5n checkpoint + QAT metrics.

---

## Phase 5 — QONNX export (board-agnostic handoff)

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

**Deliverable:** `export/yolov5n_drone.qonnx` + a short "output tensor spec" note
+ export-vs-runtime numerical parity check.

---

## Sequencing & gates summary

```
Phase 0  Scaffold ──▶ 1 Dataset ──▶ 2 Augment ──▶ 3 Float train
                                                      │ gate: mAP50 > 0.85
                                                      ▼
                                          4 QAT (SiLU→ReLU, INT8)
                                                      │ gate: quant mAP within budget
                                                      ▼
                                          5 QONNX export ──▶ [FINN handoff]
```

## Open items needing a user decision along the way

1. **YOLOv5 source:** classic `ultralytics/yolov5` repo (recommended) vs pip
   `ultralytics`. Blocks Phase 0.
2. **Kaggle credentials:** need `kaggle.json` API token to auto-pull datasets
   (Phase 1) — or datasets provided manually.
3. **Augmentation depth:** YOLOv5-native hyp only vs. + Albumentations for
   motion blur / noise (Phase 2).
4. **Version pinning:** confirm the Brevitas/QONNX versions against a target
   FINN release even though FINN isn't run yet, so the exported QONNX won't be
   rejected later (Phase 0/5).
```
