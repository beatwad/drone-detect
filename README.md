# drone-detect

Close-range (≤10 m) drone detection for precision aiming, targeting an
**FPGA-accelerated** deployment. A vision system detects a large-in-frame drone
and outputs angular/positional data to a downstream kinetic aiming subsystem.
Latency budget end-to-end ~50–100 ms.

**Status: Proof of Concept.** The host-side float model is trained and passes its
gate; quantization and FPGA synthesis are not started.

Target pipeline:

```
YOLOv5n (float)  →  Brevitas QAT (INT8, SiLU→ReLU)  →  QONNX export  →  FINN  →  bitstream  →  PYNQ
└──────────────────────── host GPU, board-agnostic ────────────────────────┘   └──── deferred ────┘
```

| Phase | What | State |
|---|---|---|
| 0 | Scaffolding | done |
| 1 | Dataset acquisition + merge | done |
| 2 | Augmentation (close-range framing) | done (in-loop, via hyp) |
| 3 | Float training YOLOv5n — gate mAP50 > 0.85 | **done — mAP50 0.963** |
| 4 | Brevitas QAT (INT8, SiLU→ReLU) | not started |
| 5 | QONNX export | not started |

Everything through QONNX export is board-agnostic and runs on the host GPU. FINN
synthesis and everything downstream (camera/UVC on target, on-ARM NMS, latency
validation) waits until a devkit is chosen.

---

## 1. Install

Requires **Python 3.11** and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU with
CUDA 12.1-capable drivers is expected for training; inference will fall back to CPU.

```bash
git clone <this repo> drone-detect
cd drone-detect
uv sync            # creates .venv/ from uv.lock, pinned torch 2.4.1+cu121
```

Run everything through `uv run` so the vendored YOLOv5 and the cu121 torch build
are picked up:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.4.1+cu121 True
```

YOLOv5 v7.0 is **vendored** at [yolov5/](yolov5/) (pin + local patches in
[yolov5/VENDORED_PIN.txt](yolov5/VENDORED_PIN.txt)) — do not `pip install yolov5`
or swap in master, which pulls the anchor-free rewrite and breaks the QAT plan.

Kaggle downloads need credentials at `~/.kaggle/kaggle.json` (manual step).

### Notes on the environment

- The shell may have `VIRTUAL_ENV=~/anaconda3` set; uv ignores it and correctly
  uses `.venv/`. Harmless.
- The torch build ships no `sm_89` cubin (arch list stops at `sm_86`, plus `sm_90`).
  `sm_86` is binary-compatible with Ada (RTX 4090), so this is fine — don't "fix" it.
- onnxruntime may warn about `/sys/class/drm/card0`; cosmetic, it falls back to CPU.

---

## 2. Build the training dataset from raw

`data/` and `runs/` are **gitignored** — the dataset is rebuilt locally, not cloned.

### 2.1 Fetch the raw sources

Nine sources are merged, referenced by letter throughout the code
(`SOURCES` in [scripts/merge_dataset.py:77](scripts/merge_dataset.py#L77)). Each
must land in `data/raw/<exact dir name>`:

| Src | `data/raw/` directory | Origin | Notes |
|---|---|---|---|
| A | `drone_dataset` | Kaggle `muki2003/yolo-drone-detection-dataset` | close-range, single-class |
| B | `Database1` | Kaggle `sshikamaru/drone-yolo-detection` | long-range video frames |
| C | `Drone.v1i.yolov5pytorch` | Roboflow Universe | mixed range |
| D | `UAVs.v2i.yolov5pytorch` | Roboflow Universe | mixed range |
| E | `Drone detection.v8i.yolov5pytorch` | Roboflow Universe | + bird/plane/sky hard negatives |
| F | `Drone detection.v3i.yolov5pytorch` | Roboflow Universe | multi-class; only `drone` kept |
| G | `Drone Detection.v1i.yolov5pytorch` | Roboflow Universe | single-class |
| H | `Drone Detection.v6i.yolov5pytorch` | Roboflow Universe | single-class |
| I | `Drone Detection.v5i.yolov5pytorch` | Roboflow Universe | multi-class UAV+drone → both to 0 |

Kaggle sources:

```bash
uv run kaggle datasets download -d muki2003/yolo-drone-detection-dataset -p data/raw --unzip
uv run kaggle datasets download -d sshikamaru/drone-yolo-detection       -p data/raw --unzip
```

Roboflow sources are downloaded manually from Roboflow Universe in **YOLOv5
PyTorch** format and unzipped into `data/raw/` under the directory names above.
The names are matched literally — a renamed directory is silently skipped.

### 2.2 Pre-clean source I

Source I is the largest and needs two passes before merging. Both are **dry-run by
default**; pass `--apply` to actually delete.

```bash
# 1. drop images that already exist in an earlier-built merged set (~700)
uv run python scripts/dedup_vs_merged.py --apply

# 2. drop frames whose every box is under 12px at the 640 input (~3901)
uv run python scripts/drop_tiny_boxes.py --apply
```

`dedup_vs_merged.py` matches on a 64-bit pHash for recall, then verifies with two
pixel correlations (intensity NCC + gradient NCC) on grey thumbnails. The
verification stage is **not optional**: most of this data is a small object on flat
sky, where pHash is near-degenerate and unrelated images collide at Hamming 0–6.
It writes a CSV of every candidate for review.

`drop_tiny_boxes.py` removes an image only when *every* box in it is sub-12px —
below YOLOv5's P3/stride-8 floor, so unreachable (measured miss rate 33% at 8–12px
vs 2.5% above 64px). Frames mixing a resolvable box with a tiny one are left alone:
stripping a box while its drone stays in frame teaches the model that a drone is
background. Empty frames are kept — they are negatives, not long-range targets.

### 2.3 Merge

```bash
uv run python scripts/merge_dataset.py            # --out data/drone --val-frac 0.2 --seed 42
```

This pairs images to labels by stem, drops unpaired files, collapses Roboflow
augmentation copies (per-source — see below), drops exact-hash duplicates, prefixes
filenames by source letter, clusters **near**-duplicates by perceptual hash, then
does a seeded 80/20 split stratified by source *and* regime, with whole pHash
clusters kept on one side so near-duplicate video frames can't leak across the split.

Outputs:

- `data/drone/{train,val}/{images,labels}/`
- `data/drone/manifest.csv` — per-image `stem, source, regime, split, cluster, orig_image`
- `configs/val_{close,mid,long,empty}.txt` — per-regime val subsets, written by the
  *same run* that decides the split, so they can't drift out of sync with it

Current build: **36,903 images**, 29,609 train / 7,294 val.

| | close | mid | long | empty |
|---|---|---|---|---|
| images | 8,878 | 9,831 | 13,059 | 5,135 |

**Regime** is derived per image from the largest ground-truth box: `close` ≥10% of
frame area, `long` <1%, `mid` between, `empty` no box. Close-range is the target
regime — track it separately, aggregate mAP mixes wildly different difficulties.

> The project brief's §5 claim that both Kaggle sets are long-range is **wrong**.
> Measured box areas show source A is already close-range (median box 33% of frame);
> source B is the long-range one (median 0.55%).

**Roboflow augmentation collapse** is per-source (the `collapse` flag). Most Roboflow
exports ship each source image up to 6× as `<stem>_jpg.rf.<hash>.jpg`; their
rotations move the pHash past any sane clustering threshold, so copies could land on
both sides of the split — hence keep one per base stem. But it must be **off** for a
source that ships no augmentation, because the base stem is upstream-provided and not
unique: `00004_jpg.rf.*` can appear in train/valid/test as three unrelated images.
Check the source's `README.roboflow.txt` for "No image augmentation techniques were
applied" before adding a source.

---

## 3. Train the float model

Primary interface is the notebook — it has run control, live metrics, per-regime
eval, and prediction visualization:

```bash
uv run jupyter lab      # then open training/train_baseline.ipynb
```

Configure the run in the `CFG` cell (name, epochs, batch, hyp overrides) and run the
launch cell; it shells out to `yolov5/train.py` and streams progress. Interrupting
the kernel terminates the child and keeps the best checkpoint so far.

The equivalent direct CLI, matching the current best run:

```bash
uv run python yolov5/train.py \
  --weights weights/yolov5n.pt \
  --data configs/drone.yaml \
  --hyp configs/hyp_gen_more_data_5.yaml \
  --epochs 100 --batch-size 192 --imgsz 640 \
  --device 0 --workers 12 --patience 30 --seed 0 \
  --project runs/train --name more_data_5
```

`weights/yolov5n.pt` is the **COCO-pretrained initialization**, not a drone model.
It is gitignored; `train.py` auto-downloads it from the ultralytics v7.0 release if
missing.

**Batch scaling gotcha.** The `configs/hyp_gen_*.yaml` files are *generated* by the
notebook, not hand-written. YOLOv5 pins its optimizer math to `nbs=64` and scales
`weight_decay` by the batch multiplier but never touches `lr0`. So for
`batch = 64*m` the notebook pre-adjusts `lr0 *= m` (linear scaling rule) and
`weight_decay /= m` (cancelling YOLOv5's own multiply), keeping the training regime
constant as batch changes. If you write a hyp file by hand at a non-64 batch, you
have to do this yourself. The current run used `m = 3` → batch 192, `lr0` 0.03.

**Augmentation.** Close-range framing comes from the in-loop augmentation
(`scale: 0.5`, `translate: 0.1`, `fliplr: 0.5`), not from an offline pass — there is
no separate augmented dataset on disk. Note `mosaic: 0.5` rather than the stock 1.0:
mosaic at 1.0 costs about 12 points of empty-frame false-alarm rate, and the damage
is **invisible to val mAP**. Don't raise it back without checking the false-alarm cell.

**SiLU→ReLU** is deliberately *not* swapped here. The float model trains with stock
SiLU for the strongest baseline; ReLU substitution happens at the QAT stage. Note
YOLOv5 shares one class-level `Conv.default_act = nn.SiLU()` across all Conv layers,
so `model.modules()` reports a single SiLU — the swap is done by rebinding
`Conv.default_act` before instantiation, not by per-layer object replacement.

### Evaluation

Notebook cells 6–6c cover what actually matters for aiming, beyond aggregate mAP:
per-regime mAP, mean IoU of matched true positives, **mean center error** in pixels
and as a fraction of image diagonal (this is the aim-precision metric), and false
positives per regime. `empty` has no ground truth, so it is scored as a background
false-alarm rate instead of mAP.

Per-regime val from the CLI:

```bash
uv run python yolov5/val.py \
  --weights runs/train/more_data_5/weights/best.pt \
  --data configs/drone_val_close.yaml --imgsz 640 --device 0
```

(`drone_val_{close,mid,long,empty}.yaml` each point `val:` at the matching
`configs/val_<regime>.txt`.)

> **Never compare mAP across runs built on different val sets.** The dataset changed
> repeatedly during development; older run numbers in `runs/` are not comparable to
> current ones.

---

## 4. Where the weights are

| Path | What |
|---|---|
| `weights/yolov5n.pt` | COCO-pretrained init used to start training. **Not a drone detector.** |
| `runs/train/<run>/weights/best.pt` | Trained drone detector, best val epoch |
| `runs/train/<run>/weights/last.pt` | Trained drone detector, final epoch |

**Current best: `runs/train/more_data_5/weights/best.pt`** — YOLOv5n, 3.9 MB,
trained on the 36.9k-image merged set.

Final val: **P 0.949 · R 0.925 · mAP50 0.963 · mAP50-95 0.662** — past the >0.85 gate.

Each run directory also holds `opt.yaml` (the exact args it ran with), `results.csv`
(per-epoch metrics), and plots. Runs are also logged to Weights & Biases; the run id
is saved to `wandb_run_id.txt` so eval metrics attach to the training run rather than
landing beside it.

`runs/` is gitignored — **these weights exist only on the machine that trained them.**
Back up any checkpoint you care about.

---

## 5. Run the model on a webcam stream

The vendored `detect.py` handles webcams natively — a numeric `--source` selects a
device index and routes through the threaded `LoadStreams` reader.

```bash
uv run python yolov5/detect.py \
  --weights runs/train/more_data_5/weights/best.pt \
  --source 0 \
  --imgsz 640 \
  --conf-thres 0.25 \
  --device 0 \
  --view-img --nosave
```

- `--source 0` — camera index; matches `/dev/video0`. Check what's attached with
  `ls /dev/video*` (a UVC camera usually claims two nodes — use the lower index).
  `v4l2-ctl --list-devices` gives more detail if you install `v4l-utils`.
- `--view-img` — live annotated preview window. Needs a display; drop it over SSH.
- `--nosave` — don't write frames out. Omit to record the session; note `detect.py`
  defaults its output to `yolov5/runs/detect/`, so pass `--project runs/detect` to
  keep it out of the vendored tree.
- `--vid-stride 2` — process every Nth frame if you want headroom.
- `--conf-thres` — raise it to cut false positives at some recall cost.

Other sources work through the same flag: a file path, a directory, a glob, an RTSP
or HTTP URL, or `screen` for a screen grab.

### Caveat: this is a demo harness, not the aiming pipeline

`detect.py` is per-frame and stateless — NMS, annotate, display. It answers *"does
the model see the drone through my camera"*, and nothing more. The tracking and
aiming logic sketched in [TODO.md](TODO.md) — seed-box selection nearest screen
center, IoU cluster gathering, weighted box fusion, Kalman predict/update with a miss
counter and re-seeding, then a hysteresis + debounce gate producing the center offset
and a `close_enough` flag — **is not implemented anywhere yet.** It needs a separate
script working off the pre-NMS boxes.

---

## 6. Repo layout

```
configs/       dataset yamls, per-regime val subsets, generated hyp files
data/          raw sources + merged set + manifest.csv        (gitignored)
export/        QONNX export                                   (phase 5, empty)
qat/           Brevitas QAT                                   (phase 4, empty)
runs/          training runs + checkpoints                    (gitignored)
scripts/       dataset merge + cleaning tools
training/      train_baseline.ipynb — float training & evaluation
weights/       COCO-pretrained yolov5n.pt                     (gitignored)
yolov5/        vendored ultralytics/yolov5 v7.0 @ 915bbf2, locally patched
```

## 7. Conventions

- Commit and push only when asked.
- Change only what's necessary; no unrequested tests or examples.
- The devkit is **undecided** — do no devkit-specific work (PYNQ image, FINN
  part-targeting, camera, actuator) until the board is chosen.
