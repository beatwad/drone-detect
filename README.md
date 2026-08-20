# drone-detect

Close-range (≤10 m) drone detection for precision aiming, targeting an
**FPGA-accelerated** deployment. A vision system detects a large-in-frame drone
and outputs angular/positional data to a downstream kinetic aiming subsystem.
Latency budget end-to-end ~50–100 ms.

**Status: Proof of Concept, host side complete.** The detector is a bitstream on
a ZCU102, the PYNQ driver exists, the numeric path from PyTorch to the built
graph is verified end to end, and the board's Linux image is built. The board
itself has not arrived, so exactly one junction is untested: real hardware
against the simulation.

```
yolov8n-P3 (float, ReLU6)  →  Brevitas QAT (W4A4)  →  QONNX  →  FINN  →  bitstream  →  PYNQ
└──────────────── host GPU, board-agnostic ────────────────┘  └──── Vivado 2022.2 ────┘
```

| Phase | What | State |
|---|---|---|
| 0 | Scaffolding | done |
| 1 | Dataset acquisition + merge | done — 36,903 images |
| 2 | Augmentation (close-range framing) | done, in-loop via hyp |
| 3 | Float training — gate mAP50 > 0.85 | done — **0.988** close, **0.993** mid |
| 4 | Brevitas QAT | done — W8A8 free, W4A4 costs ~1 pt |
| 5 | QONNX export | done, both gates pass |
| 5b–5c | FINN build → bitstream | done — 30.0% LUT, 75.8% BRAM, timing closed |
| 7 | Retarget to drones (yolov8n-P3) | done |
| 8 | Board bring-up, host side | done — driver, verification, Linux image |
| 9 | **Board bring-up, hardware** | **blocked: board not here** |
| 10 | Tracking + aim output (§11) | code exists, **never run on real data** |

The network that ships is **yolov8n-P3 ReLU6 at W4A4, 192×320**. The project
started on YOLOv5n and that line was removed once it stopped being used; the
dataset tooling it produced is still the tooling in `scripts/`, and the history
is in git.

## Where the documentation lives

| | |
|---|---|
| **this file** | how to reproduce, step by step |
| [.claude/docs/build_notes.md](.claude/docs/build_notes.md) | **why** each step is the way it is — every measurement, every trap. ~1,600 lines, organised as findings, not instructions. Read the relevant § before changing anything in `qat/`, `export/` or a FINN build. |
| [CLAUDE.md](CLAUDE.md) | current status, locked decisions, open questions |
| [deploy/petalinux/README.md](deploy/petalinux/README.md) | the board's Linux image, in detail |
| [TODO.md](TODO.md) | the original sketch of the tracking flow, now written up as §11 |

---

## 1. Install

Requires **Python 3.11** and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU with
CUDA 12.1-capable drivers is expected for training; inference falls back to CPU.

```bash
git clone <this repo> drone-detect
cd drone-detect
uv sync            # creates .venv/ from uv.lock, pinned torch 2.4.1+cu121
```

Run everything through `uv run`:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.4.1+cu121 True
```

Pinned versions that matter as a set: **torch 2.4.1+cu121, brevitas 0.13.0,
qonnx 1.0.0, onnx 1.22, ultralytics 8.3.253, numpy <2**. Brevitas/QONNX/FINN are
version-sensitive as a trio.

**The model code is the `ultralytics` package, not vendored.** Nothing in this
repo carries a copy of YOLO; `configs/yolov8n_p3_relu6.yaml` describes the
topology and Ultralytics builds it. That makes the pin load-bearing: quantization
hooks in `qat/quantize_v8.py` attach to `C2f` and `Detect` internals, and those
move between releases — `non_max_suppression` already migrated from
`ultralytics.utils.ops` to `ultralytics.utils.nms` inside the 8.3 line. Rebuild
through `uv sync`, and re-run the export gates after any bump.

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
| A | `drone_dataset` | Kaggle — [muki2003/yolo-drone-detection-dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) | close-range, single-class |
| B | `Database1` | Kaggle — [sshikamaru/drone-yolo-detection](https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection) | long-range video frames |
| C | `Drone.v1i.yolov5pytorch` | Roboflow — [project-986i8/drone-uskpc](https://universe.roboflow.com/project-986i8/drone-uskpc) | mixed range |
| D | `UAVs.v2i.yolov5pytorch` | Roboflow — [uavs-7l7kv/uavs-vqpqt](https://universe.roboflow.com/uavs-7l7kv/uavs-vqpqt) | mixed range |
| E | `Drone detection.v8i.yolov5pytorch` | Roboflow — [itzak/drone-detection-6f8tk](https://universe.roboflow.com/itzak/drone-detection-6f8tk) | + bird/plane/sky hard negatives |
| F | `Drone detection.v3i.yolov5pytorch` | Roboflow — [computer-vision-yxj4a/drone-detection-oqauc](https://universe.roboflow.com/computer-vision-yxj4a/drone-detection-oqauc) | multi-class; only `drone` kept |
| G | `Drone Detection.v1i.yolov5pytorch` | Roboflow — [ai-bmkoo/drone-detection-inlmy](https://universe.roboflow.com/ai-bmkoo/drone-detection-inlmy) | single-class |
| H | `Drone Detection.v6i.yolov5pytorch` | Roboflow — [drone-detection-g4d3g/drone-detection-a1tsf](https://universe.roboflow.com/drone-detection-g4d3g/drone-detection-a1tsf) | single-class |
| I | `Drone Detection.v5i.yolov5pytorch` | Roboflow — [aatish-kumar-sahu-57emd/drone-detection-1ghph](https://universe.roboflow.com/aatish-kumar-sahu-57emd/drone-detection-1ghph) | multi-class UAV+drone → both to 0 |

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
below the P3/stride-8 detection floor, so unreachable (measured miss rate 33% at 8–12px
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
frame area, `long` <1%, `mid` between, `empty` no box.

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

> **SiLU→ReLU is NOT swapped after training.** Train in ReLU6 from the start —
> see §4. The older instruction to "keep stock SiLU here" was measured to be
> wrong: pasting ReLU into SiLU-trained weights took mAP50 from 0.963 to 0.078.

### 3.1 The network, and how to train it

The topology is the live subgraph of the reference build: yolov8n layers 0–15
plus one stride-8 head, reproduced one-to-one in
`configs/yolov8n_p3_relu6.yaml` (41/41 ops verified against the compiled graph).
ReLU**6** rather than plain ReLU because the reference quantizes every activation
to one shared range `[0, 6]` — which is also what makes join-tying free. The
clamp is free: measured identical to plain ReLU on every regime.

```bash
uv run yolo detect train \
  model=configs/yolov8n_p3_relu6.yaml \
  pretrained=weights/yolov8n.pt \
  data=configs/drone.yaml \
  epochs=100 patience=30 batch=192 imgsz=320 \
  optimizer=SGD lr0=0.03 mosaic=0.5 device=0 \
  project=runs/train name=v8n_p3_relu6
```

Result (`runs/train/v8n_p3_relu62`): close **0.9855** / mid **0.9893** /
long 0.8691 — better on centre error *and* recall than any YOLOv5n variant
tried before it.

**Disable Ultralytics' Conv+BN fusion** before quantizing — build_notes §10.13.

### 3.2 The notebook

The notebook is the primary interface — run control, live metrics, per-regime
eval, IoU and centre-error cells, false-alarm rate, prediction visualization. It
drives the same Ultralytics trainer as the CLI above:

```bash
uv run jupyter lab      # then open training/train_baseline.ipynb
```

Set the run in the `CFG` cell and execute the launch cell; interrupting the
kernel stops training and keeps the best checkpoint so far. `CFG['model_cfg']`
selects the topology — `configs/yolov8n_p3_relu6.yaml` is the compiled one.

**Batch scaling.** Ultralytics pins its optimizer math to a nominal batch
`nbs = 64`: it scales `weight_decay` by the batch multiplier but never touches
`lr0`. So for `batch = 64*m` the notebook pre-adjusts `lr0 *= m` and
`weight_decay /= m` (`derive_hyp`), keeping the training regime constant as batch
changes. Passing a hand-written `lr0` at a non-64 batch means doing this
yourself. The shipping run used `m = 3` → batch 192, `lr0` 0.03.

**Augmentation.** Close-range framing comes from in-loop augmentation, not an
offline pass — there is no augmented dataset on disk. Note `mosaic: 0.5` rather
than the stock 1.0: mosaic at 1.0 costs ~12 points of empty-frame false-alarm
rate, and the damage is **invisible to val mAP**. Don't raise it without checking
the false-alarm cell.

### Evaluation

Per-regime mAP, mean IoU of matched true positives, **mean centre error** in
pixels and as a fraction of image diagonal, and false positives per regime.
`empty` has no ground truth and is scored as a background false-alarm rate.

```bash
uv run yolo detect val model=runs/train/v8n_p3_relu62/weights/best.pt \
  data=configs/drone_val_close.yaml imgsz=320 device=0

uv run python scripts/center_error.py \
  --weights runs/train/v8n_p3_relu62/weights/best.pt --regimes close,mid,long
```

`center_error.py` loads both checkpoint families — plain Ultralytics `.pt` and
the Brevitas QAT checkpoints — so float and quantized models are scored by
identical code. Its `--imgsz` accepts `'H,W'`, which is the only way to score at
the non-square shape the bitstream runs (`--imgsz 192,320`).

> **Never compare mAP across runs built on different val sets.** The dataset
> changed repeatedly during development; older numbers in `runs/` are not
> comparable to current ones.

### Where the weights are

| Path | What |
|---|---|
| `weights/yolov8n.pt` | COCO-pretrained init. **Not a drone detector.** |
| `runs/train/v8n_p3_relu62/weights/best.pt` | **the float model that ships** |
| `runs/qat/v8n_p3_w4a4/weights/best.pt` | **the quantized model that ships** |

---

## 4. Quantization-aware training

Read [build_notes.md](.claude/docs/build_notes.md) §1, §9, §10.9–10.10 first.
Three things here are load-bearing and none is obvious.

### The rules that cannot be repaired later

1. **Train float in ReLU/ReLU6 from the start.** Pasting ReLU into SiLU-trained
   weights destroys the model — mAP50 0.963 → **0.078**, because ~49% of
   pre-activations are negative and ReLU zeroes them across 57 layers. This
   reverses the project's original plan; the plan was wrong.
2. **Tie the quantiser at every join.** FINN can only streamline a join whose
   branches share one scale: `a·x + b·y` does not factor unless `a == b`. This
   is a QAT-time decision that **cannot** be fixed at export. Costs nothing in
   accuracy. Gate: `check_join_scales.py` must report all joins tied.
   For yolov8n-P3 the reference's "comact" convention makes this free — every
   activation shares one range, `[0, 6]`, which is why the float model is ReLU6.
3. **Every shared quantiser needs a plain `nn.ReLU()` in front of it.**

### W8A8 needs no fine-tuning at all

Post-training calibration alone scores mAP50 **0.9628** vs float 0.9629. The
fine-tune budget only earns its keep at 4-bit, where PTQ collapses (0.197).

That was measured on the YOLOv5n line, whose calibrate-only script is gone with
it; on the v8 line the same holds — W8A8 needed no recovery, W4A4 did. Set the
widths with `--low-bits` / `--high-bits` below and skip the fine-tune to
reproduce it.

### W4A4, the configuration that ships

```bash
uv run python qat/train_qat_v8.py \
  --weights runs/train/v8n_p3_relu62/weights/best.pt \
  --data configs/drone.yaml \
  --low-bits 4 --high-bits 8 \
  --epochs 30 --imgsz 320 --batch 192 --lr0 0.002 --mosaic 0.5 --device 0
```

`--low-bits 4 --high-bits 8` is the reference's mixed scheme: W8 on the stem and
head, W4 through the middle. W4 is not a micro-optimisation — it is what lets
FINN pack MACs into DSPs instead of LUT fabric, which moves the whole design into
a different cost class (LUT estimate ×1.44 instead of ×2.73, and CARRY8 usage
from 104% to 4.6%). See build_notes §10.7.

Result: `runs/qat/v8n_p3_w4a4/weights/best.pt` — close 0.9835 / mid 0.9879 /
long 0.8514, centre error unchanged at 0.0139.

The v8 line evaluates through **Ultralytics' own validator, inside the trainer**
(`train_qat_v8.py` runs it at the end and writes `runs/qat/<name>/results.csv`).
For the number that actually matters — where the box centre lands — use the aim
metric, which loads all three checkpoint families (`v5`, `v8`, `v8_qat`):

```bash
uv run python scripts/center_error.py \
  --weights runs/qat/v8n_p3_w4a4/weights/best.pt --regimes close,mid,long
```

> **Judge quantization by centre error, not mAP50-95.** The number that reaches
> the aiming subsystem is where the box centre lands. mAP50-95 moves for reasons
> that never touch aim.

---

## 5. QONNX export

```bash
uv run python export/export_qonnx_v8.py \
  --ckpt runs/qat/v8n_p3_w4a4/weights/best.pt \
  --imgsz 192 320 \
  --out export/v8n_p3_w4a4_192x320.onnx
```

Build resolution is **192×320**, not square — build_notes §10.12. `C2f.chunk`
must become `split` for FINN's frontend; the export script handles it.

Two gates, both of which must pass before FINN is worth starting:

```bash
uv run python export/check_join_scales.py --onnx export/v8n_p3_w4a4_192x320_clean.onnx
uv run python export/verify_qonnx_v8.py \
  --onnx export/v8n_p3_w4a4_192x320_clean.onnx \
  --ckpt runs/qat/v8n_p3_w4a4/weights/best.pt --n-images 40
```

`verify_qonnx_v8.py` compares detections against the torch model it came from.
Expect deltas of **exactly ±1 activation step** — that is the quantizer boundary
convention, not a bug: Brevitas rounds half-to-even on `x/s`, while the graph
compares against integer-rounded thresholds. Acceptance is "no detection gained
or lost, median centre delta < 0.5 px, nothing moves more than one cell".

---

## 6. FINN build → bitstream

**This is the step that will not reproduce from a clean checkout.** See the
honesty section below before starting. Requires Vivado/Vitis 2022.2 and runs
inside FINN's Docker; budget most of a day.

### Folding first — it is the whole game

`--target-fps` under-folds by ~11× and produces FIFOs Vivado cannot build.
Search against real resource limits instead:

```bash
uv run python export/balance_folding.py \
  --onnx export/v8n_p3_w4a4_192x320_clean.onnx \
  --headroom 0.27 --out folding.json
```

The headroom is not decoration: FINN's own estimates are low by **×1.44 on LUT**
and **×2.86 on BRAM** (it excludes FIFOs from `estimate_layer_resources`). BRAM
is the binding constraint, not LUT. Both multipliers have now been measured on
two unrelated networks and predicted the third within 3.2%, so treat them as a
budgeting rule.

### The build

```bash
# inside FINN's docker
python export/finn_build.py \
  --onnx export/v8n_p3_w4a4_192x320_clean.onnx \
  --folding-config folding.json \
  --clk-ns 10 --standalone-thresholds --bitfile
```

**`BD 5-336` will hit this build.** It is unfixed in FINN and recurs every time:
FINN registers a 460 KB `ip_repo_paths` list, Vivado's catalog evicts the
partition wrapper's definition, and `validate_bd_design` fails hours in. The
workaround is a harness that consolidates every IP into one repository directory
and patches the generated `ip_config.tcl` — build_notes §10.15. **Prepare it
before launching, not after the failure.**

Result on ZCU102 (2026-08-15): 82,222 LUT (30.0%), 691 BRAM tiles (75.8%),
334 DSP (13.3%), WNS **+1.765 ns** at 100 MHz → ~121 MHz achievable, 5.10 W of
which the PS alone is 2.74 W.

### Verify the compiled graph, do not assume it

```bash
uv run python export/verify_finn_steps.py --build <build dir> \
  --onnx export/v8n_p3_w4a4_192x320_clean.onnx --n-images 40
uv run python scripts/center_error_onnx.py --onnx <checkpoint> --n 250
```

Measured: FINN's frontend is exact (4.8e-06), streamlining touches 570 of 62,400
values by ≤2.7e-02, and **`convert_to_hw` is bit-exact**. Aim error is
unchanged. So every accuracy number recorded on the host stands for the hardware.

> **Sampling trap.** `configs/val_*.txt` is ordered by source, so `paths[:n]`
> draws from only the first one or two of nine. Sample with a stride. This
> produced a false "192×320 costs 40% of aim precision" alarm once.

---

## 7. Driver and deployment package

FINN generates the PYNQ driver in `step_make_pynq_driver` — which never runs if
the build died at `MakeZYNQProject`. Recovering it needs the parent graph
ZynqBuild produces (input IODMA → dataflow → output IODMA); the three children
survive in `intermediate_models/kernel_partitions/` and the parent is five lines
of graph rebuilt from them. build_notes §10.17.

The accelerator's contract:

```
in    UINT8   (1, 192, 320, 3)   NHWC
out   INT21   (1, 24,  40, 65)   NHWC, packed (1,24,40,65,3)
```

**The output is raw integers and the dequantization is not optional.** FINN's
`step_create_dataflow_partition` leaves the final per-channel `Mul`/`Add` in the
*parent* graph, outside the accelerator. The true value is
`int21 * scale[c] + bias[c]` with scale ~1e-4; the reference driver omits this
and applies sigmoid/softmax straight to values ~10⁴ too large. Constants live in
[deploy/v8n_p3_w4a4_192x320_dequant.npz](deploy/v8n_p3_w4a4_192x320_dequant.npz)
and change on every rebuild.

[deploy/postprocess.py](deploy/postprocess.py) does dequantize → DFL → boxes →
sigmoid → NMS in NumPy only, checked against Ultralytics' own head to 6.1e-05 px.

---

## 8. Linux image for the board

Full recipe in [deploy/petalinux/README.md](deploy/petalinux/README.md); the
short version is that PetaLinux 2022.2 runs in a container (this host is far too
new for Yocto kirkstone), the XSA comes from the Vivado project the FINN harness
built, and three settings are load-bearing: `MACHINE_NAME=zcu102-rev1.0`, the
**GTR mux hogs the device-tree generator omits** (without them SEL=0000 and USB
3.0 does not exist), and `CONFIG_USB_DWC3_DUAL_ROLE` (host-only does not link).

`petalinux-*` commands **exit 0 when they fail**; the real error is in
`build/config.log`.

---

## 9. First run on hardware

Not done — the board has not arrived.

| | |
|---|---|
| Boot mode | **SW6 [4:1] = off, off, off, on** (SD; factory default is QSPI32) |
| USB host | **J7 OPEN → ON**, **J110 1-2 → 2-3**; J109/J112/J113 unchanged |
| Console | CP2108 on J83, 115200 8N1, first of four `/dev/ttyUSB*` |

```bash
deploy/petalinux/mksd.sh /dev/sdX          # dry run
deploy/petalinux/mksd.sh /dev/sdX --yes    # writes the card
```

Then, on the board, the one check the host cannot make:

```bash
python3 run_on_board.py
```

[deploy/run_on_board.py](deploy/run_on_board.py) runs 60 frames through the real
accelerator and compares INT21 against the same frames through the simulated
graph, in LSB units. Tolerance **0.05 LSB**: a genuine error is 1.0, float32
noise at these magnitudes is ~0.008. Positive and negative controls both pass on
the host, so a failure means hardware, not harness.

Still missing before this works: **`pip install pynq` over the built XRT.** No
official PYNQ image exists for the ZCU102 — this is the one genuinely unknown
step left in the chain.

---

## 10. Run a model on a webcam stream

Ultralytics' predictor takes a camera index directly:

```bash
uv run yolo predict \
  model=runs/train/v8n_p3_relu62/weights/best.pt \
  source=0 imgsz=320 conf=0.25 device=0 show=True \
  project=runs/predict
```

- `source=0` — camera index, matching `/dev/video0`. Check what is attached with
  `ls /dev/video*`; a UVC camera usually claims two nodes, use the lower index.
  `v4l2-ctl --list-devices` gives more detail.
- `show=True` — live annotated window. Needs a display; drop it over SSH.
- `save=False` — don't write frames out. Other sources work through the same
  flag: a file, a directory, a glob, or an RTSP/HTTP URL.

### Caveat: this is a demo harness

`predict` is per-frame and stateless — NMS, annotate, display. It answers *"does
the model see the drone through my camera"*, and nothing more. Everything that
turns detections into an aim command is specified in §11 and **not implemented**.

---

## 11. Tracking and aim output

[deploy/track.py](deploy/track.py) — `AimTracker`, one instance per camera,
`update()` once per frame.

**What is and is not done.** The algorithm is written and exercised on synthetic
tracks: it converges on a crossing target, holds state through a detection
dropout, rejects a teleporting outlier and re-seeds after `MISS_LIMIT`, holds the
centring flag inside the hysteresis band and drops it on a slow drift past
`D_high`. It has **never seen a real detection**, never run on the board, and
**every threshold in it is a guess** — no footage exists through the real lens
yet, and the centring gate depends on the aiming subsystem's tolerance, which is
not specified. Treat the numbers as placeholders.

It consumes the **pre-NMS** boxes — `deploy/postprocess.decode()` output, before
`nms()` — and emits two things for the aiming subsystem: the centre offset, and a
`close_enough` flag.

Usage:

```python
from deploy.postprocess import dequantize, decode
from deploy.track import AimTracker

tracker = AimTracker(frame_wh=(320, 192))
...
feat = dequantize(raw_nhwc, scale, bias)
boxes, conf = decode(feat)                 # PRE-NMS
aim = tracker.update(boxes[0], conf[0, :, 0], dt)
if aim.close_enough:
    dx, dy = aim.offset                    # pixels from frame centre
```

`dt` is seconds since the previous frame's **exposure**, and is passed in rather
than measured inside: the filter has to advance from when the photons landed, not
from when the frame reached userspace. That is the same argument as the camera
trigger, and until there is one, `dt` is only as good as the timestamp available.

```
1. Seed box      of the pre-NMS boxes, keep conf > thresh_conf;
                 pick the one closest to screen centre.
                 none -> close_enough = False, exit.
2. Cluster       collect every box with IoU > thresh_iou against the seed.
3. Fuse          WBF over the cluster -> one measured box.
4. Predict       Kalman predict -> predicted box.
5. Gate          IoU(measured, predicted) > thresh_frame_iou ?
                   yes -> miss_counter = 0
                   no  -> miss_counter += 1, append measured to the recent list
                          if miss_counter > M:
                              re-seed the filter from the last consecutive boxes
                              that agree with each other by thresh_frame_iou
                          close_enough = False, exit.
6. Update        Kalman update -> smoothed centre -> offset from frame centre,
                 and its magnitude `dist`.
7. Centring gate hysteresis on `dist` (D_low / D_high) + debounce over N frames
                 -> close_enough.
```

Notes that are not in the sketch but follow from decisions already made:

- **Step 1 assumes a boresight-aligned frame.** Picking the box nearest the
  centre is only meaningful because the deployment plan is a centre crop, which
  needs no coordinate transform to become an aim angle. If the crop ever becomes
  steerable, this step needs the transform.
- **Single target by construction.** The seed-and-cluster structure resolves one
  object, deliberately: the system aims at one drone. Full NMS is not required —
  the cluster step already collapses duplicates around the seed.
- **Alpha-beta, not a full Kalman.** A fixed-gain filter is what a
  constant-velocity Kalman converges to, without covariance bookkeeping. The gate
  needs a predicted *box*, so size is carried too — as a plain EMA, since size has
  no useful dynamics here. Replace this first if measurement noise ever needs
  estimating rather than assuming.
- **Distances are fractions of the frame diagonal**, not pixels — the unit aim
  error is already reported in (`scripts/center_error.py`, 0.0139 for the
  shipping model), so `D_low` / `D_high` can be read against existing numbers.
- **Runs on the A53s under Linux — for the baseline only.** Nothing above needs
  the fabric: the tracking itself is ~10k operations per frame, microseconds
  either way. The point of the baseline is to prove the pipeline end to end, and
  Linux on the A53s is where the capture and the driver already live, so that is
  the shortest path. **This is explicitly temporary.** Full determinism needs the
  whole chain — preprocessing, decode, tracker — in PL, fed by a MIPI camera
  straight into the fabric, with no Linux scheduler between exposure and aim
  command. Until then, scheduler jitter lands directly in aim error.
- **Order the decode to avoid the softmax.** A naive implementation dequantizes
  everything and runs DFL over all 960 cells: 960 x 4 x 16 = **61,440 `exp`
  calls**. Instead dequantize the class channel only, threshold in *logit* space
  (`conf > 0.25` is `logit > -1.0986`, so no sigmoid at all), and run DFL only on
  the survivors — of order 1,000 `exp` calls. This is an operation count, not a
  measurement; the constant depends on the libm.
- **Parameters unset.** `thresh_conf`, `thresh_iou`, `thresh_frame_iou`, `M`,
  `D_low`, `D_high`, `N` all need measuring against real footage, which needs the
  camera and the lens — open question 1.

---

## What will NOT reproduce out of the box

Stated plainly, because a clean list of commands would otherwise be a lie.

- **FINN needs the authors' fork, not upstream.** Neither `finn` v0.10.1 nor
  `finn-dev` works alone: v0.10.1 cannot compile a joined graph at all, and the
  authors' fork ships a `finn-hlslib` that predates a fix every concat needs. The
  working combination is their fork **plus one `concat.hpp` from finn-dev**.
  build_notes §10.1–10.2; the fork and a sparse-checkout recipe are in
  [References](#references).
- **`BD 5-336` is unfixed upstream** and will hit every bitstream build. The
  harness that works around it (`ip_config_drone.tcl`, a consolidated IP
  repository, `run.sh`/`inner.sh`) lives outside this repo, in the FINN build
  directory. build_notes §10.15 describes it completely; it is not committed.
- **Vivado/Vitis 2022.2 exactly.** Not the latest, and not the 2022.2.2 update.
- **PetaLinux 2022.2** is a 2.7 GB installer behind an AMD account login.
- **The reference YOLOv8n's Brevitas source is not public.** `qat/quantize_v8.py`
  reproduces its quantization scheme by matching the released ONNX op-for-op
  (41/41 verified), not by using their code.
- **Datasets are not redistributable** — nine sources, two from Kaggle (needs an
  API token) and seven downloaded by hand from Roboflow Universe.
- **`data/`, `runs/` and `weights/` are gitignored.** Trained checkpoints exist
  only on the machine that trained them. Back up anything you care about.

## Repo layout

```
.claude/docs/  build_notes.md — the measured record behind every decision
configs/       dataset yamls, per-regime val subsets, model yamls, generated hyps
data/          raw sources + merged set + manifest.csv          (gitignored)
deploy/        postprocess.py, track.py, run_on_board.py, dequant constants
  petalinux/   the board's Linux image: Dockerfile, configure.sh, dtsi, mksd.sh
export/        QONNX export, verification gates, FINN driver + folding search
qat/           Brevitas QAT — quantize_v8.py builds the graph, train_qat_v8.py fine-tunes
runs/          training runs + checkpoints                      (gitignored)
scripts/       dataset merge + cleaning, centre-error metrics
training/      train_baseline.ipynb — float training & evaluation
weights/       COCO-pretrained initialisations                  (gitignored)
```

The model code itself is the `ultralytics` package in `.venv/`; nothing is
vendored.

## Conventions

- Commit and push only when asked.
- Change only what's necessary; no unrequested tests or examples.
- The devkit is **ZCU102** (chosen 2026-08-03) and is deliberately oversized —
  experiment freely, but do not let its headroom drive architecture decisions.
  The deployment target is a smaller ~2–5 W UltraScale+ part, still undecided.
- Record the on-chip footprint of anything trained, so we know what ports.

## What the original plan got wrong

Kept from the phased host-side plan this README replaces, because the errors are
more instructive than the plan was.

- **The SiLU→ReLU ordering.** The plan said swap at QAT; measurement said the
  opposite. Caught only because the pasted-ReLU model scored 0.078. The single
  costliest planning error.
- **"Fine-tune 20–30 epochs to recover accuracy."** At INT8 there was nothing to
  recover. Budgeted training time for a problem that did not exist.
- **The dataset regime assumption**, inherited from the brief: it claimed both
  sources were long/medium range. Source A was already close-range (median box
  33% of frame). We had the data we needed and did not know it.
- **"C3 blocks may need simplification for clean FINN compilation."** They did
  not. C3, SPPF and the full FPN all compile untouched.
- **"FINN cannot compile branched networks."** Refuted 2026-08-04. Joins compile;
  they just need their branches to share one quantisation scale.
- **What the plan never anticipated at all:** that the hard part would be neither
  accuracy nor quantization, but **join scales, folding, and toolchain
  archaeology**. Phases 0–5 landed close to schedule. Everything expensive has
  been downstream of the QONNX handoff.

---

## References

The FPGA path is not original work — it follows two published builds, and two
FINN discussions supply numbers used in the resource budget.

- **Danilowicz & Kryjak, ARC 2025** — branched YOLOv8n on ZCU102, W4A4, 320×192.
  <https://arxiv.org/abs/2503.13023>. The topology in
  `configs/yolov8n_p3_relu6.yaml` reproduces the live subgraph of their released
  model, and their FINN fork is the one that compiles it.
  Their fork: <https://github.com/mdanilow/finn> branch `yolov8_dev` — **read it,
  don't clone it**; the substance is merged upstream. Sparse checkout, to avoid
  pulling a full FINN history:

  ```bash
  git clone --depth 1 --branch yolov8_dev --filter=blob:none --no-checkout \
      https://github.com/mdanilow/finn.git
  cd finn && git sparse-checkout set notebooks/experiments && git checkout
  ```

  Useful files: `notebooks/experiments/yolov8/{build_yolov8.py,
  final_hw_config_90fps.json,yolov8_output_dir/report/}`.
- **Calì, Falaschetti & Biagetti, Electronics 2025, 14, 3993** — YOLOv3-Tiny on a
  Zynq-7020, 208 FPS, 2.55 W. <https://doi.org/10.3390/electronics14203993>.
  Source of the folding-balance method (§3.6.1) that `export/balance_folding.py`
  implements, and of the ×2 BRAM rule used for portability estimates.
  Code: <https://github.com/sn0wst0rm/FINN-VisDrone-YOLO> ·
  thesis: <https://tesi.univpm.it/handle/20.500.12075/20897>
- **LPYOLO** (Günay, Okcu, Bilge 2022) — the network the Electronics paper
  reuses. <https://github.com/sefaburakokcu/quantized-yolov5>
- **FINN discussion 1021 — DSP packing in MVAU/VVU.**
  <https://github.com/Xilinx/finn/discussions/1021>. Where the MAC-per-DSP
  figures come from: RTL DSP48E2 packs 4 MACs at W4A4 and 2 at W8A8, HLS packs
  none. This is why `MVAU_rtl` at W4 is a different cost class, not a tweak.
- **FINN discussion 383 — FIFO depth between layers.**
  <https://github.com/Xilinx/finn/discussions/383>. The over/under-sizing
  tradeoff behind three failed bitfile builds.
- **RTL ConvolutionInputGenerator** (`parallel_window`) —
  <https://finn.readthedocs.io/en/latest/internals.html#rtl-convolutioninputgenerator>

Toolchain homes: [FINN](https://github.com/Xilinx/finn) ·
[Brevitas](https://github.com/Xilinx/brevitas) · [PYNQ](https://www.pynq.io) ·
[Ultralytics](https://github.com/ultralytics/ultralytics)

### Not the primary path, but worth knowing

- **Yu-Zhewen, Tiny YOLOv3 on Zynq** —
  <https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ>. A different route to the same
  destination: it does not go through FINN.
