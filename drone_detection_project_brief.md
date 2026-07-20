# Project Brief: Close-Range Drone Detection for Precision Aiming (FPGA-Accelerated)

**Status:** Proof of Concept (PoC) phase
**Last updated:** 2026-07-20

---

## 1. Objective

Build a compact, low-latency vision system that detects a drone at close range and outputs precise angular/positional data to a downstream aiming subsystem, which will trigger a mechanical (kinetic) countermeasure.

This device is **not** responsible for coarse target acquisition — that is handled by an external system. Its sole job is **fine, accurate aim correction** in the final engagement phase.

---

## 2. Operating Requirements

| Parameter | Value |
|---|---|
| Engagement range | Up to 10 m (close range) |
| Countermeasure type | Mechanical / kinetic intercept |
| Deployment form factor | Mobile / wearable |
| Camera field of view | Narrow (precision aiming, not search) |
| Latency budget (end-to-end) | ~50–100 ms (drone at 15–20 m/s covers 1.5–2 m in 100 ms) |

### Latency budget breakdown (target)

```
Camera exposure:        ~1–5 ms
Frame transfer to FPGA: ~1–2 ms
YOLO inference (FPGA):  ~5 ms
NMS / post-processing:  ~5–10 ms
Command to actuator:    ~1–2 ms
Mechanical response:    ~20–50 ms (dominant term, outside vision system scope)
─────────────────────────────────
Total:                  ~33–74 ms (within budget)
```

---

## 3. Hardware Platform

### PoC stage (current)
- **Board:** Xilinx/AMD ZedBoard (Zynq-7000, XC7Z020)
  - 220 DSP48E1, 140× 36Kb BRAM
  - USB OTG host port (ULPI PHY, TUSB1210) — HS mode capable (480 Mbps theoretical)
  - Native GigE via onboard Ethernet (fallback interface for earliest bring-up)
- **Deployment OS:** PYNQ (official SD card image) — Linux + Jupyter + Python control of the PL accelerator, standard V4L2/UVC camera support, hot-swappable bitstream overlays.

### Next step under evaluation
- Zynq UltraScale+ class board (Ultra96-V2 or similar/better, sourcing in progress) — same FINN toolchain, more DSP/BRAM headroom, native USB3, better positioned for higher resolution/FPS.

### Long-term production target
- Custom board around **Xilinx Kintex-7 XC7K325T** (840 DSP48E1, 445× 36Kb BRAM, external DDR)
  - No hardened ARM PS — requires MicroBlaze softcore (or external host) for NMS/post-processing and control logic.
  - Same FINN-generated dataflow core is expected to retarget with re-synthesis; migration path proven conceptually, not yet executed.

---

## 4. Model & Toolchain

| Stage | Choice | Notes |
|---|---|---|
| Base architecture | **YOLOv5n** | Chosen as the practical sweet spot: fully convolutional (no attention blocks), anchor-based (simpler export than anchor-free heads), lighter and more accurate than YOLOv3/v4-tiny |
| Quantization / QAT | **Brevitas** (PyTorch) | INT8 weights/activations target; SiLU activations must be replaced with ReLU (Brevitas/FINN-compatible) before or during QAT |
| Export format | **QONNX** | via `brevitas.export.export_qonnx` |
| Hardware compiler | **FINN** (AMD/Xilinx, open source) | Converts QONNX → streaming dataflow HLS/RTL → Vivado bitstream + PYNQ deployment package (`.bit`, `.hwh`, auto-generated Python driver) |
| Deployment runtime | **PYNQ Overlay** | Python-level control of the accelerator IP, DMA, and camera capture on the same Linux system |

### Why not newer YOLO versions (v8/v10/v11/v12/26)
Newer architectures (C2f blocks, anchor-free decoupled heads, self-attention modules like C2PSA / Area Attention, DFL loss) are progressively harder or currently impractical to compile through FINN, which is fundamentally a CNN-oriented dataflow compiler. YOLO26's NMS-free, DFL-free design is architecturally attractive (would remove the need for a separate NMS stage) but its export path through FINN is unproven; revisit if/when FINN gains support for these constructs.

### Why not the Yu-Zhewen/Tiny_YOLO_v3_ZYNQ reference project
Evaluated and rejected as the primary path: the reference design is bare-metal (no OS), has **no camera interface** — each input frame must be baked into a compile-time header file — and uses an older toolchain (Vivado 2019.1). Incompatible with a live-video PoC. Its resource/latency analytical models and Design Space Exploration approach remain a useful conceptual reference for folding/parallelism tuning.

---

## 5. Datasets

| Source | Notes |
|---|---|
| Kaggle: `muki2003/yolo-drone-detection-dataset` | ~1359 images, YOLO-format annotations, multiple angles/altitudes/backgrounds |
| Kaggle: `sshikamaru/drone-yolo-detection` | Secondary set, also YOLO format, combine to increase volume |

**Known gap:** both datasets are shot at long/medium range (drone as a small object in frame). The target use case is close range (drone large in frame). Mitigation: aggressive augmentation during training —

```python
transforms = [
    RandomCrop(scale=(0.3, 1.0)),   # simulate close-range framing
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.3, contrast=0.3),
    MotionBlur(p=0.3),
    GaussianNoise(p=0.2),
]
```

Real close-range footage should replace/augment the Kaggle data once the PoC pipeline is validated end-to-end.

---

## 6. Camera

- **Interface priority for PoC:** USB (UVC) via ZedBoard's USB OTG host port, using standard Linux `v4l2` under PYNQ — no GigE Vision stack (`aravis`) needed if UVC path works.
- **Fallback for earliest bring-up:** GigE Vision camera over onboard Ethernet (simpler to validate pipeline before committing to USB host-mode kernel config).
- **Optics:** narrow FOV lens (matches the "precision aiming" role; wide-angle search is out of scope, handled externally).
- Global shutter preferred once a production-grade camera is selected (avoids rolling-shutter artifacts on fast motion); not a hard requirement for the PoC.

---

## 7. Roadmap

1. **Environment setup** — PYNQ image on ZedBoard SD card; FINN via Docker on host (Vivado installed).
2. **Float model training** — YOLOv5n on combined Kaggle dataset with close-range augmentation; target mAP50 > 0.85 as a sanity bar.
3. **Brevitas QAT** — convert to quantized layers (INT8), replace SiLU→ReLU, fine-tune from float weights (~20–30 epochs).
4. **FINN build** — QONNX → dataflow build targeting ZedBoard; expect to iterate on unsupported-operator issues (this is historically the most time-consuming step).
5. **Camera integration** — bring up UVC capture under PYNQ Linux; wire frame → accelerator → output tensor.
6. **Post-processing** — decode YOLO outputs (anchors, boxes) and NMS in Python on the ARM core.
7. **Latency validation** — instrument end-to-end timing (frame in → bbox out), compare against the ~50–100 ms budget.
8. **Iterate on real close-range footage** — replace/extend the Kaggle-based training set once hardware pipeline is proven.

---

## 8. Open Questions / Risks

- **FINN operator compatibility** for YOLOv5n specifics (SiLU replacement, C3 blocks) — not yet validated; may require architectural simplification.
- **Board sourcing** — evaluating Ultra96-V2 or a stronger Zynq UltraScale+ board as a possible upgrade path from ZedBoard; decision pending availability.
- **NMS/post-processing implementation** — no existing reference to build from; will be written from scratch in Python (PoC) and later possibly ported to MicroBlaze C for the Kintex-7 production target.
- **Kintex-7 migration** — conceptually compatible with the FINN flow (re-target + re-synthesize), but unproven; PS-less design will require MicroBlaze or external host for control/post-processing that Zynq's ARM currently handles for free.
- **Camera selection for production** — narrow-FOV, global-shutter, high-FPS camera not yet finalized; PoC will proceed with any UVC/GigE camera on hand.

---

## 9. Key References

- FINN (AMD/Xilinx): https://github.com/Xilinx/finn
- FINN examples (Tiny-YOLOv3 template): https://github.com/Xilinx/finn-examples
- Brevitas: https://github.com/Xilinx/brevitas
- PYNQ: https://www.pynq.io
- Yu-Zhewen Tiny YOLOv3 ZYNQ (reference only, not primary path): https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ
- Kaggle drone datasets: `muki2003/yolo-drone-detection-dataset`, `sshikamaru/drone-yolo-detection`
