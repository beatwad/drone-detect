# Project Links

## Datasets (Kaggle)
- `muki2003/yolo-drone-detection-dataset` — https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset
  (~1359 images, YOLO-format, multiple angles/altitudes/backgrounds; primary set)
- `sshikamaru/drone-yolo-detection` — https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection
  (secondary YOLO-format set, combined to increase volume)

Pull with: `uv run kaggle datasets download -d <slug> -p data/raw --unzip`
(needs `~/.kaggle/kaggle.json`, `chmod 600`).

### Candidates (Roboflow Universe)
Relatively high performance, but a lot of garbage images


Used:
https://universe.roboflow.com/pauls-workspace-bpzqa/universal-drone-tracker
https://universe.roboflow.com/project-986i8/drone-uskpc
https://universe.roboflow.com/itzak/drone-detection-6f8tk
https://universe.roboflow.com/computer-vision-yxj4a/drone-detection-oqauc
https://universe.roboflow.com/ai-bmkoo/drone-detection-inlmy
https://universe.roboflow.com/drone-detection-g4d3g/drone-detection-a1tsf
https://universe.roboflow.com/aatish-kumar-sahu-57emd/drone-detection-1ghph


## Toolchain
- FINN (AMD/Xilinx) — https://github.com/Xilinx/finn
- FINN examples (Tiny-YOLOv3 template) — https://github.com/Xilinx/finn-examples
- Brevitas — https://github.com/Xilinx/brevitas
- PYNQ — https://www.pynq.io
- YOLOv5 (vendored source, v7.0) — https://github.com/ultralytics/yolov5

## Papers — the two reference FINN-YOLO builds (central to the FPGA work)
- **Danilowicz & Kryjak, ARC 2025** — branched YOLOv8n on **ZCU102**, 195.3 FPS
  @300 MHz, W4A4, 320x192. https://arxiv.org/abs/2503.13023
  - Their FINN fork: https://github.com/mdanilow/finn branch `yolov8_dev`
    (read, don't clone — substance merged upstream; see memory
    `finn-yolo-reference-repos`). Useful files:
    `notebooks/experiments/yolov8/{build_yolov8.py,final_hw_config_90fps.json,yolov8_output_dir/report/}`
  - Sparse-checkout recipe (avoids a full clone):
    `git clone --depth 1 --branch yolov8_dev --filter=blob:none --no-checkout https://github.com/mdanilow/finn.git`
    then `git sparse-checkout set notebooks/experiments && git checkout`
- **Calì, Falaschetti & Biagetti, Electronics 2025, 14, 3993** — YOLOv3-Tiny on
  **Zynq-7020**, 208 FPS @200 MHz, 2.55 W. This is the paper `configs/yolov5_pico.yaml`
  is modelled on, and the source of the folding-balance method (§3.6.1) behind
  `export/balance_folding.py`. https://doi.org/10.3390/electronics14203993
  - Code: https://github.com/sn0wst0rm/FINN-VisDrone-YOLO
  - Thesis: https://tesi.univpm.it/handle/20.500.12075/20897
- **LPYOLO** (Günay, Okcu, Bilge 2022) — the network definition the Electronics
  paper reuses. Code: https://github.com/sefaburakokcu/quantized-yolov5

## FINN discussions worth reading (cited by the Electronics paper)
- **DSP packing in MVAU/VVU** — https://github.com/Xilinx/finn/discussions/1021
  (source of their Table 4: RTL DSP48E2 = 4 MAC/DSP at W4A4, 2 at W8A8; HLS = 1,
  no packing. Explains why our MVAU_hls design uses 7 DSPs of 2,520.)
- **FIFO depth between layers** — https://github.com/Xilinx/finn/discussions/383
  (the over/under-sizing tradeoff that cost us three failed bitfile builds)
- RTL ConvolutionInputGenerator (parallel_window) —
  https://finn.readthedocs.io/en/latest/internals.html#rtl-convolutioninputgenerator

## Reference (not primary path)
- Yu-Zhewen Tiny YOLOv3 ZYNQ — https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ
