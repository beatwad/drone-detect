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

## Reference (not primary path)
- Yu-Zhewen Tiny YOLOv3 ZYNQ — https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ
