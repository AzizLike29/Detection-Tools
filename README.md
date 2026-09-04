# Detection-Tools

Realtime **YOLOv8** detector for workshop hand tools. A custom model identifies **11 tools** from still images or a webcam (OpenCV). The same weights are intended for later deployment on an **NVIDIA Jetson**.

## Features

- YOLOv8n custom training with [Ultralytics](https://github.com/ultralytics/ultralytics)
- Image inference and live webcam demo
- 11 workshop tool classes
- Dataset in standard YOLO format (`train` / `valid` / `test`)

## Detected classes

| ID | Class |
|----|--------|
| 0 | `gergaji_besi` |
| 1 | `gergaji_kayu` |
| 2 | `gunting` |
| 3 | `gunting_plat` |
| 4 | `kunci_t` |
| 5 | `meteran` |
| 6 | `penggaris_siku` |
| 7 | `tang_rivet` |
| 8 | `tang_scun_hidrolik` |
| 9 | `tekiro_c_clamp` |
| 10 | `water_pass` |

The model only detects these classes. Other objects are ignored or may be mislabeled.

## Project structure

```text
Detection-Tools/
├── app/
│   └── webcam_test.py       # live camera demo
├── notebooks/
│   └── 01_train_test.ipynb  # setup, train, test
├── train/  valid/  test/    # images + YOLO labels
├── data.yaml
└── best.pt                  # trained weights (keep local if not in git)
```

## Requirements

- Python 3.10+
- A webcam for the live demo

```bash
pip install ultralytics opencv-python
```

## Quick start (webcam)

1. Place `best.pt` in the project root.
2. In `app/webcam_test.py`, set `ROOT` to this project folder.
3. Run:

```bash
python app/webcam_test.py
```

On Windows, if `python` is not found:

```bash
py -3.10 app/webcam_test.py
```

Press **Q** in the video window to quit.

## Train

Set `path:` in `data.yaml` to this project folder, then:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU; use "cpu" if no NVIDIA GPU
)
```

Training on CPU is slow. Google Colab (T4 GPU) is recommended. After training, copy `runs/.../weights/best.pt` to the project root.

## Test on images

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="test/images", save=True, conf=0.5)
```

## NVIDIA Jetson

1. Train on a GPU machine (PC or Colab).
2. Copy `best.pt` to the Jetson.
3. Export TensorRT when ready:

```python
from ultralytics import YOLO

YOLO("best.pt").export(format="engine", imgsz=640, half=True)
```

## Notes

The current labeled set is small (~62 images). Accuracy is limited, especially when a tool is **held in hand** (most training photos show tools on a table). More photos per class — handheld shots, varied lighting, and empty-background negatives — will improve results more than changing inference code.

When exporting from Roboflow, avoid **Stretch** resize to 512×512; keep the original aspect ratio (letterbox).
