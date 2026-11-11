import json
from pathlib import Path

import cv2

ROOT = Path(r"D:\DATA C\Projek\Detection Object - Yolo8")
STATE_PATH = ROOT / "app_state.json"

CAM1 = 1
CAM2 = 1
CAM3 = 1

FACE_DIR = ROOT / "captures_face"
CAPTURE_DIR = ROOT / "captures"
THRESH = 0.40
YOLO_CONF = 0.55
YOLO_IOU = 0.5
# 480: ringan di CPU, lebih tajam dari 416; naikkan ke 640 jika ada GPU
IMGSZ = 480
STABIL_MIN = 2
SAVE_EVERY = 3.0
INVENTORY_EVERY = 8.0
FACE_EVERY = 15
YOLO_EVERY = 5
INVENTORY_EVERY_FRAMES = 8

FACE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)


def _detect_device():
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()


def run_yolo(model, frame):
    """Track + persist: box lebih stabil; dipanggil tiap YOLO_EVERY frame saja."""
    return model.track(
        frame,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=IMGSZ,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        device=DEVICE,
    )[0]


def draw_detections(frame, result):
    """Gambar box di frame live — lebih ringan daripada result.plot() tiap loop."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return frame
    shown = frame.copy()
    names = result.names
    for b in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        label = f"{names[cls_id]} {conf:.2f}"
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            shown,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return shown

_cache_mtime = None
_cache_data = None


def _default():
    return {"user": None, "borrowed": []}


def load_state():
    global _cache_mtime, _cache_data
    if not STATE_PATH.exists():
        _cache_mtime = None
        _cache_data = _default()
        return _cache_data
    try:
        mtime = STATE_PATH.stat().st_mtime
        if _cache_data is not None and mtime == _cache_mtime:
            return _cache_data
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("user", None)
        data.setdefault("borrowed", [])
        _cache_mtime = mtime
        _cache_data = data
        return data
    except Exception:
        return _default()


def save_state(data):
    global _cache_mtime, _cache_data
    STATE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _cache_data = data
    try:
        _cache_mtime = STATE_PATH.stat().st_mtime
    except OSError:
        _cache_mtime = None


def set_user(nama):
    data = load_state()
    data["user"] = nama
    save_state(data)


def clear_user():
    data = load_state()
    data["user"] = None
    save_state(data)


def get_user():
    return load_state().get("user")


def get_borrowed():
    return set(load_state().get("borrowed") or [])


def add_borrowed(tool):
    data = load_state()
    borrowed = set(data.get("borrowed") or [])
    borrowed.add(tool)
    data["borrowed"] = sorted(borrowed)
    save_state(data)


def remove_borrowed(tool):
    data = load_state()
    borrowed = set(data.get("borrowed") or [])
    borrowed.discard(tool)
    data["borrowed"] = sorted(borrowed)
    save_state(data)
