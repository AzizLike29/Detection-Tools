import json
from pathlib import Path

ROOT = Path(r"D:\DATA C\Projek\Detection Object - Yolo8")
STATE_PATH = ROOT / "app_state.json"

CAM1 = 1
CAM2 = 1
CAM3 = 1

FACE_DIR = ROOT / "captures_face"
CAPTURE_DIR = ROOT / "captures"
THRESH = 0.40
YOLO_CONF = 0.55
IMGSZ = 640
STABIL_MIN = 3
SAVE_EVERY = 3.0
INVENTORY_EVERY = 5.0

FACE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)


def _default():
    return {"user": None, "borrowed": []}


def load_state():
    if not STATE_PATH.exists():
        return _default()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("user", None)
        data.setdefault("borrowed", [])
        return data
    except Exception:
        return _default()


def save_state(data):
    STATE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
