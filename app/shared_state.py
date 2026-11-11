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
IMGSZ = 416
STABIL_MIN = 2
SAVE_EVERY = 3.0
INVENTORY_EVERY = 8.0
FACE_EVERY = 15
YOLO_EVERY = 8
INVENTORY_EVERY_FRAMES = 12

FACE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)

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
