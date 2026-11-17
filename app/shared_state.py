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
YOLO_CONF = 0.45
YOLO_IOU = 0.5
IMGSZ = 640
STABIL_MIN = 2
SAVE_EVERY = 3.0
INVENTORY_EVERY = 8.0
FACE_EVERY = 15
YOLO_EVERY = 5
INVENTORY_EVERY_FRAMES = 8

# Stok awal tools yang dihitung jumlahnya
STOCK_QTY = {
    "penggaris_siku": 5,
    "tang_rivet": 3,
    "tang_kombinasi": 5,
    "tang_potong": 5,
    "klem_plat": 2,
    "meteran": 4,
    "penggaris_100cm": 3,
    "penggaris_30cm": 3,
    "kunci_t": 4,
    "bor_tangan": 3,
    "gerinda": 4,
}

FACE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)


def _detect_device():
    """Pilih GPU jika ada, kalau tidak pakai CPU"""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()


def run_yolo(model, frame):
    """Jalankan YOLO track supaya box lebih stabil antar frame"""
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
    """Gambar box deteksi di frame untuk tampilan live"""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return frame
    display_frame = frame.copy()
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{names[class_id]} {confidence:.2f}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display_frame,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return display_frame


_cache_mtime = None
_cache_data = None


def _default_state():
    """State kosong untuk user dan daftar pinjaman"""
    return {
        "user": None,
        "borrowed": [],
        "borrowed_qty": {},
    }


def load_state():
    """Baca app_state.json dengan cache berdasarkan waktu file"""
    global _cache_mtime, _cache_data
    if not STATE_PATH.exists():
        _cache_mtime = None
        _cache_data = _default_state()
        return _cache_data
    try:
        mtime = STATE_PATH.stat().st_mtime
        if _cache_data is not None and mtime == _cache_mtime:
            return _cache_data
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("user", None)
        data.setdefault("borrowed", [])
        data.setdefault("borrowed_qty", {})
        _cache_mtime = mtime
        _cache_data = data
        return data
    except Exception:
        return _default_state()


def save_state(data):
    """Tulis state ke app_state.json dan perbarui cache"""
    global _cache_mtime, _cache_data
    STATE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _cache_data = data
    try:
        _cache_mtime = STATE_PATH.stat().st_mtime
    except OSError:
        _cache_mtime = None


def set_user(person_name):
    """Set user sesi aktif"""
    data = load_state()
    data["user"] = person_name
    save_state(data)


def clear_user():
    """Kosongkan user sesi aktif"""
    data = load_state()
    data["user"] = None
    save_state(data)


def get_user():
    """Ambil nama user sesi aktif"""
    return load_state().get("user")


def get_borrowed():
    """Ambil set nama tools yang sedang dipinjam"""
    data = load_state()
    names = set(data.get("borrowed") or [])
    for tool, qty_value in (data.get("borrowed_qty") or {}).items():
        if int(qty_value) > 0:
            names.add(tool)
    return names


def get_borrowed_qty(tool):
    """Ambil jumlah unit tool yang sedang dipinjam"""
    return int((load_state().get("borrowed_qty") or {}).get(tool, 0))


def get_sisa(tool):
    """Hitung sisa stok tool dari STOCK_QTY dikurangi yang dipinjam"""
    if tool not in STOCK_QTY:
        return None
    return STOCK_QTY[tool] - get_borrowed_qty(tool)


def add_borrowed(tool, qty=1):
    """Catat pinjam tool (qty untuk STOCK_QTY, toggle untuk tools lain)"""
    data = load_state()
    qty = max(1, int(qty))
    if tool in STOCK_QTY:
        current_qty = int((data.get("borrowed_qty") or {}).get(tool, 0))
        remaining = STOCK_QTY[tool] - current_qty
        taken_count = min(qty, remaining)
        if taken_count <= 0:
            return 0
        data.setdefault("borrowed_qty", {})[tool] = current_qty + taken_count
        save_state(data)
        return taken_count
    borrowed = set(data.get("borrowed") or [])
    borrowed.add(tool)
    data["borrowed"] = sorted(borrowed)
    save_state(data)
    return 1


def remove_borrowed(tool, qty=1):
    """Catat pengembalian tool (qty untuk STOCK_QTY, hapus dari list untuk tools lain)"""
    data = load_state()
    qty = max(1, int(qty))
    if tool in STOCK_QTY:
        current_qty = int((data.get("borrowed_qty") or {}).get(tool, 0))
        returned_count = min(qty, current_qty)
        if returned_count <= 0:
            return 0
        new_qty = current_qty - returned_count
        borrowed_qty_map = data.setdefault("borrowed_qty", {})
        if new_qty <= 0:
            borrowed_qty_map.pop(tool, None)
        else:
            borrowed_qty_map[tool] = new_qty
        save_state(data)
        return returned_count
    borrowed = set(data.get("borrowed") or [])
    if tool not in borrowed:
        return 0
    borrowed.discard(tool)
    data["borrowed"] = sorted(borrowed)
    save_state(data)
    return 1
