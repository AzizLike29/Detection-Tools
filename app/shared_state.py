from pathlib import Path

import cv2
import pymysql

ROOT = Path(r"D:\DATA C\Projek\Detection Object - Yolo8")

# Koneksi MySQL XAMPP
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "SFSmartMaintenanceDB",
    "charset": "utf8mb4",
    "autocommit": True,
}

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

FACE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def _get_open_header_id(cur):
    """Ambil ID sesi pinjam yang sedang open"""
    cur.execute(
        "SELECT intIdBorrowHeader FROM trBorrowHeader "
        "WHERE txtStatus = 'open' ORDER BY intIdBorrowHeader DESC LIMIT 1"
    )
    row = cur.fetchone()
    return row[0] if row else None


def _get_tool(cur, yolo_class):
    """Ambil ID, stok, dan mode qty dari tool berdasarkan class YOLO"""
    cur.execute(
        "SELECT intIdTools, intStockQty, bitTrackQty FROM mTools "
        "WHERE txtYoloClass = %s AND bitIsActive = 1",
        (yolo_class,),
    )
    return cur.fetchone()


def _get_active_detail(cur, tool_id):
    """Ambil baris pinjaman yang masih borrowed untuk satu tool"""
    cur.execute(
        "SELECT intIdBorrowDetail, intQty FROM trBorrowDetail "
        "WHERE intIdTools = %s AND txtStatus = 'borrowed'",
        (tool_id,),
    )
    return cur.fetchone()


def log_event(
    camera_name,
    action,
    user_name=None,
    yolo_class=None,
    qty=1,
    confidence=None,
    img_path=None,
):
    """Catat satu kejadian ke trEventLog. User dan tool dicari lewat nama/class"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO trEventLog
            (intIdUser, intIdTools, txtCamera, txtAction, intQty, fltConfidence, txtImgPath, dtEvent, txtInsertedBy, dtInserted)
            VALUES (
            (SELECT intIdUser FROM mUsers WHERE txtName=%s),
            (SELECT intIdTools FROM mTools WHERE txtYoloClass=%s),
            %s, %s, %s, %s, %s, NOW(), %s, NOW())
            """,
            (user_name, yolo_class, camera_name, action, qty, confidence, img_path, camera_name),
        )


def is_qty_tool(tool):
    """Cek apakah tool dihitung qty (bitTrackQty = 1) atau toggle"""
    with get_conn() as conn, conn.cursor() as cur:
        tool_row = _get_tool(cur, tool)
    return bool(tool_row and tool_row[2])


def set_user(person_name):
    """Mulai sesi baru untuk user yang dikenali Cam1"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT intIdUser FROM mUsers WHERE txtName = %s AND bitIsActive = 1",
            (person_name,),
        )
        user_row = cur.fetchone()
        if user_row is None:
            return
        cur.execute(
            "UPDATE trBorrowHeader SET txtStatus = 'closed', dtReturnDate = NOW(), "
            "dtUpdated = NOW() WHERE txtStatus = 'open'"
        )
        cur.execute(
            "INSERT INTO trBorrowHeader "
            "(intIdUser, dtBorrowDate, txtStatus, txtInsertedBy, dtInserted) "
            "VALUES (%s, NOW(), 'open', 'cam1', NOW())",
            (user_row[0],),
        )
    log_event("cam1", "login", user_name=person_name)


def clear_user():
    """Tutup sesi user yang sedang aktif"""
    person_name = get_user()
    if person_name is None:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE trBorrowHeader SET txtStatus = 'closed', dtReturnDate = NOW(), "
            "dtUpdated = NOW() WHERE txtStatus = 'open'"
        )
    log_event("cam1", "logout", user_name=person_name)


def get_user():
    """Ambil nama user dari sesi yang sedang open"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT u.txtName FROM trBorrowHeader h "
            "JOIN mUsers u ON u.intIdUser = h.intIdUser "
            "WHERE h.txtStatus = 'open' ORDER BY h.intIdBorrowHeader DESC LIMIT 1"
        )
        row = cur.fetchone()
    return row[0] if row else None


def get_borrowed():
    """Ambil set class YOLO yang sedang dipinjam"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT t.txtYoloClass FROM trBorrowDetail d "
            "JOIN mTools t ON t.intIdTools = d.intIdTools "
            "WHERE d.txtStatus = 'borrowed'"
        )
        return {row[0] for row in cur.fetchall()}


def get_borrowed_qty(tool):
    """Ambil jumlah unit tool yang sedang dipinjam"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(d.intQty), 0) FROM trBorrowDetail d "
            "JOIN mTools t ON t.intIdTools = d.intIdTools "
            "WHERE t.txtYoloClass = %s AND d.txtStatus = 'borrowed'",
            (tool,),
        )
        return int(cur.fetchone()[0])


def get_sisa(tool):
    """Hitung sisa stok tool dari intStockQty dikurangi yang dipinjam"""
    with get_conn() as conn, conn.cursor() as cur:
        tool_row = _get_tool(cur, tool)
    if tool_row is None or not tool_row[2]:
        return None
    return tool_row[1] - get_borrowed_qty(tool)


def add_borrowed(tool, qty=1):
    """Catat pinjam tool. Tool qty memakai jumlah box, tool toggle hanya 1 unit"""
    qty = max(1, int(qty))
    with get_conn() as conn, conn.cursor() as cur:
        header_id = _get_open_header_id(cur)
        tool_row = _get_tool(cur, tool)
        if header_id is None or tool_row is None:
            return 0
        tool_id, stock_qty, track_qty = tool_row
        active_detail = _get_active_detail(cur, tool_id)

        if track_qty:
            borrowed_now = active_detail[1] if active_detail else 0
            count = min(qty, stock_qty - borrowed_now)
        else:
            if active_detail:
                return 0
            count = 1
        if count <= 0:
            return 0

        if active_detail:
            cur.execute(
                "UPDATE trBorrowDetail SET intQty = intQty + %s, dtUpdated = NOW() "
                "WHERE intIdBorrowDetail = %s",
                (count, active_detail[0]),
            )
        else:
            cur.execute(
                "INSERT INTO trBorrowDetail "
                "(intIdBorrowHeader, intIdTools, intQty, dtBorrowDate, txtStatus, txtInsertedBy, dtInserted) "
                "VALUES (%s, %s, %s, NOW(), 'borrowed', 'cam2', NOW())",
                (header_id, tool_id, count),
            )
    return count


def remove_borrowed(tool, qty=1):
    """Catat pengembalian tool. Sisa qty tetap borrowed, habis berarti returned"""
    qty = max(1, int(qty))
    with get_conn() as conn, conn.cursor() as cur:
        tool_row = _get_tool(cur, tool)
        if tool_row is None:
            return 0
        tool_id = tool_row[0]
        active_detail = _get_active_detail(cur, tool_id)
        if active_detail is None:
            return 0

        detail_id, borrowed_now = active_detail
        returned_count = min(qty, borrowed_now)
        remaining_qty = borrowed_now - returned_count
        if remaining_qty > 0:
            cur.execute(
                "UPDATE trBorrowDetail SET intQty = %s, dtUpdated = NOW() "
                "WHERE intIdBorrowDetail = %s",
                (remaining_qty, detail_id),
            )
        else:
            cur.execute(
                "UPDATE trBorrowDetail SET txtStatus = 'returned', dtReturnDate = NOW(), "
                "dtUpdated = NOW() WHERE intIdBorrowDetail = %s",
                (detail_id,),
            )
    return returned_count


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
