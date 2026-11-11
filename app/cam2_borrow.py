import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import (
    CAM1,
    CAM2,
    CAPTURE_DIR,
    DEVICE,
    ROOT,
    SAVE_EVERY,
    STABIL_MIN,
    YOLO_EVERY,
    add_borrowed,
    draw_detections,
    get_borrowed,
    get_user,
    remove_borrowed,
    run_yolo,
)
from tts_util import ucap

yolo = YOLO(str(ROOT / "best.pt"))

cap = cv2.VideoCapture(CAM2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Kamera 2 (index {CAM2}) tidak bisa dibuka")

if CAM1 == CAM2:
    print("Peringatan: CAM1 dan CAM2 index sama. Jalankan satu kamera saja agar tidak patah.")
print(f"Cam2 pinjam/kembali (device={DEVICE}). Arahkan tools ke kamera. Q keluar")

n = 0
last_result = None
kandidat = None
stabil = 0
had_empty = True
last_event = 0.0
cached_user = None
user_refresh = 0.0


def catat(nama, aksi, tool, shown):
    nama_alat = tool.replace("_", " ")
    jam = datetime.now().strftime("%H.%M")
    ucap(f"{nama} {aksi} alat {nama_alat}")
    print(f"{nama} {aksi} alat {nama_alat} jam {jam}")
    folder = CAPTURE_DIR / nama_alat
    folder.mkdir(exist_ok=True)
    out = folder / f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(out), shown)
    print("Tersimpan", out)


while True:
    cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        ok, frame = cap.read()
    if not ok:
        break
    n += 1

    now = time.time()
    if now - user_refresh > 0.5:
        cached_user = get_user()
        user_refresh = now
    user = cached_user
    label = f"Sesi: {user}" if user else "Sesi: belum konfirmasi (Cam1)"

    if n % YOLO_EVERY == 0:
        last_result = run_yolo(yolo, frame)
        tools = (
            [last_result.names[int(c)] for c in last_result.boxes.cls]
            if last_result.boxes
            else []
        )
        sekarang = set(tools)

        if not sekarang:
            had_empty = True
            kandidat = None
            stabil = 0
        elif had_empty:
            if sekarang == kandidat:
                stabil += 1
            else:
                kandidat = set(sekarang)
                stabil = 1

            if stabil >= STABIL_MIN and user and now - last_event >= SAVE_EVERY:
                borrowed = get_borrowed()
                shown_event = draw_detections(frame, last_result)
                cv2.putText(
                    shown_event,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                for tool in sorted(sekarang):
                    if tool in borrowed:
                        remove_borrowed(tool)
                        catat(user, "mengembalikan", tool, shown_event)
                    else:
                        add_borrowed(tool)
                        catat(user, "meminjam", tool, shown_event)
                last_event = now
                had_empty = False
                kandidat = None
                stabil = 0
        else:
            kandidat = None
            stabil = 0

    shown = draw_detections(frame, last_result)
    cv2.putText(shown, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Cam2 - Pinjam/Kembali", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
