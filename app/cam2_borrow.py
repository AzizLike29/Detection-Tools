import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import (
    CAM2,
    CAPTURE_DIR,
    IMGSZ,
    ROOT,
    SAVE_EVERY,
    STABIL_MIN,
    YOLO_CONF,
    add_borrowed,
    get_borrowed,
    get_user,
    remove_borrowed,
)
from tts_util import ucap

yolo = YOLO(str(ROOT / "best.pt"))

cap = cv2.VideoCapture(CAM2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Kamera 2 (index {CAM2}) tidak bisa dibuka")

print("Cam2 pinjam/kembali. Arahkan tools ke kamera. Q keluar")

n = 0
last_result = None
kandidat = None
stabil = 0
last_empty = 0.0
had_empty = True
last_event = 0.0


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
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    shown = frame.copy()
    if last_result is not None:
        shown = last_result.plot()

    user = get_user()
    label = f"Sesi: {user}" if user else "Sesi: belum konfirmasi (Cam1)"
    cv2.putText(shown, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if n % 3 == 0:
        last_result = yolo.predict(frame, conf=YOLO_CONF, imgsz=IMGSZ, verbose=False)[0]
        shown = last_result.plot()
        cv2.putText(
            shown, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

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
            last_empty = time.time()
        elif had_empty:
            if sekarang == kandidat:
                stabil += 1
            else:
                kandidat = set(sekarang)
                stabil = 1

            if stabil >= STABIL_MIN and user and time.time() - last_event >= SAVE_EVERY:
                borrowed = get_borrowed()
                for tool in sorted(sekarang):
                    if tool in borrowed:
                        remove_borrowed(tool)
                        catat(user, "mengembalikan", tool, shown)
                    else:
                        add_borrowed(tool)
                        catat(user, "meminjam", tool, shown)
                last_event = time.time()
                had_empty = False
                kandidat = None
                stabil = 0
        else:
            kandidat = None
            stabil = 0

    cv2.imshow("Cam2 - Pinjam/Kembali", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
