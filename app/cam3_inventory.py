import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import (
    CAM3,
    IMGSZ,
    INVENTORY_EVERY,
    ROOT,
    STABIL_MIN,
    YOLO_CONF,
    get_borrowed,
)
from tts_util import ucap

yolo = YOLO(str(ROOT / "best.pt"))

cap = cv2.VideoCapture(CAM3, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Kamera 3 (index {CAM3}) tidak bisa dibuka")

print("Cam3 inventori. Baseline diambil otomatis. Q keluar")

n = 0
last_result = None
baseline = None
kandidat = None
stabil = 0
last_check = 0.0
alerted = set()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    shown = frame.copy()
    if last_result is not None:
        shown = last_result.plot()

    if n % 5 == 0:
        last_result = yolo.predict(frame, conf=YOLO_CONF, imgsz=IMGSZ, verbose=False)[0]
        shown = last_result.plot()
        tools = (
            [last_result.names[int(c)] for c in last_result.boxes.cls]
            if last_result.boxes
            else []
        )
        sekarang = set(tools)

        if baseline is None:
            if sekarang == kandidat:
                stabil += 1
            else:
                kandidat = set(sekarang)
                stabil = 1
            if stabil >= STABIL_MIN:
                baseline = set(sekarang)
                kandidat = None
                stabil = 0
                print("Baseline inventori:", sorted(baseline))
                ucap("Baseline inventori siap")
        elif time.time() - last_check >= INVENTORY_EVERY:
            last_check = time.time()
            borrowed = get_borrowed()
            hilang = baseline - sekarang
            for tool in sorted(hilang):
                if tool in borrowed:
                    continue
                if tool in alerted:
                    continue
                nama_alat = tool.replace("_", " ")
                ucap(f"Peringatan, alat {nama_alat} tidak ada di ruangan")
                print("HILANG:", tool)
                alerted.add(tool)
            kembali = alerted & sekarang
            for tool in sorted(kembali):
                alerted.discard(tool)
                print("Kembali terdeteksi:", tool)

    status = f"Baseline: {len(baseline) if baseline else 0} | Alert: {len(alerted)}"
    cv2.putText(
        shown, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
    )
    cv2.imshow("Cam3 - Inventori", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
