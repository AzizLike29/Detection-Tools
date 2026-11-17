import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import (
    CAM2,
    CAM3,
    DEVICE,
    INVENTORY_EVERY,
    INVENTORY_EVERY_FRAMES,
    ROOT,
    STABIL_MIN,
    draw_detections,
    get_borrowed,
    run_yolo,
)
from tts_util import ucap

yolo_model = YOLO(str(ROOT / "best.pt"))

camera = cv2.VideoCapture(CAM3, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not camera.isOpened():
    raise RuntimeError(f"Kamera 3 (index {CAM3}) tidak bisa dibuka")

if CAM3 == CAM2:
    print(
        "Peringatan CAM3 dan CAM2 memakai index yang sama. Jalankan satu kamera saja agar tidak konflik."
    )
print(
    f"Cam3 inventori (device={DEVICE}). Baseline diambil otomatis. Tekan Q untuk keluar."
)

frame_count = 0
last_result = None
baseline = None
candidate_tools = None
stable_count = 0
last_check_time = 0.0
alerted = set()

while True:
    camera.grab()
    frame_ok, frame = camera.retrieve()
    if not frame_ok:
        frame_ok, frame = camera.read()
    if not frame_ok:
        break
    frame_count += 1

    if frame_count % INVENTORY_EVERY_FRAMES == 0:
        last_result = run_yolo(yolo_model, frame)
        tools = (
            [last_result.names[int(c)] for c in last_result.boxes.cls]
            if last_result.boxes
            else []
        )
        detected_tools = set(tools)

        if baseline is None:
            # Ambil baseline setelah deteksi stabil
            if detected_tools == candidate_tools:
                stable_count += 1
            else:
                candidate_tools = set(detected_tools)
                stable_count = 1
            if stable_count >= STABIL_MIN:
                baseline = set(detected_tools)
                candidate_tools = None
                stable_count = 0
                print(f"Baseline inventori {sorted(baseline)}")
                ucap("Baseline inventori siap")
        elif time.time() - last_check_time >= INVENTORY_EVERY:
            last_check_time = time.time()
            borrowed = get_borrowed()
            missing_tools = baseline - detected_tools
            for tool in sorted(missing_tools):
                if tool in borrowed:
                    continue
                if tool in alerted:
                    continue
                tool_label = tool.replace("_", " ")
                ucap(f"Peringatan, alat {tool_label} tidak ada di ruangan")
                print(f"Hilang {tool}")
                alerted.add(tool)
            returned_tools = alerted & detected_tools
            for tool in sorted(returned_tools):
                alerted.discard(tool)
                print(f"Kembali terdeteksi {tool}")

    display_frame = draw_detections(frame, last_result)
    status_label = (
        f"Baseline {len(baseline) if baseline else 0} | Alert {len(alerted)}"
    )
    cv2.putText(
        display_frame,
        status_label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )
    cv2.imshow("Cam3 - Inventori", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
