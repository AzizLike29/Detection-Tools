import sys
import time
from collections import Counter
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
    get_borrowed_qty,
    get_user,
    is_qty_tool,
    log_event,
    remove_borrowed,
    run_yolo,
)
from tts_util import ucap

yolo_model = YOLO(str(ROOT / "best.pt"))

camera = cv2.VideoCapture(CAM2, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not camera.isOpened():
    raise RuntimeError(f"Kamera 2 (index {CAM2}) tidak bisa dibuka")

if CAM1 == CAM2:
    print(
        "Peringatan CAM1 dan CAM2 memakai index yang sama. Jalankan satu kamera saja agar tidak konflik."
    )
print(
    f"Cam2 pinjam/kembali (device={DEVICE}). Arahkan tools ke kamera. Tekan Q untuk keluar."
)

frame_count = 0
last_result = None
candidate_counts = None
stable_count = 0
was_empty = True
last_event_time = 0.0
cached_user = None
user_refresh_time = 0.0


ACTION_TEXT = {"borrow": "meminjam", "return": "mengembalikan"}


def record_event(person_name, action, tool, display_frame, qty=1):
    """Simpan foto, catat ke database, dan umumkan event pinjam atau kembali"""
    action_text = ACTION_TEXT[action]
    tool_label = tool.replace("_", " ")
    time_text = datetime.now().strftime("%H.%M")
    qty_text = f" ({qty})" if qty > 1 else ""
    ucap(f"{person_name} {action_text} alat {tool_label}{qty_text}")
    print(f"{person_name} {action_text} alat {tool_label}{qty_text} jam {time_text}")
    folder = CAPTURE_DIR / tool_label
    folder.mkdir(exist_ok=True)
    capture_path = folder / f"{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(capture_path), display_frame)
    print(f"Tersimpan {capture_path}")
    log_event(
        "cam2",
        action,
        user_name=person_name,
        yolo_class=tool,
        qty=qty,
        img_path=capture_path.relative_to(ROOT).as_posix(),
    )


while True:
    camera.grab()
    frame_ok, frame = camera.retrieve()
    if not frame_ok:
        frame_ok, frame = camera.read()
    if not frame_ok:
        break
    frame_count += 1

    current_time = time.time()
    if current_time - user_refresh_time > 0.5:
        cached_user = get_user()
        user_refresh_time = current_time
    session_user = cached_user
    session_label = (
        f"Sesi {session_user}" if session_user else "Sesi belum konfirmasi (Cam1)"
    )

    if frame_count % YOLO_EVERY == 0:
        last_result = run_yolo(yolo_model, frame)
        tools = (
            [last_result.names[int(c)] for c in last_result.boxes.cls]
            if last_result.boxes
            else []
        )
        detected_counts = dict(Counter(tools))

        if not detected_counts:
            was_empty = True
            candidate_counts = None
            stable_count = 0
        elif was_empty:
            if detected_counts == candidate_counts:
                stable_count += 1
            else:
                candidate_counts = dict(detected_counts)
                stable_count = 1

            if (
                stable_count >= STABIL_MIN
                and session_user
                and current_time - last_event_time >= SAVE_EVERY
            ):
                event_frame = draw_detections(frame, last_result)
                cv2.putText(
                    event_frame,
                    session_label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                for tool, count in sorted(detected_counts.items()):
                    if is_qty_tool(tool):
                        # Tools stok qty ikut jumlah box
                        if get_borrowed_qty(tool) > 0:
                            returned_count = remove_borrowed(tool, count)
                            if returned_count:
                                record_event(
                                    session_user,
                                    "return",
                                    tool,
                                    event_frame,
                                    returned_count,
                                )
                        else:
                            borrowed_count = add_borrowed(tool, count)
                            if borrowed_count:
                                record_event(
                                    session_user,
                                    "borrow",
                                    tool,
                                    event_frame,
                                    borrowed_count,
                                )
                    else:
                        # Tools lain tetap toggle pinjam/kembali
                        borrowed = get_borrowed()
                        if tool in borrowed:
                            remove_borrowed(tool)
                            record_event(
                                session_user, "return", tool, event_frame
                            )
                        else:
                            add_borrowed(tool)
                            record_event(session_user, "borrow", tool, event_frame)
                last_event_time = current_time
                was_empty = False
                candidate_counts = None
                stable_count = 0
        else:
            candidate_counts = None
            stable_count = 0

    display_frame = draw_detections(frame, last_result)
    cv2.putText(
        display_frame,
        session_label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Cam2 - Pinjam/Kembali", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
