import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import (
    CAM1,
    CAM2,
    FACE_DIR,
    FACE_EVERY,
    ROOT,
    SAVE_EVERY,
    STABIL_MIN,
    THRESH,
    clear_user,
    get_user,
    set_user,
)
from tts_util import ucap

face_analyzer = FaceAnalysis(name="buffalo_s")
face_analyzer.prepare(ctx_id=-1, det_size=(256, 256))
gallery_data = np.load(str(ROOT / "gallery_embeddings.npz"), allow_pickle=True)
GALLERY_NAMES = gallery_data["names"]
GALLERY_EMBEDDINGS = gallery_data["embeddings"]


def identify_face(face):
    """Cocokkan satu wajah ke gallery lalu kembalikan nama dan skor"""
    similarities = GALLERY_EMBEDDINGS @ face.normed_embedding
    best_index = int(np.argmax(similarities))
    match_score = float(similarities[best_index])
    if match_score < THRESH:
        return "unknown", match_score
    return str(GALLERY_NAMES[best_index]), match_score


camera = cv2.VideoCapture(CAM1, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not camera.isOpened():
    raise RuntimeError(f"Kamera 1 (index {CAM1}) tidak bisa dibuka")

if CAM1 == CAM2:
    print(
        "Peringatan CAM1 dan CAM2 memakai index yang sama. Jalankan satu kamera saja agar tidak konflik."
    )
print("Cam1 muka otomatis. Tekan C untuk clear sesi. Tekan Q untuk keluar.")

frame_count = 0
faces = []
candidate_name = None
stable_count = 0
was_empty = True
last_event_time = 0.0

while True:
    camera.grab()
    frame_ok, frame = camera.retrieve()
    if not frame_ok:
        frame_ok, frame = camera.read()
    if not frame_ok:
        break
    frame_count += 1

    if frame_count % FACE_EVERY == 0:
        faces = face_analyzer.get(frame)

    display_frame = frame
    if faces:
        display_frame = face_analyzer.draw_on(frame.copy(), faces)

    recognized_name = None
    for face in faces:
        person_name, match_score = identify_face(face)
        if person_name != "unknown":
            recognized_name = person_name
            cv2.putText(
                display_frame,
                f"{person_name} {match_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            break

    current_time = time.time()
    if not recognized_name:
        was_empty = True
        candidate_name = None
        stable_count = 0
    elif was_empty:
        if recognized_name == candidate_name:
            stable_count += 1
        else:
            candidate_name = recognized_name
            stable_count = 1

        if stable_count >= STABIL_MIN and current_time - last_event_time >= SAVE_EVERY:
            # Tunggu wajah keluar dari frame sebelum simpan lagi
            if get_user() != recognized_name:
                set_user(recognized_name)
                capture_path = (
                    FACE_DIR
                    / f"{recognized_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(str(capture_path), display_frame)
                ucap(f"Baik {recognized_name}, silakan arahkan alat ke kamera dua")
                print(f"Sesi aktif {recognized_name} tersimpan {capture_path}")
            last_event_time = current_time
            was_empty = False
            candidate_name = None
            stable_count = 0
    else:
        candidate_name = None
        stable_count = 0

    session_user = get_user()
    session_label = f"Sesi: {session_user}" if session_user else "Sesi: menunggu wajah"
    cv2.putText(
        display_frame,
        session_label,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        "Auto simpan || C clear || Q keluar",
        (20, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Cam1 - Muka", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("c"):
        clear_user()
        was_empty = True
        candidate_name = None
        stable_count = 0
        ucap("Sesi dikosongkan")
        print("Sesi dikosongkan")

camera.release()
cv2.destroyAllWindows()
