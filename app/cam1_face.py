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

app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(256, 256))
data = np.load(str(ROOT / "gallery_embeddings.npz"), allow_pickle=True)
GALLERY_NAMES = data["names"]
GALLERY_EMB = data["embeddings"]


def identify(face):
    sim = GALLERY_EMB @ face.normed_embedding
    i = int(np.argmax(sim))
    score = float(sim[i])
    if score < THRESH:
        return "unknown", score
    return str(GALLERY_NAMES[i]), score


cap = cv2.VideoCapture(CAM1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Kamera 1 (index {CAM1}) tidak bisa dibuka")

if CAM1 == CAM2:
    print(
        "Peringatan: CAM1 dan CAM2 index sama. Jalankan satu kamera saja agar tidak patah."
    )
print("Cam1 muka auto. C=clear sesi  Q=keluar")

n = 0
faces = []
kandidat = None
stabil = 0
had_empty = True
last_event = 0.0

while True:
    cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        ok, frame = cap.read()
    if not ok:
        break
    n += 1

    if n % FACE_EVERY == 0:
        faces = app.get(frame)

    shown = frame
    if faces:
        shown = app.draw_on(frame.copy(), faces)

    known = None
    for face in faces:
        nama, skor = identify(face)
        if nama != "unknown":
            known = nama
            cv2.putText(
                shown,
                f"{nama} {skor:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            break

    now = time.time()
    if not known:
        had_empty = True
        kandidat = None
        stabil = 0
    elif had_empty:
        if known == kandidat:
            stabil += 1
        else:
            kandidat = known
            stabil = 1

        if stabil >= STABIL_MIN and now - last_event >= SAVE_EVERY:
            # skip foto ulang kalau wajah masih masuk frame, tunggu keluar dari frame dan jeda berapa detik
            if get_user() != known:
                set_user(known)
                out = (
                    FACE_DIR / f"{known}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(str(out), shown)
                ucap(f"Baik {known}, silakan arahkan alat ke kamera dua")
                print("Sesi aktif:", known, "tersimpan", out)
            last_event = now
            had_empty = False
            kandidat = None
            stabil = 0
    else:
        kandidat = None
        stabil = 0

    sesi = get_user()
    label = f"Sesi: {sesi}" if sesi else "Sesi: menunggu wajah"
    cv2.putText(shown, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        shown,
        "Auto simpan | C clear | Q keluar",
        (20, shown.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Cam1 - Muka", shown)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("c"):
        clear_user()
        had_empty = True
        kandidat = None
        stabil = 0
        ucap("Sesi dikosongkan")
        print("Sesi dikosongkan")

cap.release()
cv2.destroyAllWindows()
