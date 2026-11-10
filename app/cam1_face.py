import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import CAM1, FACE_DIR, ROOT, THRESH, set_user, clear_user
from tts_util import ucap

app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))
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

print("Cam1 muka. Y=konfirmasi  N=batal  C=clear sesi  Q=keluar")

n = 0
faces = []
pending = None
asked = False
last_ask = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    if n % 8 == 0:
        faces = app.get(frame)

    shown = frame.copy()
    if faces:
        shown = app.draw_on(shown, faces)

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

    if known:
        if pending != known:
            pending = known
            asked = False
        if not asked and time.time() - last_ask > 4.0:
            ucap(f"Apakah benar ini muka {known}?")
            asked = True
            last_ask = time.time()
            print("Menunggu konfirmasi Y/N untuk", known)
    else:
        pending = None
        asked = False

    cv2.putText(
        shown,
        "Y konfirmasi | N batal | C clear | Q keluar",
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
        pending = None
        asked = False
        ucap("Sesi dikosongkan")
        print("Sesi dikosongkan")
    if key == ord("y") and pending:
        set_user(pending)
        out = FACE_DIR / f"{pending}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(out), shown)
        ucap(f"Baik {pending}, silakan arahkan alat ke kamera dua")
        print("Sesi aktif:", pending, "tersimpan", out)
        asked = False
        pending = None
    if key == ord("n") and pending:
        ucap("Baik, silakan scan ulang")
        print("Ditolak:", pending)
        pending = None
        asked = False

cap.release()
cv2.destroyAllWindows()
