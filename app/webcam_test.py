from datetime import datetime
from pathlib import Path
from threading import Thread
import time

import cv2
import numpy as np
import pygame
from gtts import gTTS
from insightface.app import FaceAnalysis
from ultralytics import YOLO

ROOT = Path(r"D:\DATA C\Projek\Detection Object - Yolo8")
SAVE = ROOT / "captures"
THRESH = 0.40
SAVE_EVERY = 3.0

SAVE.mkdir(exist_ok=True)
pygame.mixer.init()

yolo = YOLO(str(ROOT / "best.pt"))
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

data = np.load(str(ROOT / "gallery_embeddings.npz"), allow_pickle=True)
GALLERY_NAMES = data["names"]
GALLERY_EMB = data["embeddings"]
last_saved = {}
speaking = False


def identify(face):
    sim = GALLERY_EMB @ face.normed_embedding
    i = int(np.argmax(sim))
    score = float(sim[i])
    if score < THRESH:
        return "unknown", score
    return str(GALLERY_NAMES[i]), score


def ucap(teks):
    global speaking
    if speaking:
        return
    speaking = True

    def run():
        global speaking
        path = ROOT / f"_tts_{time.time_ns()}.mp3"
        try:
            gTTS(teks, lang="id").save(str(path))
            pygame.mixer.music.stop()
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            print("TTS ok:", teks)
        except Exception as e:
            print("TTS error:", e)
        finally:
            speaking = False
            path.unlink(missing_ok=True)

    Thread(target=run, daemon=True).start()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka")

print("Webcam jalan. Q untuk berhenti")

n = 0
alat = []
faces = []
last_result = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    if n % 3 == 0:
        last_result = yolo.predict(frame, conf=0.45, imgsz=640, verbose=False)[0]
        alat = [last_result.names[int(c)] for c in last_result.boxes.cls] if last_result.boxes else []

    if n % 10 == 0:
        faces = app.get(frame)

    shown = last_result.plot() if last_result is not None else frame.copy()
    if faces:
        shown = app.draw_on(shown, faces)

    now = time.time()
    for face in faces:
        nama, skor = identify(face)
        if nama == "unknown":
            continue
        if not alat:
            continue

        daftar = " dan ".join(a.replace("_", " ") for a in dict.fromkeys(alat))
        msg = f"{nama} membawa {daftar}"
        print(msg)
        ucap(msg)

        if now - last_saved.get(nama, 0) >= SAVE_EVERY:
            last_saved[nama] = now
            for tool in dict.fromkeys(alat):
                folder = SAVE / tool.replace("_", " ")
                folder.mkdir(exist_ok=True)
                out = folder / f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(out), shown)
                print("Tersimpan", out)

    cv2.imshow("Tools + Wajah", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()