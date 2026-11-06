import time
from datetime import datetime
from pathlib import Path
from threading import Thread

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
STABIL_MIN = 3

SAVE.mkdir(exist_ok=True)
pygame.mixer.init()

yolo = YOLO(str(ROOT / "best.pt"))
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

data = np.load(str(ROOT / "gallery_embeddings.npz"), allow_pickle=True)
GALLERY_NAMES = data["names"]
GALLERY_EMB = data["embeddings"]
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


def catat(nama, aksi, tool, jam, shown):
    nama_alat = tool.replace("_", " ")
    ucap(f"{nama} {aksi} alat {nama_alat}")
    print(f"{nama} {aksi} alat {nama_alat} jam {jam}")
    folder = SAVE / nama_alat
    folder.mkdir(exist_ok=True)
    out = folder / f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(out), shown)
    print("Tersimpan", out)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka")

print("Webcam jalan. Q untuk berhenti")

n = 0
alat = []
faces = []
last_result = None
alat_di_rak = None
kandidat = None
stabil = 0
last_event = {}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    if n % 10 == 0:
        faces = app.get(frame)

    shown = frame.copy()
    if last_result is not None:
        shown = last_result.plot()
    if faces:
        shown = app.draw_on(shown, faces)

    if n % 3 == 0:
        last_result = yolo.predict(frame, conf=0.40, imgsz=1280, iou=0.5, verbose=False)[0]
        alat = (
            [last_result.names[int(c)] for c in last_result.boxes.cls]
            if last_result.boxes
            else []
        )
        shown = last_result.plot()
        if faces:
            shown = app.draw_on(shown, faces)

        sekarang = set(alat)
        if alat_di_rak is None:
            alat_di_rak = sekarang
            print("Baseline rak:", sorted(alat_di_rak))
        elif sekarang == alat_di_rak:
            kandidat = None
            stabil = 0
        elif sekarang == kandidat:
            stabil += 1
            if stabil >= STABIL_MIN:
                hilang = alat_di_rak - sekarang
                muncul = sekarang - alat_di_rak
                nama = None
                for face in faces:
                    n_wajah, _ = identify(face)
                    if n_wajah != "unknown":
                        nama = n_wajah
                        break
                now = time.time()
                jam = datetime.now().strftime("%H.%M")
                if nama and (hilang or muncul):
                    if now - last_event.get(nama, 0) >= SAVE_EVERY:
                        last_event[nama] = now
                        for tool in sorted(hilang):
                            catat(nama, "meminjam", tool, jam, shown)
                        for tool in sorted(muncul):
                            catat(nama, "mengembalikan", tool, jam, shown)
                alat_di_rak = sekarang
                kandidat = None
                stabil = 0
                print("Rak sekarang:", sorted(alat_di_rak))
        else:
            kandidat = sekarang
            stabil = 1

    cv2.imshow("Tools + Wajah", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
