import sys
import time
from pathlib import Path
from threading import Thread

import pygame
from gtts import gTTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_state import ROOT

speaking = False
_pygame_ready = False


def _ensure_mixer():
    global _pygame_ready
    if not _pygame_ready:
        pygame.mixer.init()
        _pygame_ready = True


def ucap(teks):
    global speaking
    if speaking:
        return
    speaking = True

    def run():
        global speaking
        _ensure_mixer()
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
