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
    """Inisialisasi pygame mixer sekali saja"""
    global _pygame_ready
    if not _pygame_ready:
        pygame.mixer.init()
        _pygame_ready = True


def ucap(text):
    """Ucapkan teks Indonesia lewat TTS di thread terpisah"""
    global speaking
    if speaking:
        return
    speaking = True

    def speak_worker():
        global speaking
        _ensure_mixer()
        audio_path = ROOT / f"_tts_{time.time_ns()}.mp3"
        try:
            gTTS(text, lang="id").save(str(audio_path))
            pygame.mixer.music.stop()
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            print(f"TTS ok {text}")
        except Exception as error:
            print(f"TTS error {error}")
        finally:
            speaking = False
            audio_path.unlink(missing_ok=True)

    Thread(target=speak_worker, daemon=True).start()
