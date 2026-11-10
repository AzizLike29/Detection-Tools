import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
py = sys.executable

procs = [
    subprocess.Popen([py, str(APP / "cam1_face.py")], cwd=str(APP)),
    subprocess.Popen([py, str(APP / "cam2_borrow.py")], cwd=str(APP)),
]

print("Cam1 + Cam2 dijalankan (Cam3 index sama dengan Cam2; jalankan terpisah).")
print("Tutup tiap jendela atau Ctrl+C di sini.")
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
