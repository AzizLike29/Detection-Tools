import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
python_executable = sys.executable

processes = [
    subprocess.Popen([python_executable, str(APP_DIR / "cam1_face.py")], cwd=str(APP_DIR)),
    subprocess.Popen(
        [python_executable, str(APP_DIR / "cam2_borrow.py")], cwd=str(APP_DIR)
    ),
]

print("Cam1 dan Cam2 dijalankan. Cam3 index sama dengan Cam2 jadi jalankan terpisah.")
print("Tutup tiap jendela atau tekan Ctrl+C di sini.")
try:
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    for process in processes:
        process.terminate()
