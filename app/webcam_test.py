from pathlib import Path
from ultralytics import YOLO
import cv2

ROOT = Path(r"D:\DATA C\Projek\Detection Object - Yolo8")
MODEL = ROOT / "best.pt"

model = YOLO(str(MODEL))
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Laptop camera could not be opened")

print("Camera is running. Press Q in the video window to stop")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, conf=0.7, verbose=False)[0]
    names = [result.names[int(c)] for c in result.boxes.cls] if result.boxes else []
    if names:
        print("Detected:", ", ".join(names))

    cv2.imshow("Tools Detection", result.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()