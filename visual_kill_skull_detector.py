import cv2
import numpy as np
from pathlib import Path

TEMPLATE_PATH = Path("valokillskull.png")

def detect_kill_skull(video_path, threshold=0.6):
    if not TEMPLATE_PATH.exists():
        print("❌ Kill skull template missing")
        return 0.0

    template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("❌ Failed to load kill skull template")
        return 0.0

    cap = cv2.VideoCapture(str(video_path))
    matches = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        roi = gray[int(h * 0.78):int(h * 0.92), int(w * 0.42):int(w * 0.58)]

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            matches += 1

    cap.release()
    confidence = matches / total_frames if total_frames else 0
    return round(confidence, 2)