import cv2
import numpy as np
from pathlib import Path

KILLFEED_REGION = (1400, 50, 500, 300)  # x, y, w, h for top-right killfeed area
TEMPLATE_DIR = Path("killfeed_templates")

def load_templates():
    templates = []
    for file in TEMPLATE_DIR.glob("*.png"):
        img = cv2.imread(str(file), 0)
        if img is not None:
            templates.append((file.name, img))
    return templates

def detect_killfeed(frame):
    x, y, w, h = KILLFEED_REGION
    cropped = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    templates = load_templates()

    match_count = 0
    for name, tmpl in templates:
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        if np.max(res) > 0.75:
            match_count += 1
    return match_count > 0

def score_killfeed_presence(video_path, sample_interval=30):
    cap = cv2.VideoCapture(str(video_path))
    frames_checked = 0
    killframes = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frames_checked % sample_interval == 0:
            if detect_killfeed(frame):
                killframes += 1
        frames_checked += 1

    cap.release()
    score = killframes / max(1, frames_checked // sample_interval)
    return round(score, 2)