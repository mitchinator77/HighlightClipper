"""Improved killfeed detection with ROI cropping and frame skipping."""
import cv2
import numpy as np
from pathlib import Path

KILLFEED_REGION = (1400, 50, 500, 300)
TEMPLATE_DIR = Path("killfeed_templates")
THRESHOLD = 0.6
FRAME_SKIP = 30

def load_templates():
    templates = []
    for file in TEMPLATE_DIR.glob("*.png"):
        img = cv2.imread(str(file), 0)
        if img is not None:
            templates.append(img)
    return templates

def detect_killfeed(frame, templates):
    x, y, w, h = KILLFEED_REGION
    cropped = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    for tmpl in templates:
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        if np.max(res) >= THRESHOLD:
            return True
    return False

def score_killfeed_presence(video_path):
    cap = cv2.VideoCapture(str(video_path))
    templates = load_templates()
    frames_checked = 0
    hit_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frames_checked % FRAME_SKIP == 0:
            if detect_killfeed(frame, templates):
                hit_count += 1
        frames_checked += 1

    cap.release()
    return min(hit_count / max(frames_checked // FRAME_SKIP, 1), 1.0)