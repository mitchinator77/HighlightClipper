import cv2
import numpy as np
from pathlib import Path

# Region of Interest (ROI) — top-right corner (killfeed area in Valorant)
KILLFEED_REGION = (1400, 50, 500, 300)  # (x, y, width, height)

def detect_killfeed(video_path, templates=None, threshold=0.75):
    timestamps = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return 0, []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        x, y, w, h = KILLFEED_REGION
        roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        for tmpl in templates:
            if tmpl.shape[0] > gray_roi.shape[0] or tmpl.shape[1] > gray_roi.shape[1]:
                continue  # Skip oversized templates

            result = cv2.matchTemplate(gray_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val >= threshold:
                timestamp = frame_number / fps
                timestamps.append(round(timestamp, 2))
                break  # Stop checking more templates for this frame

    cap.release()

    score = len(timestamps)
    return score, sorted(set(timestamps))


def load_templates(path):
    template_dir = Path(path)
    if not template_dir.exists():
        print(f"⚠️ Template directory {path} not found.")
        return []

    templates = []
    for file in template_dir.glob("*.png"):
        template = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates.append(template)
    return templates


def detect_killfeed_events(video_path, templates):
    return detect_killfeed(video_path, templates)
