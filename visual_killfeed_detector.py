import cv2
import numpy as np
from pathlib import Path

KILLFEED_REGION = (1400, 50, 500, 300)  # (x, y, w, h)

def detect_killfeed(video_path, templates, threshold=0.6, frame_skip=5):
    cap = cv2.VideoCapture(str(video_path))  # Ensure video_path is string
    hit_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            x, y, w, h = KILLFEED_REGION
            cropped = frame[y:y+h, x:x+w]  # This is safe now

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            for template in templates:
                if template is None:
                    continue
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    hit_frames += 1
                    break

        frame_idx += 1

    cap.release()
    return hit_frames > 0  # Or return hit_frames for score granularity

def load_templates(template_dir):
    templates = []
    for p in Path(template_dir).glob("*.png"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append(img)
    return templates

