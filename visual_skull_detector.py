import cv2
import numpy as np
from pathlib import Path

def load_skull_templates(template_dir="skull_templates"):
    template_dir = Path(template_dir)
    templates = []

    for template_path in template_dir.glob("*.png"):
        img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append(img)
    
    return templates

def detect_skull_kill(video_path, templates, threshold=0.7):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    max_confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue  # Skip some frames for speed

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(h*0.75):, int(w*0.3):int(w*0.7)]  # Bottom-center region

        for template in templates:
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > max_confidence:
                max_confidence = max_val

            if max_val >= threshold:
                cap.release()
                return round(max_val, 2)

    cap.release()
    return round(max_confidence, 2)
