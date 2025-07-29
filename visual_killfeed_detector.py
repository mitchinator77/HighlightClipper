
import cv2
import numpy as np
from pathlib import Path
import sys
import json

# --- Settings ---
KILLFEED_REGION = (1400, 50, 500, 300)
TEMPLATE_DIR = Path("killfeed_templates")
THRESHOLD = 0.6
FRAME_SKIP = 30
CLIP_THRESHOLD = 0.25

def load_templates():
    templates = []
    for file in TEMPLATE_DIR.glob("*.png"):
        img = cv2.imread(str(file), 0)
        if img is not None:
            templates.append((file.name, img))
    return templates

def detect_killfeed(frame, templates):
    x, y, w, h = KILLFEED_REGION
    cropped = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    for name, tmpl in templates:
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        score = np.max(res)
        if score >= THRESHOLD:
            return True
    return False

def score_killfeed_presence(video_path):
    templates = load_templates()
    cap = cv2.VideoCapture(str(video_path))
    frames_checked = 0
    killframes = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frames_checked % FRAME_SKIP == 0:
            if detect_killfeed(frame, templates):
                killframes += 1
        frames_checked += 1

    cap.release()
    confidence = round(killframes / max(1, frames_checked // FRAME_SKIP), 2)
    is_clipworthy = confidence >= CLIP_THRESHOLD
    print(f"📊 FINAL SCORE: {killframes} hit frames → {confidence} confidence")
    print(f"🔥 Clip-worthy: {'YES' if is_clipworthy else 'NO'}")

    return {
        "score": int(is_clipworthy),
        "confidence": confidence,
        "events": killframes
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visual_killfeed_detector.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    score_killfeed_presence(video_path)
