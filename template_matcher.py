import cv2
import os
import numpy as np
from pathlib import Path

TEMPLATE_DIRS = [
    "templates/killfeed",
    "templates/hud_elements",
    "templates/buyphase_banner",
    "templates/scoreboard"
]

def load_templates():
    templates = []
    for folder in TEMPLATE_DIRS:
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if fname.endswith(".png") or fname.endswith(".jpg"):
                templates.append((folder.split("/")[-1], cv2.imread(path, 0)))
    return templates

def sample_frames(chunk_path, num_frames=10):
    cap = cv2.VideoCapture(chunk_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in np.linspace(0, total - 1, num=num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames

def match_templates_on_chunk(chunk_path, threshold=0.75):
    templates = load_templates()
    frames = sample_frames(chunk_path)
    match_scores = {cat: [] for cat, _ in templates}

    for frame in frames:
        for category, tmpl in templates:
            h, w = tmpl.shape
            res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, _, _ = cv2.minMaxLoc(res)
            match_scores[category].append(max_val)

    category_scores = {k: max(v) if v else 0.0 for k, v in match_scores.items()}
    highest = max(category_scores.values())

    return {
        "label": "valorant" if highest >= threshold else "other",
        "confidence": highest,
        "scores": category_scores
    }
