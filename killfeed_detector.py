
import cv2
from pathlib import Path

def detect_killfeed(video_path, templates=None):
    # Dummy implementation (replace with actual logic)
    score = 2
    timestamps = [22.0, 40.2]
    return score, timestamps

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
    # Unified interface expected by run_all.py
    return detect_killfeed(video_path, templates)
