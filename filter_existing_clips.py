import os
import json
import shutil
from pathlib import Path
import whisper
import cv2
import numpy as np
from visual_event_detector import has_visual_event

CONFIG_PATH = Path("highlight_config.json")
LOG_PATH = Path("clip_score_log.json")
CLIPS_FOLDER = Path("AutoClips")
MIN_CLIP_LENGTH = 3  # in seconds

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"min_words": 3, "visual_threshold": 500000}

def transcribe_clip(clip_path, model):
    result = model.transcribe(str(clip_path), verbose=False)
    text = result["text"]
    word_count = len(text.strip().split())
    return text, word_count

def get_clip_duration(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0
    cap.release()
    return duration

def log_clip(score_entry):
    logs = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    logs.append(score_entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

def main():
    config = load_config()
    model = whisper.load_model("base")
    clips = list(CLIPS_FOLDER.glob("*.mp4"))

    for clip in clips:
        duration = get_clip_duration(clip)
        if duration < MIN_CLIP_LENGTH:
            print(f"🗑 Deleting {clip.name} (too short: {duration:.1f}s)")
            clip.unlink()
            continue

        text, words = transcribe_clip(clip, model)
        visual_score = 0
        mid_time = duration / 2
        if has_visual_event(clip, mid_time):
            visual_score = 600000
        else:
            visual_score = 200000

        passed = words >= config["min_words"] and visual_score >= config["visual_threshold"]

        log_clip({
            "clip": clip.name,
            "duration": duration,
            "score": (words / 10 + visual_score / 1000000) / 2,
            "words": words,
            "visual_score": visual_score,
            "kept": passed
        })

        if not passed:
            print(f"🗑 Deleting {clip.name} (score too low: {words} words, visual: {visual_score})")
            clip.unlink()
        else:
            print(f"✅ Keeping {clip.name}")

if __name__ == "__main__":
    main()