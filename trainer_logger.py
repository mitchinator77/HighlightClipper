import os
import json
from datetime import datetime

TRAINER_LOG_DIR = "TrainerLogs"
os.makedirs(TRAINER_LOG_DIR, exist_ok=True)

def compute_viewability_score(transcript: str, confidence: float, tags: list[str]) -> int:
    score = 0
    if "headshot" in tags:
        score += 2
    if confidence > 0.75:
        score += 1
    if transcript and any(word in transcript.lower() for word in ["let's go", "nice shot", "got him", "cracked", "insane"]):
        score += 2
    return min(score, 5)

def log_clip_feedback(clip_name, predicted_tags, confidence, transcript="", manual_approved=None):
    score = compute_viewability_score(transcript, confidence, predicted_tags)
    log_entry = {
        "clip": clip_name,
        "predicted_tags": predicted_tags,
        "confidence": confidence,
        "transcript": transcript,
        "ai_score": score,
        "manual_approved": manual_approved,
        "datetime": datetime.utcnow().isoformat()
    }

    log_file = os.path.join(TRAINER_LOG_DIR, f"{clip_name}.json")
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)