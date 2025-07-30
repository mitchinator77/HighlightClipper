# scoring_logger.py

import json
from typing import List, Dict, Tuple

def log_scores_to_file(scored_moments: List[Tuple[float, float]], events: Dict[str, List[float]], path: str):
    """
    Saves a JSON log of highlight scores with associated signals.
    """
    log = []
    for timestamp, score in scored_moments:
        contributing = {
            "timestamp": round(timestamp, 3),
            "score": score,
            "visual_nearby": [t for t in events.get("visual", []) if abs(t - timestamp) <= 2],
            "audio_nearby": [t for t in events.get("audio", []) if abs(t - timestamp) <= 2],
            "headshot_nearby": [t for t in events.get("headshot", []) if abs(t - timestamp) <= 2],
        }
        log.append(contributing)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
        
def log_clip_decision(timestamp: float, score: float, label: str, reason: str, path: str):
    """
    Appends a clip decision (keep/discard) with score and reason to a log file.
    """
    entry = {
        "timestamp": round(timestamp, 3),
        "score": round(score, 2),
        "label": label,
        "decision_reason": reason
    }

    # Load existing log
    try:
        with open(path, "r") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    log.append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
