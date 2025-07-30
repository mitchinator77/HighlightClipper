import os
import json
from datetime import datetime

LOG_DIR = "Logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(video_name, event_type, timestamp, confidence=None, tags=None):
    log_file = os.path.join(LOG_DIR, f"{video_name}_log.json")

    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "confidence": confidence,
        "tags": tags,
        "datetime": datetime.utcnow().isoformat()
    }

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)