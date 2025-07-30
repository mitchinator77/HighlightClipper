import json
import os
from audio_model_trainer import train_headshot_audio_model

STATUS_FILE = "trainer_status.json"
TRAINER_LOG = "trainer_log.txt"
CLIP_THRESHOLD = 3
TRAINING_DIR = "audiosamples/headshot"  # You can create this folder later

def load_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"clips_since_last_train": 0}

def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

def log_training_event(message):
    with open(TRAINER_LOG, "a") as f:
        f.write(message + "\n")

def check_and_trigger_trainer():
    status = load_status()
    status["clips_since_last_train"] += 1

    if status["clips_since_last_train"] >= CLIP_THRESHOLD:
        if os.path.isdir(TRAINING_DIR):
            try:
                train_headshot_audio_model(TRAINING_DIR)
                log_training_event("✅ Training complete and model saved.")
            except Exception as e:
                log_training_event(f"⚠️ Training failed: {e}")
        else:
            log_training_event(f"⚠️ Training directory not found: {TRAINING_DIR}")
        status["clips_since_last_train"] = 0

    save_status(status)
