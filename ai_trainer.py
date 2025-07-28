import json
import shutil
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("clip_score_log.json")
CONFIG_PATH = Path("highlight_config.json")
BACKUP_PATH = Path("backups")
BACKUP_PATH.mkdir(exist_ok=True)

def load_logs():
    if not LOG_PATH.exists():
        print("❌ No log file found.")
        return []
    with open(LOG_PATH, "r") as f:
        return json.load(f)

def load_config():
    if not CONFIG_PATH.exists():
        return {"min_words": 3, "visual_threshold": 500000}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def backup_config(config):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(BACKUP_PATH / f"highlight_config_backup_{timestamp}.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"🛡️ Backed up config as highlight_config_backup_{timestamp}.json")

def tune_config(logs, config):
    good_clips = [l for l in logs if l.get("kept")]
    bad_clips = [l for l in logs if not l.get("kept")]

    if not good_clips:
        print("⚠️ No good clips found to base tuning on.")
        return config

    avg_words = sum([c["words"] for c in good_clips]) / len(good_clips)
    avg_visual = sum([c["visual_score"] for c in good_clips]) / len(good_clips)

    new_config = config.copy()
    new_config["min_words"] = round(avg_words * 0.9)
    new_config["visual_threshold"] = round(avg_visual * 0.9)

    return new_config

def save_config(new_config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)
    print("✅ Updated highlight_config.json with tuned values.")

def main():
    logs = load_logs()
    config = load_config()
    backup_config(config)
    new_config = tune_config(logs, config)
    save_config(new_config)

if __name__ == "__main__":
    main()