import json
from pathlib import Path

CONFIG_PATH = Path("highlight_config.json")
CLIP_LOG_PATH = Path("clip_score_log.json")

def tune_config():
    if not CONFIG_PATH.exists() or not CLIP_LOG_PATH.exists():
        print("⚠️ Required files not found.")
        return

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(CLIP_LOG_PATH, "r") as f:
        logs = json.load(f)

    total = len(logs)
    bad = sum(1 for item in logs if item["score"] < 0.3)

    if total == 0:
        return

    bad_ratio = bad / total

    if bad_ratio > 0.4:
        print(f"🔧 Too many low-quality clips ({bad_ratio:.0%}), increasing filter strictness.")
        config["min_words"] = min(config.get("min_words", 2) + 1, 10)
        config["visual_threshold"] = config.get("visual_threshold", 500000) + 50000
    elif bad_ratio < 0.1:
        print(f"✅ Good quality clip rate ({bad_ratio:.0%}), relaxing filter slightly.")
        config["min_words"] = max(config.get("min_words", 2) - 1, 2)

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("🧠 Updated config based on clip logs.")

if __name__ == "__main__":
    tune_config()