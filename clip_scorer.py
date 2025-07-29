import os, json
from glob import glob
from pathlib import Path

KILLFEED_FOLDER = "DetectedKillfeeds"
SCORE_FOLDER = "ClipScores"
LOG_FILE = Path(SCORE_FOLDER) / "clip_scores.json"
os.makedirs(SCORE_FOLDER, exist_ok=True)

def score_all_clips():
    scores = []

    for jp in glob(os.path.join(KILLFEED_FOLDER, "*.json")):
        with open(jp, "r") as f:
            data = json.load(f)

        clip_name = data.get("video", os.path.basename(jp))

        # Compatibility: support both new and old formats
        if "killframe_count" in data:
            n = data["killframe_count"]
        elif "killfeed_timestamps" in data:
            n = len(data["killfeed_timestamps"])
        else:
            n = 0

        score = min(n * 25, 100)
        print(f"📊 {clip_name} → score {score} (events: {n})")
        scores.append({"clip": clip_name, "killfeed_events": n, "score": score})

    with open(LOG_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"✅ Scores saved to {LOG_FILE}")
