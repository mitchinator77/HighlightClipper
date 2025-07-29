
import json
from pathlib import Path
from visual_killfeed_detector import score_killfeed_presence

CHUNKS_DIR = Path("Chunks")
SCORES_PATH = Path("ClipScores/clip_scores.json")

def score_all_chunks():
    scores = {}
    for video_file in CHUNKS_DIR.glob("*.mp4"):
        result = score_killfeed_presence(video_file)
        scores[video_file.name] = result

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=4)

    print("✅ Scores saved to", SCORES_PATH)

if __name__ == "__main__":
    score_all_chunks()
