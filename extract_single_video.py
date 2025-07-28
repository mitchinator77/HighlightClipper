import sys
from pathlib import Path
import json
import random
import shutil
from visual_killfeed_detector import score_killfeed_presence

CONFIG_PATH = Path("highlight_config.json")
OUTPUT_DIR = Path("AutoClips")
OUTPUT_DIR.mkdir(exist_ok=True)

def dummy_clip(video_path):
    name = video_path.stem
    out_path = OUTPUT_DIR / f"{name}_clip.mp4"
    shutil.copy(video_path, out_path)
    return out_path

def main(video_file):
    video_path = Path(video_file)
    if not video_path.exists():
        print("Video not found.")
        return

    kill_score = score_killfeed_presence(video_path)
    kept = kill_score > 0.1

    clip_path = dummy_clip(video_path) if kept else None

    log = {
        "video": video_path.name,
        "kept": kept,
        "killfeed_score": kill_score
    }

    with open("clip_score_log.json", "a") as f:
        json.dump(log, f)
        f.write("\n")

    print(f"🔍 Killfeed score: {kill_score} | {'✅ Kept' if kept else '❌ Discarded'}")

if __name__ == "__main__":
    main(sys.argv[1])