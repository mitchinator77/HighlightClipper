import os
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import shutil

load_dotenv()

VIDEO_DIR = Path("SourceVideos")
CLIP_OUTPUT_DIR = Path("AutoClips")
CONFIG_PATH = Path("highlight_config.json")
TRAINER_PATH = Path("ai_trainer.py")
CONVERTED_DIR = Path("ConvertedVideos")
CONVERTED_DIR.mkdir(exist_ok=True)

def run_trainer():
    print("\n🧠 Running AI trainer module to evolve pipeline...")
    subprocess.run(["python", "ai_trainer.py"])

def convert_to_mp4(video_path):
    if video_path.suffix.lower() == ".mp4":
        return video_path  # No conversion needed
    output_path = CONVERTED_DIR / (video_path.stem + ".mp4")
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c", "copy", str(output_path)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    video_path.unlink()  # Delete original .mkv
    print(f"🔁 Converted and deleted original: {video_path.name}")
    return output_path

def process_video(video_path, count):
    print(f"🎥 Processing: {video_path.name}")
    mp4_path = convert_to_mp4(video_path)
    subprocess.run(["python", "extract_single_video.py", str(mp4_path)])
    if (count + 1) % 3 == 0:
        run_trainer()

def main():
    all_videos = list(VIDEO_DIR.glob("*.mkv")) + list(VIDEO_DIR.glob("*.mp4"))
    if not all_videos:
        print("❌ No videos found in SourceVideos.")
        return

    for idx, video in enumerate(all_videos):
        process_video(video, idx)

    print("\n✅ All videos processed.")

if __name__ == "__main__":
    main()