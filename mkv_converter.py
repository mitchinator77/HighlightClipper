import os
import glob
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path  # make sure this is at the top of your file
SOURCE_FOLDER = "SourceVideos"
CONVERTED_FOLDER = "Converted"
NUM_WORKERS = 4

os.makedirs(CONVERTED_FOLDER, exist_ok=True)

def convert_video(video_file):
    filename = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(CONVERTED_FOLDER, f"{filename}.mp4")

    cmd = ["ffmpeg", "-y", "-i", video_file, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", output_path]
   # subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from pathlib import Path  # make sure this is at the top of your file

def convert_all_mkv(source_dir="SourceVideos"):
    mkv_files = sorted(Path(source_dir).glob("*.mkv"))

    for mkv_file in mkv_files:
        output_file = mkv_file.with_suffix(".mp4")

        if output_file.exists():
            print(f"🔁 Skipping already converted: {output_file.name}")
            continue

        print(f"🎞️ Converting: {mkv_file.name} → {output_file.name}")
        try:
            subprocess.run([
                "ffmpeg",
                "-i", str(mkv_file),
                "-c:v", "copy",  # video codec: stream copy (no re-encode)
                "-c:a", "aac",   # audio codec: convert to AAC
                "-b:a", "192k",
                "-y",            # overwrite output file if it exists
                str(output_file)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {mkv_file.name}: {e}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(convert_video, input_files)