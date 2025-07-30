import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

NUM_WORKERS = 4

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
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
                str(output_file)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {mkv_file.name}: {e}")
