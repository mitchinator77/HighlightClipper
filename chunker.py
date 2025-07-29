import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

CONVERTED_FOLDER = "Converted"
CHUNKS_FOLDER = "Chunks"
CHUNK_LENGTH = 120  # seconds
NUM_WORKERS = 4

os.makedirs(CHUNKS_FOLDER, exist_ok=True)

def chunk_video(video_path):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_template = os.path.join(CHUNKS_FOLDER, f"{filename}_chunk_%03d.mp4")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c", "copy", "-map", "0",
        "-segment_time", str(CHUNK_LENGTH),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_template
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def chunk_all_videos():
    mp4_files = glob.glob(os.path.join(CONVERTED_FOLDER, "*.mp4"))
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(chunk_video, mp4_files)