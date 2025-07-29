import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

SOURCE_FOLDER = "SourceVideos"
CONVERTED_FOLDER = "Converted"
NUM_WORKERS = 4

os.makedirs(CONVERTED_FOLDER, exist_ok=True)

def convert_video(video_file):
    filename = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(CONVERTED_FOLDER, f"{filename}.mp4")

    cmd = ["ffmpeg", "-y", "-i", video_file, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", output_path]
   # subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def convert_all_mkv():
    mkv_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.mkv"))
    mp4_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.mp4"))
    input_files = mkv_files + mp4_files

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(convert_video, input_files)