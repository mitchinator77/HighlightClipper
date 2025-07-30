import cv2
import os
from pathlib import Path

def trim_and_save_clip(video_path, timestamp, output_folder, tag="highlight", pre=2.0, post=2.0):
    os.makedirs(output_folder, exist_ok=True)
    filename = Path(video_path).stem
    output_filename = f"{filename}_{int(timestamp)}s_{tag}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    cmd = (
        f'ffmpeg -y -ss {max(0, timestamp - pre):.2f} -i "{video_path}" '
        f'-t {pre + post:.2f} -c copy "{output_path}"'
    )
    os.system(cmd)
