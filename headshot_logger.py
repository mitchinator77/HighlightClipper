import os
import cv2
import numpy as np
from pathlib import Path
from visual_headshot_flash_detector import detect_headshot_flash
from audio_headshot_detector import extract_audio_snippet

def log_headshot_audio_clips(video_path, output_dir="headshot_patterns"):
    Path(output_dir).mkdir(exist_ok=True)
    flashes = detect_headshot_flash(video_path, return_timestamps=True)

    for i, timestamp in enumerate(flashes):
        wav_path = os.path.join(output_dir, f"{Path(video_path).stem}_flash_{i}.wav")
        extract_audio_snippet(video_path, timestamp, 0.5, wav_path)
        print(f"🔊 Saved: {wav_path}")