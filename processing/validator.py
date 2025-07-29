
import os

def validate_clips(directory):
    print(f"[VALIDATOR] Validating clips in: {directory}")
    valid_clips = []
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            valid_clips.append(filename)
    return valid_clips
