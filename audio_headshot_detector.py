import os
import pickle
import subprocess
import numpy as np
from audio_feature_utils import extract_audio_features

MODEL_PATH = "trained_headshot_model.pkl"

def extract_audio(video_path, output_wav="temp_audio.wav"):
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", output_wav
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav

def detect_headshot_audio(video_path):
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No trained model found. Skipping audio detection.")
        return 0, []  # score 0, empty timestamps

    audio_path = extract_audio(video_path)
    features = extract_audio_features(audio_path)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict_proba([features])[0][1]
    return round(prediction, 2)
