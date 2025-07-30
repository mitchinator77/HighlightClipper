import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm

MODEL_PATH = "game_classifier.joblib"

def extract_frame_features(video_path, frame_sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, frame_count, frame_sample_rate))

    all_features = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 96))  # 128 x 96 = 12,288 features
        features = resized.flatten()
        all_features.append(features)

    cap.release()

    if not all_features:
        return np.zeros((64 * 64,))

    return np.mean(all_features, axis=0)

def classify_chunks_by_game(chunk_dir):
    clf = joblib.load(MODEL_PATH)
    predictions = {}

    for filename in tqdm(os.listdir(chunk_dir), desc="🎮 Classifying chunks"):
        if filename.endswith(".mp4"):
            filepath = os.path.join(chunk_dir, filename)
            features = extract_frame_features(filepath)
            prediction = clf.predict([features])[0]
            predictions[filename] = prediction

    return predictions
