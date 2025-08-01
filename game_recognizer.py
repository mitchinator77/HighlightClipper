
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from joblib import load
from pathlib import Path

def extract_features_from_frame(frame, size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    return resized.flatten()

def classify_game(frame, model):
    features = extract_features_from_frame(frame)
    features = features.reshape(1, -1)
    return model.predict(features)[0]

def classify_all_chunks(chunks_folder, output_path="logs/classification_scores.json", force_valo=False, max_chunks=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = load("game_classifier.joblib")

    classification_cache = {}

    # Try to load previous results, if corrupted, warn and reset
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                classification_cache = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Corrupted JSON in {output_path}, resetting...")
            classification_cache = {}
            os.remove(output_path)

    chunk_game_map = {}
    chunk_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith(".mp4")])

    if max_chunks:
        chunk_files = chunk_files[:max_chunks]

    for chunk_file in tqdm(chunk_files, desc="Classifying chunks"):
        if chunk_file in classification_cache:
            chunk_game_map[chunk_file] = classification_cache[chunk_file]
            continue

        chunk_path = os.path.join(chunks_folder, chunk_file)
        cap = cv2.VideoCapture(chunk_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = 5

        predictions = []
        for i in np.linspace(0, frame_count - 1, sample_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if success:
                prediction = classify_game(frame, model)
                predictions.append(prediction)

        cap.release()

        if force_valo:
            final_prediction = "valorant"
        else:
            final_prediction = max(set(predictions), key=predictions.count) if predictions else "unknown"

        chunk_game_map[chunk_file] = final_prediction
        classification_cache[chunk_file] = final_prediction
        print(f"📁 {chunk_file} → {final_prediction}")

    with open(output_path, "w") as f:
        json.dump(classification_cache, f, indent=2)

    return chunk_game_map
