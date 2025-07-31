import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

classifier_path = "models/game_classifier.joblib"
clf = joblib.load(classifier_path)

def extract_frame_features(filepath, skip_interval=15):
    cap = cv2.VideoCapture(filepath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []

    for idx in range(0, frame_count, skip_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_crop = gray[h//4:3*h//4, w//4:3*w//4]
        resized = cv2.resize(center_crop, (64, 64)).flatten()
        selected_frames.append(resized)

    cap.release()
    if not selected_frames:
        return np.zeros(4096)
    return np.mean(selected_frames, axis=0)

def classify_chunk(chunk_path):
    try:
        features = extract_frame_features(chunk_path)
        prediction = clf.predict([features])[0]
        return chunk_path, prediction
    except Exception as e:
        return chunk_path, f"error: {e}"

def classify_chunks_parallel(chunks_dir, num_workers=None):
    num_workers = num_workers or max(cpu_count() - 1, 1)
    chunk_paths = [str(p) for p in Path(chunks_dir).rglob("*.mp4")]

    results = {}
    with Pool(num_workers) as pool:
        for path, prediction in tqdm(pool.imap_unordered(classify_chunk, chunk_paths), total=len(chunk_paths), desc="🎮 Classifying chunks (parallel)"):
            results[path] = prediction

    return results
