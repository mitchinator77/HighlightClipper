
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import librosa

LABELS_DIR = "headshot_labels"
AUDIO_DIR = "Chunks"
MODEL_PATH = "headshot_audio_classifier.joblib"

def extract_audio_features(chunk_path, timestamps, sr=16000, window_size=0.5):
    y, _ = librosa.load(chunk_path, sr=sr)
    features = []
    labels = []
    for ts in timestamps:
        start = int((ts - window_size/2) * sr)
        end = int((ts + window_size/2) * sr)
        if start < 0 or end > len(y):
            continue
        segment = y[start:end]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).flatten()
        features.append(mfcc)
        labels.append(1)
    return features, labels

def generate_negative_samples(y, positive_ts, sr=16000, window_size=0.5, count=10):
    features = []
    labels = []
    duration = len(y) / sr
    attempts = 0
    while len(features) < count and attempts < count * 5:
        ts = np.random.uniform(window_size/2, duration - window_size/2)
        if all(abs(ts - p) > window_size for p in positive_ts):
            start = int((ts - window_size/2) * sr)
            end = int((ts + window_size/2) * sr)
            if start < 0 or end > len(y):
                continue
            segment = y[start:end]
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).flatten()
            features.append(mfcc)
            labels.append(0)
        attempts += 1
    return features, labels

def train_from_labels():
    X, y = [], []
    for label_file in os.listdir(LABELS_DIR):
        if not label_file.endswith(".json"):
            continue
        path = os.path.join(LABELS_DIR, label_file)
        with open(path) as f:
            timestamps = json.load(f)
        chunk_name = os.path.splitext(label_file)[0] + ".mp4"
        chunk_path = os.path.join(AUDIO_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            print(f"Missing audio file for {chunk_name}")
            continue
        y_audio, sr = librosa.load(chunk_path, sr=16000)
        pos_X, pos_y = extract_audio_features(chunk_path, timestamps, sr)
        neg_X, neg_y = generate_negative_samples(y_audio, timestamps, sr, count=len(pos_X))
        X.extend(pos_X + neg_X)
        y.extend(pos_y + neg_y)

    if not X:
        print("No training data found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Saved new classifier to: {MODEL_PATH}")

if __name__ == "__main__":
    train_from_labels()
