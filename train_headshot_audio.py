import os
import numpy as np
import librosa
import pickle
from sklearn.cluster import KMeans
from pathlib import Path

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def train_model_from_dir(audio_dir="headshot_patterns", model_path="models/headshot_model.pkl", n_clusters=2):
    X = []
    file_paths = []

    for file in Path(audio_dir).glob("*.wav"):
        try:
            feats = extract_features(file)
            X.append(feats)
            file_paths.append(file)
        except Exception as e:
            print(f"⚠️ Failed to load {file}: {e}")

    if not X:
        print("❌ No features extracted.")
        return

    X = np.array(X)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")