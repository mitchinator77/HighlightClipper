import os
import numpy as np
import librosa
from pathlib import Path

HEADSHOT_DIR = Path("TrainingAudio/headshot")
NON_HEADSHOT_DIR = Path("TrainingAudio/non_headshot")
OUTPUT_FILE = Path("TrainingAudio/audio_features.npz")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    return np.concatenate([mfccs, chroma, contrast])

def process_directory(directory, label):
    features = []
    for file in directory.glob("*.wav"):
        try:
            vec = extract_features(file)
            features.append((vec, label))
        except Exception as e:
            print(f"⚠️ Failed to process {file.name}: {e}")
    return features

def main():
    data = []
    data += process_directory(HEADSHOT_DIR, 1)
    data += process_directory(NON_HEADSHOT_DIR, 0)

    if not data:
        print("❌ No data found. Check folders.")
        return

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"✅ Extracted features saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
