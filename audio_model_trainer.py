import os
import pickle
import numpy as np
from audio_feature_utils import extract_audio_features
from sklearn.ensemble import RandomForestClassifier

def train_headshot_audio_model(training_dir, output_path="trained_headshot_model.pkl"):
    X, y = [], []

    for label in ["headshot", "non_headshot"]:
        folder = os.path.join(training_dir, label)
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                features = extract_audio_features(os.path.join(folder, fname))
                X.append(features)
                y.append(1 if label == "headshot" else 0)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {output_path}")
