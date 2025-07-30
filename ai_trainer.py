import os
import json
import shutil
import librosa
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip

LOG_DIR = "TrainerLogs"
AUDIO_DIR = "AudioSamples"
VIDEO_DIR = "Highlights/Headshots"
MODEL_DIR = "Models"
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "headshot_audio_classifier_latest.h5")
SAMPLE_RATE = 22050
DURATION = 2.0
MFCC_DIM = (40, 87)

def extract_audio_from_clip(clip_name, label):
    mp4_path = os.path.join(VIDEO_DIR, clip_name + ".mp4")
    wav_path = os.path.join(AUDIO_DIR, label, clip_name + ".wav")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    if os.path.exists(mp4_path):
        try:
            video = VideoFileClip(mp4_path)
            video.audio.write_audiofile(wav_path, logger=None)
            return True
        except Exception as e:
            print(f"⚠️ Failed to extract audio from {mp4_path}: {e}")
    return False

def validate_audio_sample(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < sr * 1.5:  # too short
            return False
        if np.sqrt(np.mean(y**2)) < 0.005:  # low RMS (silence)
            return False
        return True
    except Exception:
        return False

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM[0])
    if mfcc.shape[1] < MFCC_DIM[1]:
        pad_width = MFCC_DIM[1] - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    return mfcc[..., np.newaxis]

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(MFCC_DIM[0], MFCC_DIM[1], 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_labeled_training_data(min_score=3):
    X, y = [], []
    label_map = {"headshot": 1, "normal": 0}
    for file in os.listdir(LOG_DIR):
        if file.endswith(".json"):
            with open(os.path.join(LOG_DIR, file), "r") as f:
                data = json.load(f)
            if data.get("ai_score", 0) >= min_score or data.get("manual_approved") is True:
                clip_name = data["clip"]
                label = "headshot" if "headshot" in data["predicted_tags"] else "normal"
                audio_path = os.path.join(AUDIO_DIR, label, clip_name + ".wav")

                if not os.path.exists(audio_path):
                    extract_audio_from_clip(clip_name, label)

                if os.path.exists(audio_path) and validate_audio_sample(audio_path):
                    mfcc = extract_features(audio_path)
                    weight = max(1, data.get("ai_score", 1))
                    X.extend([mfcc] * weight)
                    y.extend([label_map[label]] * weight)
    X = np.array(X)
    y = to_categorical(np.array(y))
    return train_test_split(X, y, test_size=0.2, random_state=42)

def retrain_model():
    if not os.path.exists(LOG_DIR):
        print("No training logs found.")
        return
    X_train, X_test, y_train, y_test = load_labeled_training_data()
    if len(X_train) == 0:
        print("No valid data to retrain.")
        return
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"headshot_audio_classifier_v{timestamp}.h5")
    model.save(versioned_path)
    model.save(LATEST_MODEL_PATH)
    print(f"✅ Model retrained.
→ Versioned model: {versioned_path}
→ Latest model: {LATEST_MODEL_PATH}")

if __name__ == "__main__":
    retrain_model()