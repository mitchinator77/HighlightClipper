import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

AUDIO_DIR = "AudioSamples"
MODEL_OUT_DIR = "Models/headshot_model_tf"  # Changed from .h5 to directory
SAMPLE_RATE = 22050
DURATION = 2.0  # seconds
MFCC_DIM = (40, 87)  # Adjusted based on sample duration and hop length

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM[0])
    if mfcc.shape[1] < MFCC_DIM[1]:
        pad_width = MFCC_DIM[1] - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    return mfcc

def load_data():
    X, y = [], []
    labels = {"headshot": 1, "normal": 0}
    for label in labels:
        folder = os.path.join(AUDIO_DIR, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                mfcc = extract_features(os.path.join(folder, file))
                X.append(mfcc)
                y.append(labels[label])
    X = np.array(X)[..., np.newaxis]
    y = to_categorical(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

def train():
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()

    # Save best model in SavedModel format (directory-based)
    checkpoint = ModelCheckpoint(
        filepath=MODEL_OUT_DIR,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_format="tf"  # <-- important!
    )

    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Optional: force manual save after training completes
    model.save(MODEL_OUT_DIR, save_format="tf")

if __name__ == "__main__":
    train()
