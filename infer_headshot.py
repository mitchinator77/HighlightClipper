import librosa
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "Models/headshot_audio_classifier.h5"
SAMPLE_RATE = 22050
DURATION = 2.0
MFCC_DIM = (40, 87)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM[0])
    if mfcc.shape[1] < MFCC_DIM[1]:
        pad_width = MFCC_DIM[1] - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    return mfcc[..., np.newaxis]

def predict(audio_file):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    mfcc = extract_features(audio_file)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)[0]
    label = "headshot" if np.argmax(prediction) == 1 else "normal"
    confidence = float(np.max(prediction))
    return label, confidence

if __name__ == "__main__":
    import sys
    label, confidence = predict(sys.argv[1])
    print(f"Prediction: {label} ({confidence:.2f})")