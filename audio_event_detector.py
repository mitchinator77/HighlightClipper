"""Detects spikes in audio to infer exciting moments."""
import numpy as np
import librosa

def detect_audio_spikes(audio_path, threshold_db=20):
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    spikes = energy > np.percentile(energy, 95)
    return np.sum(spikes)