# audio_spike_detector.py

from typing import List
import numpy as np

def detect_audio_peaks(audio_waveform: np.ndarray, sample_rate: int, threshold: float = 0.6) -> List[float]:
    """
    Detects loud peaks in audio (possible kill/hype moments).
    Returns timestamps (in seconds).
    """
    window_size = int(0.1 * sample_rate)  # 100ms
    peaks = []
    for i in range(0, len(audio_waveform), window_size):
        window = audio_waveform[i:i + window_size]
        if np.max(np.abs(window)) > threshold:
            peaks.append(i / sample_rate)
    return peaks