import os
import wave
import numpy as np

SAMPLE_RATE = 44100  # CD quality
DURATION = 1.0       # 1 second
AMPLITUDE = 32000

def generate_sharp_peak():
    # Gunshot-like spike in the middle
    audio = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.int16)
    audio[SAMPLE_RATE // 2] = AMPLITUDE
    return audio

def generate_ambient_noise():
    # Soft background hum
    return (np.random.rand(int(SAMPLE_RATE * DURATION)) * 500 - 250).astype(np.int16)

def save_wav(path, data):
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())

def main():
    os.makedirs("audiosamples/headshot/headshot", exist_ok=True)
    os.makedirs("audiosamples/headshot/non_headshot", exist_ok=True)

    for i in range(5):
        save_wav(f"audiosamples/headshot/headshot/headshot_{i}.wav", generate_sharp_peak())
        save_wav(f"audiosamples/headshot/non_headshot/non_headshot_{i}.wav", generate_ambient_noise())

    print("✅ Dummy audio samples created in audiosamples/headshot/")

if __name__ == "__main__":
    main()
