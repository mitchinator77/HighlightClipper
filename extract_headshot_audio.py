import os
import moviepy.editor as mp

AUDIO_CLIP_DURATION = 2.0  # seconds before/after timestamp
OUTPUT_DIR = "AudioSamples"

def extract_audio_samples(video_path, timestamps, label):
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    video = mp.VideoFileClip(video_path)

    for i, ts in enumerate(timestamps):
        start = max(0, ts - AUDIO_CLIP_DURATION / 2)
        end = min(video.duration, ts + AUDIO_CLIP_DURATION / 2)
        audio_clip = video.subclip(start, end).audio
        output_path = os.path.join(OUTPUT_DIR, label, f"{basename}_{label}_{i}.wav")
        audio_clip.write_audiofile(output_path, logger=None)