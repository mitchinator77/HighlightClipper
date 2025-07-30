
import os
from pathlib import Path
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import json

SOURCE_DIR = "SourceVideos"
AUDIO_OUT_DIR = "AudioSamples/headshot"
EVENTS_JSON = "Detected/headshot_audio_events.json"
CLIP_DURATION = 2.0
SAMPLE_RATE = 22050

os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

def extract_audio_clip(video_path, timestamp, out_path):
    try:
        clip = VideoFileClip(str(video_path)).subclip(max(0, timestamp - CLIP_DURATION/2), timestamp + CLIP_DURATION/2)
        audio = clip.audio
        audio_path = out_path.with_suffix(".wav")
        audio.write_audiofile(str(audio_path), fps=SAMPLE_RATE, verbose=False, logger=None)
        return True
    except Exception as e:
        print(f"❌ Failed to extract audio at {timestamp} from {video_path.name}: {e}")
        return False

def load_event_timestamps():
    if not os.path.exists(EVENTS_JSON):
        print(f"❌ Missing: {EVENTS_JSON}")
        return {}

    with open(EVENTS_JSON, "r") as f:
        return json.load(f)

def main():
    events = load_event_timestamps()

    for video_name, timestamps in events.items():
        video_path = Path(SOURCE_DIR) / video_name
        if not video_path.exists():
            print(f"⚠️ Missing video: {video_name}")
            continue

        for i, ts in enumerate(timestamps):
            out_filename = f"{video_path.stem}_ts{int(ts*1000)}ms.wav"
            out_path = Path(AUDIO_OUT_DIR) / out_filename
            if extract_audio_clip(video_path, ts, out_path):
                print(f"✅ Saved: {out_filename}")

if __name__ == "__main__":
    main()
