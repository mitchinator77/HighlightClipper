import os
import whisper
from pathlib import Path

CLIP_DIR = Path("./AutoClips")
TRANSCRIPT_DIR = Path("./Transcripts")
MODEL_SIZE = "base"

TRANSCRIPT_DIR.mkdir(exist_ok=True)

print(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)

def transcribe_clip(clip_path):
    print(f"Transcribing {clip_path.name}...")
    result = model.transcribe(str(clip_path))
    txt_path = TRANSCRIPT_DIR / (clip_path.stem + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return result["text"]

def main():
    for file in CLIP_DIR.iterdir():
        if file.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}:
            txt_file = TRANSCRIPT_DIR / (file.stem + ".txt")
            if not txt_file.exists():
                transcribe_clip(file)
    print("Transcription complete.")

if __name__ == "__main__":
    main()