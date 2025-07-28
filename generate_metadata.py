import os
import pandas as pd
from pathlib import Path

CLIP_DIR = Path("./AutoClips")
TRANSCRIPT_DIR = Path("./Transcripts")
METADATA_OUTPUT = Path("./metadata.csv")

CATEGORY_KEYWORDS = {
    "Clutch": ["1v5", "clutch", "last man", "ace"],
    "Fails": ["fail", "oops", "no!", "whiff", "throw"],
    "Trash Talk": ["trash", "easy", "you suck", "kid", "noob"],
    "Highlights": ["insane", "epic", "crazy", "what a play", "let's go"],
    "Funny": ["lol", "funny", "lmao", "dumb", "why", "wtf"]
}

def categorize_clip(transcript_text):
    text = transcript_text.lower()
    categories = [cat for cat, keywords in CATEGORY_KEYWORDS.items() if any(word in text for word in keywords)]
    return categories or ["Uncategorized"]

def suggest_title(transcript_text):
    words = transcript_text.strip().split()
    if len(words) == 0:
        return "Untitled Clip"
    core = " ".join(words[:6]) + ("..." if len(words) > 6 else "")
    return f'"{core}" – Best Moments'

metadata = []
for clip_file in CLIP_DIR.iterdir():
    if clip_file.suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi"}:
        continue
    transcript_path = TRANSCRIPT_DIR / (clip_file.stem + ".txt")
    if not transcript_path.exists():
        continue
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    tags = categorize_clip(transcript)
    title = suggest_title(transcript)
    metadata.append({
        "clip_filename": clip_file.name,
        "title_idea": title,
        "tags": ", ".join(tags),
        "transcript_excerpt": transcript[:100] + ("..." if len(transcript) > 100 else "")
    })

df = pd.DataFrame(metadata)
df.to_csv(METADATA_OUTPUT, index=False)
print(f"Metadata exported to {METADATA_OUTPUT}")