import json
import os
import cv2
from tqdm import tqdm

CHUNKS_DIR = "Chunks"
OUTPUT_JSON = "training_data.json"

with open("chunk_feedback.json", "r") as f:
    feedback = json.load(f)

training_data = []

for video_path, label in tqdm(feedback.items(), desc="🔍 Extracting features"):
    # ✅ Normalize to just the filename
    filename = os.path.basename(video_path)
    full_path = os.path.join(CHUNKS_DIR, filename)

    if not os.path.isfile(full_path):
        print(f"⚠️ Missing: {full_path}")
        continue

    cap = cv2.VideoCapture(full_path)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        print(f"❌ Failed to read: {filename}")
        continue

    resized = cv2.resize(frame, (64, 64)).flatten().tolist()  # ↓ adjustable resolution
    training_data.append({"features": resized, "label": label})

# ✅ Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(training_data, f, indent=2)

print(f"✅ Saved {len(training_data)} entries to {OUTPUT_JSON}")
