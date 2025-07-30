import cv2
import json
import os
from pathlib import Path

CHUNKS_DIR = "Chunks"
FEEDBACK_FILE = "chunk_feedback.json"
SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]
MAX_WIDTH = 1280
MAX_HEIGHT = 720

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}

def save_feedback(feedback):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)

def resize_frame(frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        scaling_factor = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        return cv2.resize(frame, (int(w * scaling_factor), int(h * scaling_factor)))
    return frame

def label_chunks():
    feedback = load_feedback()
    chunk_files = sorted(Path(CHUNKS_DIR).glob("*"))
    unlabeled = [str(f) for f in chunk_files if f.suffix.lower() in SUPPORTED_EXTENSIONS and str(f) not in feedback]

    print(f"🧠 Found {len(unlabeled)} unlabeled chunks.")

    for chunk_path in unlabeled:
        cap = cv2.VideoCapture(chunk_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"❌ Failed to load {chunk_path}")
            continue

        frame = resize_frame(frame)
        window_title = f"Label: {os.path.basename(chunk_path)} (v = Valorant, o = Other, q = Quit)"
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(0)

        if key == ord('v'):
            feedback[chunk_path] = "valorant"
            print(f"✅ Labeled as valorant")
        elif key == ord('o'):
            feedback[chunk_path] = "other"
            print(f"❌ Labeled as other")
        elif key == ord('q'):
            print("👋 Exiting early...")
            break

        save_feedback(feedback)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    label_chunks()
