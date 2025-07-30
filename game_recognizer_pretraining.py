import cv2
import os
from pathlib import Path

VALORANT_LABEL = "valorant"
NON_VALO_LABEL = "other"
TEMPLATE_ROOT_DIR = "templates"
CLEANED_TEMPLATE_DIR = Path("cleaned_templates")

# Define subfolder paths for various HUD elements
SUBFOLDERS = ["killfeed", "buyphase_banner", "hud_elements", "scoreboard"]
TEMPLATE_SETS = {} 

DETECTION_THRESHOLD = 0.7
FRAME_SKIP = 30
MAX_FRAMES = 10

def load_templates_from_directory(directory):
    templates = []
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"⚠️ Directory does not exist: {directory}")
        return []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((filename, img))
    return templates

# Load templates from all subfolders
for sub in SUBFOLDERS:
    path = os.path.join(TEMPLATE_ROOT_DIR, sub)
    TEMPLATE_SETS[sub] = load_templates_from_directory(path)

# Also load cleaned killfeed templates
TEMPLATE_SETS["cleaned_killfeed"] = load_templates_from_directory(CLEANED_TEMPLATE_DIR)

def classify_frame(frame, templates, threshold=DETECTION_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for name, template in templates:
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= threshold:
            print(f"✅ Match: {name} (score: {max_val:.2f})")
            return True
    return False

def classify_video_game(video_path, frame_skip=FRAME_SKIP, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    matched = False
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎥 Classifying game in {video_path} using {len(TEMPLATE_SETS)} template groups")

    while cap.isOpened() and frame_count < max_frames:
        frame_idx = frame_count * frame_skip
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        print(f"🔍 Analyzing frame {frame_idx}...")

        for group, templates in TEMPLATE_SETS.items():
            print(f"  🧪 Checking templates from: {group}")
            if classify_frame(frame, templates):
                print(f"🎯 Game detected as VALORANT via group '{group}' at frame {frame_idx}")
                matched = True
                break
        if matched:
            break
        frame_count += 1

    cap.release()
    return VALORANT_LABEL if matched else NON_VALO_LABEL

def classify_chunks_by_game(chunks_dir):
    chunk_game_map = {}
    for chunk_file in Path(chunks_dir).glob("*.mp4"):
        print(f"\n🧠 Classifying game for chunk: {chunk_file.name}")
        game_label = classify_video_game(chunk_file)
        chunk_game_map[str(chunk_file)] = game_label
        print(f"🔖 Result: {chunk_file.name} => {game_label}")
    return chunk_game_map
