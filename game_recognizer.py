import cv2
import os
from pathlib import Path

VALORANT_LABEL = "valorant"
NON_VALO_LABEL = "other"
CLEANED_TEMPLATE_DIR = Path("cleaned_templates")
HUD_TEMPLATE_DIR = "templates/hud_elements"
TEMPLATE_ROOT_DIR = "templates"
HUD_TEMPLATE_DIR = os.path.join(TEMPLATE_ROOT_DIR, "hud_elements")
BUYPHASE_DIR = os.path.join(TEMPLATE_ROOT_DIR, "buyphase_banner")
SCOREBOARD_DIR = os.path.join(TEMPLATE_ROOT_DIR, "scoreboard")
DETECTION_THRESHOLD = 0.7

def load_templates_from_directory(directory):
    templates = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((filename, img))
    return templates

KILLFEED_TEMPLATES = load_templates_from_directory(CLEANED_TEMPLATE_DIR)
HUD_TEMPLATES = load_templates_from_directory(HUD_TEMPLATE_DIR)

def classify_frame(frame, templates, threshold=DETECTION_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for name, template in templates:
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= threshold:
            print(f"✅ Match: {name} (score: {max_val:.2f})")
            return True
    return False

def classify_video_game(video_path, frame_skip=30, max_frames=10):
    cap = cv2.VideoCapture(str(video_path))
    matched = False
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened() and frame_count < max_frames:
        frame_idx = frame_count * frame_skip
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if classify_frame(frame, KILLFEED_TEMPLATES) or classify_frame(frame, HUD_TEMPLATES):
            matched = True
            break
        frame_count += 1

    cap.release()
    return VALORANT_LABEL if matched else NON_VALO_LABEL
    

def classify_chunks_by_game(chunks_dir):
    chunk_game_map = {}

    for chunk_file in Path(chunks_dir).glob("*.mp4"):
        print(f"🧠 Classifying game for chunk: {chunk_file.name}")
        game_label = classify_video_game(chunk_file)
        chunk_game_map[str(chunk_file)] = game_label
        print(f"🔖 Classified as: {game_label}")

    return chunk_game_map
