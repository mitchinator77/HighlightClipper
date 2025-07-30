
import cv2
import json
from pathlib import Path

def classify_chunks_by_game(chunk_folder, templates_folder="killfeed_templates"):
    game_labels = {}
    template_files = list(Path(templates_folder).glob("*.png"))

    if not template_files:
        print("⚠️ No killfeed templates found in", templates_folder)
        return game_labels

    for chunk_path in Path(chunk_folder).glob("*.mp4"):
        cap = cv2.VideoCapture(str(chunk_path))
        success, frame = cap.read()
        found = False
        max_conf = 0
        if success:
            for template_path in template_files:
                template = cv2.imread(str(template_path), 0)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                max_conf = max(max_conf, max_val)
                if max_val > 0.8:
                    found = True
                    break
        cap.release()
        label = "valorant" if found else "other"
        print(f"🧪 Chunk: {chunk_path.name} | Game: {label} | Max match confidence: {max_conf:.3f}")
        game_labels[chunk_path.name] = label

    with open("clip_game_labels.json", "w") as f:
        json.dump(game_labels, f, indent=2)

    return game_labels
