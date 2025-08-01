import os
import json
import logging
from template_matcher import match_templates_on_chunk

logger = logging.getLogger("game_recognizer")
logging.basicConfig(level=logging.INFO)

def load_clarified_labels(path="manual_clarification/clarified_labels.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def classify_all_chunks(chunk_dir, output_path="logs/run_log.json", threshold=0.75, force_valo=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifications = {}
    clarified_labels = load_clarified_labels()

    for chunk_file in os.listdir(chunk_dir):
        if not chunk_file.endswith(".mp4"):
            continue

        full_path = os.path.join(chunk_dir, chunk_file)

        if force_valo:
            logger.info(f"🧪 Forcing classification to 'valorant' for {chunk_file}")
            classifications[chunk_file] = "valorant"
            continue

        if chunk_file in clarified_labels:
            label = clarified_labels[chunk_file]
            logger.info(f"✅ Using clarified label for {chunk_file}: {label}")
            classifications[chunk_file] = label
            continue

        result = match_templates_on_chunk(full_path, threshold=threshold)
        classifications[chunk_file] = result["label"]

        # Save debug info
        result["chunk"] = chunk_file
        with open("logs/classification_scores.json", "a") as log_file:
            log_file.write(json.dumps(result) + "\n")

    with open(output_path, "w") as f:
        json.dump({"game_classifications": classifications}, f, indent=2)

    logger.info(f"✅ Saved classifications to {output_path}")
    return classifications