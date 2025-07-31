import os
import json
import logging
from random import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Placeholder for actual model prediction function
def classify_game(chunk_path, confidence_threshold=0.7, force_valo=False):
    """
    Simulates game classification. Replace with real ML model logic.
    """
    if force_valo:
        logger.info(f"🧪 [DEBUG] Forcing Valorant classification for {chunk_path}")
        return "valorant"

    # Simulated prediction output (should be replaced by real model inference)
    simulated_prediction = "valorant" if "valo" in chunk_path.lower() else "other"
    simulated_confidence = 0.85 if simulated_prediction == "valorant" else 0.65

    logger.info(f"📼 {chunk_path} → Predicted: {simulated_prediction} (confidence: {simulated_confidence:.2f})")

    if simulated_confidence >= confidence_threshold:
        return simulated_prediction
    else:
        return "other"  # Avoid 'unknown' unless explicitly required

def classify_all_chunks(chunk_folder, output_json="logs/run_log.json", confidence_threshold=0.7):
    chunk_files = [f for f in os.listdir(chunk_folder) if f.endswith(".mp4")]
    results = {}

    for chunk in chunk_files:
        chunk_path = os.path.join(chunk_folder, chunk)
        game = classify_game(chunk_path, confidence_threshold=confidence_threshold)
        results[chunk] = game

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"game_classifications": results}, f, indent=2)

    logger.info(f"✅ Game classifications saved to {output_json}")
