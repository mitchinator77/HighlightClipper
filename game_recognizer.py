import os
import cv2
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dummy features for known games
KNOWN_FEATURES = {
    "valorant": np.array([0.6, 0.8, 0.75]),
    "other": np.array([0.2, 0.1, 0.15])
}

def extract_features(chunk_path):
    # Simulated feature extraction
    return np.random.rand(3)

def classify_chunk_features(features, threshold=0.75, force_valo=False):
    if force_valo:
        logger.info("🧪 Forcing classification to 'valorant'")
        return "valorant"

    similarities = {
        game: cosine_similarity([features], [feat])[0][0]
        for game, feat in KNOWN_FEATURES.items()
    }

    best_game = max(similarities, key=similarities.get)
    confidence = similarities[best_game]

    logger.info(f"🔍 Similarities: {similarities} → Best: {best_game} ({confidence:.2f})")

    if confidence >= threshold:
        return best_game
    else:
        fallback = "valorant" if best_game == "valorant" else "other"
        logger.warning(f"⚠️ Confidence {confidence:.2f} below threshold. Using fallback: {fallback}")
        return fallback

def classify_all_chunks(chunk_dir, output_path="logs/run_log.json", threshold=0.75, force_valo=False):
    chunk_dir = Path(chunk_dir)
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    classifications = {}

    for chunk_file in chunk_dir.glob("*.mp4"):
        features = extract_features(str(chunk_file))
        game = classify_chunk_features(features, threshold=threshold, force_valo=force_valo)
        classifications[chunk_file.name] = game

    with open(output_path, "w") as f:
        json.dump({"game_classifications": classifications}, f, indent=2)

    logger.info(f"✅ Saved classifications to {output_path}")
    return classifications
