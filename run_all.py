import logging
import os
from pathlib import Path
import json

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

log = setup_logger()

def log_header(text):
    log.info(f"🔧 {text} 🔧")

from mkv_converter import convert_all_mkv
from chunker import chunk_all_videos
from game_recognizer import classify_chunks_by_game_parallel
from killfeed_detector import detect_killfeed_events, load_templates
from highlight_logger import log_event
from headshot_audio import detect_headshot_audio_peaks
from audio_spike_detector import detect_audio_peaks
from temporal_convergence_scorer import compute_convergence_score
from clip_scorer import score_all_clips
from scoring_logger import log_scores_to_file, log_game_classifications
from highlight_filter_and_trimmer import trim_highlights

log_header("Classifying game per chunk...")
chunk_game_map = classify_chunks_by_game_parallel("Chunks")
log_game_classifications(chunk_game_map)

# 🧠 Save predictions for accuracy comparison
Path("logs").mkdir(exist_ok=True)
with open("logs/run_log.json", "w") as f:
    json.dump(chunk_game_map, f, indent=2)



def main():
    # ✅ Step 1: Convert MKV to MP4
    log_header("Converting MKV to MP4...")
    convert_all_mkv()

    # ✅ Step 2: Chunk videos
    log_header("Chunking videos...")
    chunk_all_videos()

    # ✅ Step 3: Game classification per chunk
    log_header("Classifying game per chunk...")
    from game_recognizer import classify_chunks_by_game_parallel
    chunk_game_map = classify_chunks_by_game_parallel("Chunks")

    log_game_classifications(chunk_game_map)

    # ✅ Step 4: Highlight detection for Valorant chunks
    log_header("Running highlight detection on Valorant chunks...")
    cleaned_template_path = os.path.join("cleaned_templates")
    all_templates = load_templates(cleaned_template_path)

    for chunk_filename, game in chunk_game_map.items():
        if game != "valorant":
            log.info(f"⏭️ Skipping non-Valorant chunk: {chunk_filename}")
            continue

        chunk_path = os.path.join("Chunks", chunk_filename)
        log.info(f"🎯 Processing Valorant chunk: {chunk_filename}")

        try:
            # Killfeed detection
            killfeed_events = detect_killfeed_events(chunk_path, all_templates.get("killfeed", []))
            log_event(chunk_filename, "killfeed", killfeed_events)

            # Audio-based headshot detection
            headshot_peaks = detect_headshot_audio_peaks(chunk_path)
            log_event(chunk_filename, "headshot_audio", headshot_peaks)

            # Audio spikes
            audio_peaks = detect_audio_peaks(chunk_path)
            log_event(chunk_filename, "audio_spike", audio_peaks)

            # Score convergence
            convergence_scores = compute_convergence_score(chunk_filename)
            log_scores_to_file(chunk_filename, convergence_scores)

            # Trim & export highlights
            trim_highlights(chunk_filename, convergence_scores)

        except Exception as e:
            log.error(f"❌ Error processing {chunk_filename}: {e}")

    log.info("✅ Pipeline complete.")

if __name__ == "__main__":
    main()
