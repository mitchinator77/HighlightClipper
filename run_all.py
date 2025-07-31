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
from killfeed_detector import detect_killfeed_events, load_templates
from highlight_logger import log_event
from headshot_audio import detect_headshot_audio_peaks
from audio_spike_detector import detect_audio_peaks
from temporal_convergence_scorer import compute_convergence_score
from clip_scorer import score_all_clips
from scoring_logger import log_scores_to_file, log_game_classifications
from highlight_filter_and_trimmer import trim_highlights

if __name__ == "__main__":
    from game_recognizer import classify_chunks_by_game_parallel

    log_header("Converting MKV to MP4...")
    convert_all_mkv()

    log_header("Chunking videos...")
    chunk_all_videos()

    log_header("Classifying game per chunk...")
    chunk_game_map = classify_chunks_by_game_parallel("Chunks")
    log_game_classifications(chunk_game_map)

    Path("logs").mkdir(exist_ok=True)
    with open("logs/run_log.json", "w") as f:
        json.dump(chunk_game_map, f, indent=2)

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
            killfeed_events = detect_killfeed_events(chunk_path, all_templates.get("killfeed", []))
            log_event(chunk_filename, "killfeed", killfeed_events)

            headshot_peaks = detect_headshot_audio_peaks(chunk_path)
            log_event(chunk_filename, "headshot_audio", headshot_peaks)

            audio_peaks = detect_audio_peaks(chunk_path)
            log_event(chunk_filename, "audio_spike", audio_peaks)

            convergence_scores = compute_convergence_score(chunk_filename)
            log_scores_to_file(chunk_filename, convergence_scores)

            trim_highlights(chunk_filename, convergence_scores)

        except Exception as e:
            log.error(f"❌ Error processing {chunk_filename}: {e}")

    log.info("✅ Pipeline complete.")
