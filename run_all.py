import logging
import os
from chunker import chunk_all_videos
from mkv_converter import convert_all_mkv
from game_recognizer import classify_chunks_by_game
from killfeed_detector import detect_killfeed_events, load_templates
from audio_spike_detector import detect_audio_peaks
from headshot_audio import detect_headshot_audio_peaks
from event_utils import normalize_event_timestamps
from temporal_convergence_scorer import compute_convergence_score
from highlight_filter_and_trimmer import trim_highlights
from highlight_logger import log_event
from clip_scorer import score_all_clips
from scoring_logger import log_scores_to_file
from trainer_trigger import check_and_trigger_trainer

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logger()

    logger.info("🔄 Converting MKV to MP4...")
    convert_all_mkv()

    logger.info("✂️ Chunking videos...")
    chunk_all_videos()

    logger.info("🎮 Classifying game per chunk...")
    game_labels = classify_chunks_by_game("Chunks")

    logger.info("⚙️ Running highlight detection on Valorant chunks...")
    killfeed_templates = load_templates("killfeed_templates")

    for chunk_name, game in game_labels.items():
        if game != "valorant":
            logger.info(f"⏭️ Skipping non-Valorant chunk: {chunk_name}")
            continue

        logger.info(f"🎯 Processing Valorant chunk: {chunk_name}")
        chunk_path = os.path.join("Chunks", chunk_name)

        # Detect events
        audio_peaks = detect_audio_peaks(chunk_path)
        headshot_peaks = detect_headshot_audio_peaks(chunk_path)
        killfeed_peaks = detect_killfeed_events(chunk_path, killfeed_templates)

        # Normalize timestamps
        all_events = normalize_event_timestamps(audio_peaks, killfeed_peaks, headshot_peaks)

        # Score based on convergence
        scored_clips = compute_convergence_score(chunk_path, all_events)

        # Trim highlights
        trim_highlights(scored_clips, chunk_path)

        # Log results
        log_event(chunk_path, audio_peaks, killfeed_peaks, headshot_peaks)
        log_scores_to_file(chunk_path, scored_clips)

        # Score for ranking
        score_all_clips()

        # Trigger self-training if needed
        check_and_trigger_trainer()

    logger.info("✅ Pipeline complete.")

if __name__ == "__main__":
    main()
