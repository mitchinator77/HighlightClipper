import logging
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path

from mkv_converter import convert_all_mkv
from chunker import chunk_all_videos
from game_recognizer import classify_all_chunks
from killfeed_detector import detect_killfeed_events, load_templates
from audio_spike_detector import detect_audio_peaks
from headshot_audio import detect_headshot_audio_peaks
from clip_scorer import score_all_clips
from clip_feedback_trainer import log_feedback, train_on_feedback
from highlight_filter_and_trimmer import trim_highlights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-chunks', type=int, default=None, help='Limit number of chunks to classify')
    return parser.parse_args()


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def log_header(text):
    logging.info(f"🔧 {text} 🔧")

def move_processed_chunk(chunk_path):
    processed_dir = "ProcessedChunks"
    os.makedirs(processed_dir, exist_ok=True)
    shutil.move(chunk_path, os.path.join(processed_dir, os.path.basename(chunk_path)))

def main(max_chunks=None):
    logger = setup_logger()

    log_header("Converting MKV to MP4...")
    convert_all_mkv("SourceVideos")

    log_header("Chunking videos...")
    chunk_all_videos("ConvertedVideos")

    args = parse_args()

    log_header("Classifying game per chunk...")
    chunk_game_map = classify_all_chunks("Chunks", output_path="logs/run_log.json", force_valo=False, max_chunks=args.max_chunks)


    log_header("Running highlight detection on Valorant chunks...")
    templates = load_templates(Path("templates/killfeed"))

    for chunk_file, game in chunk_game_map.items():
        if game != "valorant":
            logging.info(f"⏭️ Skipping non-Valorant chunk: {chunk_file}")
            continue

        chunk_path = os.path.join("Chunks", chunk_file)

        killfeed_events = detect_killfeed_events(chunk_path, templates)
        audio_peaks = detect_audio_peaks(chunk_path)
        headshot_peaks = detect_headshot_audio_peaks(chunk_path)

        score = score_all_clips(chunk_path, killfeed_events, audio_peaks, headshot_peaks)
        logging.info(f"⭐ Highlight score for {chunk_file}: {score:.2f}")

        is_highlight = score >= 0.75
        log_feedback(chunk_path, is_highlight)

        move_processed_chunk(chunk_path)

    feedback_log = "feedback_logs/clip_feedback.json"
    if os.path.exists(feedback_log):
        with open(feedback_log) as f:
            feedback_count = len(f.readlines())
            if feedback_count and feedback_count % 3 == 0:
                logging.info("🧠 Triggering feedback trainer module...")
                train_on_feedback()

    log_header("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to process")
    args = parser.parse_args()
    main(max_chunks=args.max_chunks)