import logging
from datetime import datetime
from pathlib import Path

from mkv_converter import convert_all_mkv
from chunker import chunk_all_videos
from killfeed_detector import detect_killfeed_events, load_templates
from clip_scorer import score_all_clips

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def log_header(text):
    logging.info(f"🔧 {text} 🔧")

def main():
    setup_logger()
    log_header("Starting HighlightClipper Pipeline")

    try:
        logging.info("🎞️ Converting MKV to MP4...")
        convert_all_mkv()
        logging.info("✅ MKV Conversion complete.")
    except Exception as e:
        logging.error(f"❌ Error in MKV conversion: {e}")

    try:
        logging.info("✂️ Chunking videos...")
        chunk_all_videos()
        logging.info("✅ Chunking complete.")
    except Exception as e:
        logging.error(f"❌ Error in chunking: {e}")

    try:
        logging.info("🧠 Running Killfeed Detection (Visual)…")
        templates = load_templates("killfeed_templates")
        chunked_videos = sorted(Path("Chunks").glob("*.mp4"))
        for clip_path in chunked_videos:
            score, timestamps = detect_killfeed_events(str(clip_path), templates)
            logging.info(f"🔍 {clip_path.name} → Killfeed score: {score} → Timestamps: {timestamps}")
        logging.info("✅ Killfeed detection complete.")
    except Exception as e:
        logging.error(f"❌ Error in killfeed detection: {e}")

    try:
        logging.info("📊 Scoring Clips...")
        score_all_clips()
        logging.info("✅ Clip scoring complete.")
    except Exception as e:
        logging.error(f"❌ Error in scoring clips: {e}")

    log_header("Pipeline Finished ✅")

if __name__ == "__main__":
    main()
