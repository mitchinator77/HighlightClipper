import logging
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from mkv_converter import convert_all_mkv
from chunker import chunk_all_videos
from killfeed_detector import detect_killfeed_events, load_templates
from clip_scorer import score_all_clips
from temporal_convergence_scorer import compute_convergence_score
from event_utils import normalize_event_timestamps
from scoring_logger import log_scores_to_file
from highlight_filter_and_trimmer import trim_highlights
from audio_spike_detector import detect_audio_peaks
from headshot_audio import detect_headshot_audio_peaks
from highlight_logger import log_event
from parallel_processing_utils import process_clip

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def log_header(text):
    logging.info(f"🔧 {text} 🔧")

def classify_and_log_audio(clip_path, label, timestamp, confidence):
    log_event(
        video_name=clip_path.stem,
        event_type="headshot_detection",
        timestamp=timestamp,
        confidence=confidence,
        tags=[label]
    )
    if label == "headshot":
        headshot_dir = Path("Highlights/Headshots")
        headshot_dir.mkdir(parents=True, exist_ok=True)
        dest = headshot_dir / clip_path.name
        clip_path.rename(dest)

def main():
    logger = setup_logger()
    log_header("Starting HighlightClipper (Parallel Mode)")

    def load_predict_headshot():
        from infer_headshot import predict
        return predict

    try:
        logger.info("🎞️ Converting MKV to MP4 from SourceVideos/...")
        convert_all_mkv(source_dir="SourceVideos")
        logger.info("✅ MKV Conversion complete.")
    except Exception as e:
        logger.error(f"❌ Error in MKV conversion: {e}")

    try:
        logger.info("✂️ Chunking newly converted MP4s...")
        chunk_all_videos(source_dir="SourceVideos")
        logger.info("✅ Chunking complete.")
    except Exception as e:
        logger.error(f"❌ Error in chunking: {e}")

    try:
        logger.info("🧠 Running Parallel Detection + Scoring...")
        templates = load_templates("killfeed_templates")
        chunked_videos = sorted(Path("Chunks").glob("*.mp4"))

        args = [(clip_path, templates, logger, trim_highlights, detect_killfeed_events, log_scores_to_file,
                 detect_audio_peaks, detect_headshot_audio_peaks, load_predict_headshot)
                for clip_path in chunked_videos]

        with Pool(processes=4) as pool:
            pool.starmap(process_clip, args)

        logger.info("✅ Parallel detection, scoring, and trimming complete.")
    except Exception as e:
        logger.error(f"❌ Error during detection/scoring: {e}")

    try:
        logger.info("📊 Running Fallback Clip Scoring...")
        score_all_clips()
        logger.info("✅ Fallback scoring complete.")
    except Exception as e:
        logger.error(f"❌ Error in fallback scoring: {e}")

    log_header("Pipeline Finished ✅")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # ← makes it explicit
    # Then start your processing logic
    main()
