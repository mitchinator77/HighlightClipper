import os
import logging
from killfeed_detector import detect_killfeed_events
from audio_spike_detector import detect_audio_peaks
from headshot_audio import detect_headshot_audio_peaks
from clip_scorer import score_all_clips
from clip_feedback_trainer import log_feedback
import shutil

def move_processed_chunk(chunk_path):
    processed_dir = "ProcessedChunks"
    os.makedirs(processed_dir, exist_ok=True)
    shutil.move(chunk_path, os.path.join(processed_dir, os.path.basename(chunk_path)))

def process_valorant_chunk(chunk_path, templates):
    try:
        killfeed_events = detect_killfeed_events(chunk_path, templates)
        audio_peaks = detect_audio_peaks(chunk_path)
        headshot_peaks = detect_headshot_audio_peaks(chunk_path)

        score = score_all_clips(chunk_path, killfeed_events, audio_peaks, headshot_peaks)
        logging.info(f"⭐ Highlight score for {os.path.basename(chunk_path)}: {score:.2f}")

        is_highlight = score >= 0.75
        log_feedback(chunk_path, is_highlight)

        move_processed_chunk(chunk_path)
    except Exception as e:
        logging.error(f"❌ Error processing {chunk_path}: {e}")