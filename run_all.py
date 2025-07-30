import logging
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from moviepy.editor import AudioFileClip
from infer_headshot import predict as predict_headshot
from highlight_logger import log_event


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

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="🌀 [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def log_header(text):
    logging.info(f"🔧 {text} 🔧")

def extract_audio_waveform(video_path: str) -> tuple[np.ndarray, int]:
    """
    Extracts audio from video and returns a normalized waveform and sample rate.
    """
    try:
        audio = AudioFileClip(video_path)
        temp_path = "temp.wav"
        audio.write_audiofile(temp_path, verbose=False, logger=None)
        rate, waveform = wavfile.read(temp_path)
        os.remove(temp_path)

        if len(waveform.shape) == 2:  # stereo
            waveform = waveform.mean(axis=1)

        if np.issubdtype(waveform.dtype, np.integer):
            waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
        elif np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float32)  # already normalized
        else:
            raise TypeError(f"Unsupported waveform dtype: {waveform.dtype}")
        return waveform, rate
    except Exception as e:
        logging.error(f"❌ Error extracting audio: {e}")
        return np.array([]), 44100


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
        logging.info("🧠 Running Detection + Convergence Scoring...")
        templates = load_templates("killfeed_templates")
        chunked_videos = sorted(Path("Chunks").glob("*.mp4"))

        for clip_path in chunked_videos:
            clip_name = clip_path.name
            logging.info(f"🎯 Processing: {clip_name}")

            visual_score, visual_timestamps = detect_killfeed_events(str(clip_path), templates)
            logging.info(f"🔍 Killfeed → Score: {visual_score} → Timestamps: {visual_timestamps}")

            waveform, rate = extract_audio_waveform(str(clip_path))

            if waveform.size > 0:
                audio_timestamps = detect_audio_peaks(waveform, rate)
                headshot_timestamps = detect_headshot_audio_peaks(waveform, rate)
            else:
                audio_timestamps = []
                headshot_timestamps = []

            events = normalize_event_timestamps({
                "visual": visual_timestamps,
                "audio": audio_timestamps,
                "headshot": headshot_timestamps
            })

            scored_moments = compute_convergence_score(events)

            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_scores_to_file(scored_moments, events, logs_dir / f"{clip_path.stem}_scoring.json")

            # Save final highlight clips
            trim_highlights(str(clip_path), scored_moments)
            # Classify and log headshots
            try:
                for ts in headshot_timestamps:
                    label, confidence = predict_headshot(str(clip_path))
                    classify_and_log_audio(clip_path, label, ts, confidence)
            except Exception as e:
                logging.error(f"⚠️ Headshot classifier failed on {clip_path.name}: {e}")


        logging.info("✅ All detection, scoring, and trimming complete.")
    except Exception as e:
        logging.error(f"❌ Error during detection/scoring: {e}")

    try:
        logging.info("📊 Running Fallback Clip Scoring...")
        score_all_clips()
        logging.info("✅ Fallback scoring complete.")
    except Exception as e:
        logging.error(f"❌ Error in fallback scoring: {e}")

    log_header("Pipeline Finished ✅")

if __name__ == "__main__":
    main()
