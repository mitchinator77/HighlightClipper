import cv2
import numpy as np
import os
from pathlib import Path
from clip_utils import trim_and_save_clip
from scoring_logger import log_clip_decision
from headshot_audio import detect_headshot_audio_peaks
from killfeed_detector import detect_killfeed_events, load_templates
from spectator_detector import is_spectator_present

def trim_highlights(video_path, all_templates, output_dir="Highlights", maybe_dir="Maybes", headshot_audio_peaks=None):
    killfeed_templates = [t for t in all_templates if t.shape[0] > 20]  # crude filter for size-based selection
    score, timestamps = detect_killfeed_events(video_path, killfeed_templates)

    if headshot_audio_peaks is None:
        headshot_audio_peaks = detect_headshot_audio_peaks(video_path)

    headshot_boosted = []
    for ts in timestamps:
        boosted = any(abs(ts - hs_ts) < 1.0 for hs_ts in headshot_audio_peaks)
        headshot_boosted.append(boosted)

    for idx, ts in enumerate(timestamps):
        is_headshot = headshot_boosted[idx]
        confidence_score = score + (2 if is_headshot else 0)

        if is_spectator_present(video_path, ts):
            log_clip_decision(video_path, ts, confidence_score, "Spectating - Skipped")
            continue

        if confidence_score >= 5:
            trim_and_save_clip(video_path, ts, output_dir, tag="headshot" if is_headshot else "highlight")
            log_clip_decision(video_path, ts, confidence_score, "Highlight")
        elif confidence_score >= 3:
            trim_and_save_clip(video_path, ts, maybe_dir, tag="maybe")
            log_clip_decision(video_path, ts, confidence_score, "Maybe")
        else:
            log_clip_decision(video_path, ts, confidence_score, "Ignored")
