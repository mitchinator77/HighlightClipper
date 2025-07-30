import os
import subprocess
from moviepy.editor import VideoFileClip
from pathlib import Path
from event_utils import normalize_event_timestamps
from clip_scorer import compute_convergence_score

def extract_audio_ffmpeg(video_path, audio_path, duration=2.0):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
        str(audio_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_clip(clip_path, templates, trim_func, scorer,
                 logger_func, audio_peak_func, headshot_func, classifier_func):
    clip_name = clip_path.name
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"🎯 Processing: {clip_name}")

    # Detect killfeed
    visual_score, visual_timestamps = scorer(str(clip_path), templates)
    logger.info(f"🔍 Killfeed → Score: {visual_score} → Timestamps: {visual_timestamps}")

    # Extract audio waveform
    waveform, rate = extract_waveform(str(clip_path))
    if waveform.size > 0:
        audio_timestamps = audio_peak_func(waveform, rate)
        headshot_timestamps = headshot_func(waveform, rate)
    else:
        audio_timestamps = []
        headshot_timestamps = []

    # 🧠 Compute convergence score using RAW timestamps (not normalized)
    killfeed_times = visual_timestamps
    audio_times = audio_timestamps
    skull_times = headshot_timestamps  # You may rename this if needed

    convergence_bonus = compute_convergence_score(
        audio_times, [], skull_times, killfeed_times
    )
    logger.info(f"🧠 Convergence bonus: {convergence_bonus}")

    # Optionally still normalize for display/logging if needed
    events = normalize_event_timestamps({
        "visual": killfeed_times,
        "audio": audio_times,
        "headshot": skull_times
    })

    # Log and trim
    scored_moments = convergence_bonus  # Currently using bonus as score
    log_path = Path("logs") / f"{clip_path.stem}_scoring.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_func(scored_moments, events, log_path)
    trim_func(str(clip_path), scored_moments)

    # Run headshot classifier if enabled
    for ts in skull_times:
        try:
            label, confidence = classifier_func(str(clip_path))
            classify_and_log_audio(clip_path, label, ts, confidence)
        except Exception as e:
            logger.error(f"⚠️ Classifier failed on {clip_path.name}: {e}")

def extract_waveform(video_path):
    from moviepy.editor import AudioFileClip
    import numpy as np
    from scipy.io import wavfile
    import uuid

    try:
        temp_path = f"temp_{uuid.uuid4().hex}.wav"
        audio = AudioFileClip(video_path)
        audio.write_audiofile(temp_path, verbose=False, logger=None)
        rate, waveform = wavfile.read(temp_path)
        os.remove(temp_path)

        # 🔧 Handle mono or stereo safely
        if len(waveform.shape) == 2 and waveform.shape[1] == 2:
            waveform = waveform.mean(axis=1)
        elif len(waveform.shape) == 1:
            pass  # Already mono
        else:
            print(f"⚠️ Unexpected waveform shape: {waveform.shape}")

        # Normalize waveform
        if np.issubdtype(waveform.dtype, np.integer):
            waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
        elif np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float32)

        return waveform, rate

    except Exception as e:
        print(f"❌ Error loading audio waveform: {e}")
        return np.array([]), 44100
