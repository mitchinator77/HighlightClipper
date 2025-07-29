import json
import cv2
from pathlib import Path
from audio_headshot_detector import detect_headshot_audio
from visual_headshot_flash_detector import detect_headshot_flash
from visual_killfeed_detector import detect_killfeed, load_templates
from visual_skull_detector import detect_skull_kill, load_skull_templates

# Load config
config_path = Path("score_config.json")
if config_path.exists():
    with config_path.open() as f:
        config = json.load(f)
else:
    config = {
        "killfeed_weight": 1.0,
        "visual_flash_multiplier": 1.5,
        "audio_confidence_multiplier": 2.0,
        "skull_weight": 1.2,
        "threshold_clipworthy": 0.5
    }

# Load templates once
killfeed_templates = load_templates("killfeed_templates")
skull_templates = load_skull_templates("skull_templates")

def score_clip(video_path, visual_score, audio_score, transcript_score):
    score = 0

    # Killfeed detection
    killfeed_score = detect_killfeed(video_path, killfeed_templates)
    if killfeed_score > 0:
        print(f"🧠 Killfeed score: {killfeed_score}")
        score += min(killfeed_score, 3) * config["killfeed_weight"]

    # Visual headshot flash
    if visual_score > 0:
        print(f"🧠 Headshot flashes: {visual_score}")
        score += visual_score * config["visual_flash_multiplier"]

    # Audio-based detection
    if audio_score > 0.6:
        print(f"🧠 Audio headshot confidence: {audio_score:.2f}")
        score += audio_score * config["audio_confidence_multiplier"]

    # Skull detection
    skull_conf = detect_skull_kill(video_path, skull_templates)
    if skull_conf >= 0.7:
        print(f"💀 Skull detected! Confidence: {skull_conf}")
        score += skull_conf * config["skull_weight"]

    return round(score, 2)

def is_clipworthy(score):
    return score >= config["threshold_clipworthy"]

def score_all_clips():
    chunk_dir = Path("Chunks")
    highlight_dir = Path("Highlights")
    maybe_dir = Path("Maybes")
    highlight_dir.mkdir(exist_ok=True)
    maybe_dir.mkdir(exist_ok=True)

    chunked_videos = sorted(chunk_dir.glob("*.mp4"))
    clip_scores = {}

    for clip_path in chunked_videos:
        visual_score = detect_headshot_flash(clip_path)
        audio_score = detect_headshot_audio(clip_path)
        transcript_score = 0  # Future use

        final_score, skull_conf = score_clip(clip_path, visual_score, audio_score, transcript_score)
        is_clip = is_clipworthy(final_score)

        clip_scores[clip_path.name] = {
            "visual_headshot_score": visual_score,
            "audio_headshot_score": audio_score,
            "skull_confidence": skull_conf,
            "final_score": final_score,
            "clipworthy": is_clip
        }

        print(f"🎯 {clip_path.name} → Score: {final_score} → Clipworthy: {is_clip}")
        target_path = (highlight_dir if is_clip else maybe_dir) / clip_path.name
        clip_path.replace(target_path)

    output_path = Path("ClipScores") / "clip_scores.json"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        json.dump(clip_scores, f, indent=2)
