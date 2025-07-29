
import json
from pathlib import Path
from audio_headshot_detector import detect_headshot_audio
from visual_headshot_flash_detector import detect_headshot_flash
from visual_skull_detector import detect_skull_kill, load_skull_templates
from killfeed_detector import detect_killfeed, load_templates

config_path = Path("score_config.json")
if config_path.exists():
    with config_path.open() as f:
        config = json.load(f)
config_path = Path("score_config.json")
default_config = {
    "killfeed_weight": 1.0,
    "visual_flash_multiplier": 1.5,
    "audio_confidence_multiplier": 2.0,
    "skull_weight": 1.2,
    "threshold_clipworthy": 0.5
}

if config_path.exists():
    with config_path.open() as f:
        config = json.load(f)
else:
    config = default_config
    with config_path.open("w") as f:
        json.dump(default_config, f, indent=2)


def compute_convergence_score(audio_times, visual_times, skull_times, killfeed_times, window=2.0):
    all_times = sorted(set(audio_times + visual_times + skull_times + killfeed_times))
    bonus = 0
    for t in all_times:
        nearby = sum([
            any(abs(t - at) <= window for at in audio_times),
            any(abs(t - vt) <= window for vt in visual_times),
            any(abs(t - st) <= window for st in skull_times),
            any(abs(t - kt) <= window for kt in killfeed_times),
        ])
        if nearby >= 3:
            bonus += 1
    return bonus

skull_templates = load_skull_templates("skull_templates")
killfeed_templates = load_templates("killfeed_templates")

def score_clip(video_path, audio_result, visual_result, skull_result, killfeed_result):
    audio_score, audio_times = audio_result
    visual_score, visual_times = visual_result
    skull_score, skull_times = skull_result
    killfeed_score, killfeed_times = killfeed_result

    score = 0
    score += audio_score * config["audio_confidence_multiplier"]
    score += visual_score * config["visual_flash_multiplier"]
    score += skull_score * config["skull_weight"]
    score += min(killfeed_score, 3) * config["killfeed_weight"]

    bonus = compute_convergence_score(audio_times, visual_times, skull_times, killfeed_times)
    print(f"🧠 Convergence bonus: {bonus}")
    score += bonus

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
        visual_result = detect_headshot_flash(clip_path)
        audio_result = detect_headshot_audio(clip_path)
        skull_result = detect_skull_kill(clip_path, skull_templates)
        killfeed_result = detect_killfeed(clip_path, killfeed_templates)

        final_score = score_clip(clip_path, audio_result, visual_result, skull_result, killfeed_result)
        is_clip = is_clipworthy(final_score)

        clip_scores[clip_path.name] = {
            "visual_score": visual_result[0],
            "audio_score": audio_result[0],
            "skull_score": skull_result[0],
            "killfeed_score": killfeed_result[0],
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
