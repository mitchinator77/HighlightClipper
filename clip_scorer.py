import json
import shutil
import cv2
from pathlib import Path
from audio_headshot_detector import detect_headshot_audio
from visual_headshot_flash_detector import detect_headshot_flash
from visual_killfeed_detector import detect_killfeed, load_templates

template_dir = Path("killfeed_templates")
killfeed_templates = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in template_dir.glob("*.png") if p.suffix == ".png"]

def score_clip(video_path, visual_score, audio_score, transcript_score):
    score = 0

    if detect_killfeed(video_path, killfeed_templates):
        print("🧠 Killfeed detected ✅")
        score += 1.0

    if visual_score > 0:
        print(f"🧠 Headshot flashes: {visual_score}")
        score += visual_score * 1.5

    if audio_score > 0.6:
        print(f"🧠 Audio headshot confidence: {audio_score:.2f}")
        score += audio_score * 2

    return round(score, 2)

def is_clipworthy(score, threshold=0.5):
    return score >= threshold

def score_all_clips():
    chunk_dir = Path("Chunks")
    highlight_dir = Path("Highlights")
    highlight_dir.mkdir(exist_ok=True)
    chunked_videos = sorted(chunk_dir.glob("*.mp4"))
    clip_scores = {}

    for clip_path in chunked_videos:
        visual_score = detect_headshot_flash(clip_path)
        audio_score = detect_headshot_audio(clip_path)
        transcript_score = 0  # Placeholder for now

        final_score = score_clip(clip_path, visual_score, audio_score, transcript_score)
        clipworthy = is_clipworthy(final_score)

        clip_scores[clip_path.name] = {
            "visual_headshot_score": visual_score,
            "audio_headshot_score": audio_score,
            "final_score": final_score,
            "clipworthy": clipworthy,
        }

        print(f"🎯 {clip_path.name} → Score: {final_score} → Clipworthy: {clipworthy}")

        # ✅ Copy clip to Highlights if clipworthy
        if clipworthy:
            highlight_path = highlight_dir / clip_path.name
            shutil.copy(clip_path, highlight_path)
            print(f"📥 Saved to Highlights: {highlight_path.name}")

    # Save results
    output_path = Path("ClipScores") / "clip_scores.json"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        json.dump(clip_scores, f, indent=2)

