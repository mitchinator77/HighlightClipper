import json
from pathlib import Path
from audio_headshot_detector import detect_headshot_audio
from visual_headshot_flash_detector import detect_headshot_flash
from visual_killfeed_detector import detect_killfeed

def score_clip(video_path, visual_score, audio_score, transcript_score):
    score = 0

    if detect_killfeed(video_path):
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

# Process all clips in Chunks directory
chunk_dir = Path("Chunks")
chunked_videos = sorted(chunk_dir.glob("*.mp4"))
clip_scores = {}

for clip_path in chunked_videos:
    visual_score = detect_headshot_flash(clip_path)
    audio_score = detect_headshot_audio(clip_path)
    transcript_score = 0  # Placeholder

    final_score = score_clip(clip_path, visual_score, audio_score, transcript_score)
    clip_scores[clip_path.name] = {
        "visual_headshot_score": visual_score,
        "audio_headshot_score": audio_score,
        "final_score": final_score,
        "clipworthy": is_clipworthy(final_score),
    }

    print(f"🎯 {clip_path.name} → Score: {final_score} → Clipworthy: {is_clipworthy(final_score)}")

# Save results
output_path = Path("ClipScores") / "clip_scores.json"
output_path.parent.mkdir(exist_ok=True)
with output_path.open("w") as f:
    json.dump(clip_scores, f, indent=2)
