import os
from moviepy.editor import VideoFileClip

def trim_highlights(video_path: str, scored_moments: list, output_dir="Highlights", pre=10, post=10, score_threshold=2.5):
    """
    Saves clips around high-score events (+/- X seconds).
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the main video file
    video = VideoFileClip(video_path)

    try:
        for i, (timestamp, score) in enumerate(scored_moments):
            if score < score_threshold:
                continue

            start = max(0, timestamp - pre)
            end = min(video.duration, timestamp + post)

            subclip = video.subclip(start, end)
            out_path = os.path.join(output_dir, f"{base_name}_highlight_{i+1}.mp4")
            subclip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)

            # Explicitly close subclip to free resources
            subclip.close()
    finally:
        # Always close the main video to prevent file lock
        video.close()
