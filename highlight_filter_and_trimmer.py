import cv2
import os
from moviepy.editor import VideoFileClip
from pathlib import Path
from scoring_logger import log_clip_decision

SPECTATOR_ICON_PATH = "templates/spectator/spectator_icon.png"
SPECTATOR_REGION = (20, 950, 200, 1060)  # (y1, y2, x1, x2)

def is_spectating(frame):
    if not os.path.exists(SPECTATOR_ICON_PATH):
        print("⚠️ Spectator icon template not found.")
        return False

    y1, y2, x1, x2 = SPECTATOR_REGION
    cropped = frame[y1:y2, x1:x2]
    icon_template = cv2.imread(SPECTATOR_ICON_PATH, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, icon_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    print(f"👁️ Spectator icon match score: {max_val:.2f}")
    return max_val > 0.75

def trim_and_save_clip(source_video, start_time, end_time, out_path):
    try:
        clip = VideoFileClip(source_video).subclip(start_time, end_time)
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    except Exception as e:
        print(f"❌ Error trimming clip: {e}")

def trim_highlights(detected_highlights, output_dir="Highlights", headshot_timestamps=None):
    os.makedirs(output_dir, exist_ok=True)
    for clip_path, info in detected_highlights.items():
        video_path = info['source']
        start = info['start']
        end = info['end']
        clip_name = Path(clip_path).stem
        save_path = os.path.join(output_dir, f"{clip_name}_highlight.mp4")

        # Spectator Check
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        ret, frame = cap.read()
        cap.release()

        if ret and is_spectating(frame):
            print(f"🚫 Skipping {clip_path} - player is spectating.")
            log_clip_decision(clip_path, "rejected", "player is spectating")
            continue

        # Headshot confidence (if applicable)
        headshot_tag = False
        if headshot_timestamps:
            for t in headshot_timestamps:
                if start <= t <= end:
                    headshot_tag = True
                    break
        reason = "contains headshot" if headshot_tag else "generic highlight"
        log_clip_decision(clip_path, "accepted", reason)

        trim_and_save_clip(video_path, start, end, save_path)
        print(f"✅ Saved: {save_path}")
