
import cv2
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

def is_static_frame(frame1, frame2, threshold=0.95):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

def extract_useful_segments(video_path, output_dir, frame_sample_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    prev_frame = None
    segment_frames = []
    segment_count = 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % (frame_rate * frame_sample_rate) == 0:
            if prev_frame is not None and not is_static_frame(prev_frame, frame):
                segment_frames.append(frame_num)
            prev_frame = frame
        frame_num += 1
    cap.release()
    return segment_frames
