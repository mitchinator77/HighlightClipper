import cv2
import numpy as np
from pathlib import Path

# Parameters
FRAME_WINDOW = 15  # how many frames around the spike to check
EVENT_ZONE = (0.75, 0.0, 1.0, 0.3)  # top-right corner of screen (x1, y1, x2, y2)

def has_visual_event(video_path, spike_time, frame_window=FRAME_WINDOW):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define zone to scan
    x1, y1, x2, y2 = [int(a * b) for a, b in zip(EVENT_ZONE, (width, height, width, height))]

    center_frame = int(spike_time * fps)
    frames_to_check = range(center_frame - frame_window, center_frame + frame_window)

    baseline = None
    visual_change_detected = False

    for frame_num in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_num, 0))
        ret, frame = cap.read()
        if not ret:
            continue

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if baseline is None:
            baseline = blur
            continue

        diff = cv2.absdiff(baseline, blur)
        score = np.sum(diff)

        if score > 500000:  # threshold for visual change
            visual_change_detected = True
            break

    cap.release()
    return visual_change_detected