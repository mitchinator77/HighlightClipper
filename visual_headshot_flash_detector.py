import cv2
import numpy as np

# Region around the crosshair (center of a 1920x1080 screen)
CROSSHAIR_REGION = (910, 490, 100, 100)  # (x, y, w, h)

def detect_visual_flash(frame, threshold=50):
    x, y, w, h = CROSSHAIR_REGION
    cropped = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Red hue can wrap around (0-10 and 160-180 in HSV)
    lower_red1 = np.array([0, 50, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 200])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    return cv2.countNonZero(mask) > threshold

def detect_headshot_flash(video_path, frame_skip=5, max_flash_count=10):
    cap = cv2.VideoCapture(str(video_path))
    flash_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            if detect_visual_flash(frame):
                flash_count += 1

        frame_idx += 1

    cap.release()

    # Normalize score if needed (optional)
    normalized_score = min(flash_count / max_flash_count, 1.0)
    return normalized_score
