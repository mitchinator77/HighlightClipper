import cv2
import numpy as np

SPECTATOR_TEMPLATE_PATH = "templates/hud_elements/spectator_overlay.png"

def is_spectator_present(video_path, timestamp_sec, roi=(50, 840, 300, 100), threshold=0.75):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        return False

    x, y, w, h = roi
    cropped = frame[y:y+h, x:x+w]
    gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(SPECTATOR_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

    if template is None or gray_frame is None:
        return False

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val > threshold
