
import cv2
import numpy as np

# Assume crosshair is roughly center of 1920x1080
CROSSHAIR_REGION = (910, 490, 100, 100)

def detect_visual_flash(frame):
    x, y, w, h = CROSSHAIR_REGION
    cropped = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # Red flash: high saturation + high value + reddish hue
    lower_red = np.array([0, 50, 200])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.countNonZero(mask) > 50
