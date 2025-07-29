import cv2, os, glob, json
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

CHUNKS_FOLDER = "Chunks"
TEMPLATE_PATH = "assets/valorant_killfeed_template.png"
RESULTS_FOLDER = "DetectedKillfeeds"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

FRAME_SKIP_SEC = 1.0
NUM_WORKERS = 4

# Only top-right 25% of frame
ROI_Y, ROI_X, ROI_H, ROI_W = 0, 0.75, 0.25, 0.25

def detect_killfeed_in_frame(frame, template, thresh=0.85):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return (res >= thresh).any()

def detect_killfeed_for_video(video_path):
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"❌ Could not load template at {TEMPLATE_PATH}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip_interval = int(fps * FRAME_SKIP_SEC)
    detected = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_num % skip_interval == 0:
            h, w = frame.shape[:2]
            roi = frame[int(h * ROI_Y):int(h * (ROI_Y + ROI_H)), int(w * ROI_X):int(w * (ROI_X + ROI_W))]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if detect_killfeed_in_frame(gray, template):
                timestamp = str(timedelta(seconds=frame_num / fps))
                detected.append(timestamp)
        frame_num += 1
    cap.release()

    result = {"video": Path(video_path).name, "killfeed_timestamps": detected}
    fname = Path(RESULTS_FOLDER) / (Path(video_path).stem + "_killfeed.json")
    with open(fname, "w") as f: json.dump(result, f, indent=2)

def detect_killfeed_events():
    mp4_files = glob.glob(os.path.join(CHUNKS_FOLDER, "*.mp4"))
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(detect_killfeed_for_video, mp4_files)