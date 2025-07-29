import cv2
from pathlib import Path
from audio_headshot_detector import detect_headshot_audio

def extract_headshot_clips(input_dir="Chunks", output_dir="HeadshotClips", buffer=2.0, chain_window=10.0):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for video_file in input_path.glob("*.mp4"):
        score, timestamps = detect_headshot_audio(video_file)
        if not timestamps:
            continue

        # Chain timestamps
        chains = []
        current_chain = [timestamps[0]]
        for t in timestamps[1:]:
            if t - current_chain[-1] <= chain_window:
                current_chain.append(t)
            else:
                chains.append(current_chain)
                current_chain = [t]
        chains.append(current_chain)

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for i, chain in enumerate(chains):
            start = max(min(chain) - buffer, 0)
            end = max(chain) + buffer
            out_name = output_path / f"{video_file.stem}_headshot_{i}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            out = cv2.VideoWriter(str(out_name), fourcc, fps,
                                  (int(cap.get(3)), int(cap.get(4))))

            while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            out.release()
        cap.release()