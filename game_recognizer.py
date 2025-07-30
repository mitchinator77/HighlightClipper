import os
import json
import joblib
import numpy as np
import ffmpeg
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Load trained classifier
clf = joblib.load("game_classifier.joblib")
FEATURE_CACHE_FILE = "logs/feature_cache.json"

def extract_center_frame_ffmpeg(filepath):
    """
    Extracts the center frame of the video using ffmpeg as a grayscale 64x64 flattened numpy array.
    """
    try:
        # Get total number of frames
        probe = ffmpeg.probe(filepath)
        num_frames = int(probe['streams'][0]['nb_frames'])
        center_frame = num_frames // 2

        # Extract frame using ffmpeg
        out, _ = (
            ffmpeg
            .input(filepath)
            .filter('select', f'eq(n\,{center_frame})')
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='gray')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        # Convert byte data to numpy array
        frame = np.frombuffer(out, np.uint8)
        frame = frame.reshape((-1,))  # Flatten to 1D
        return frame
    except Exception as e:
        print(f"⚠️ Failed to extract frame from {filepath}: {e}")
        return None

def extract_features(filepath):
    return extract_center_frame_ffmpeg(filepath)

def load_cache():
    if os.path.exists(FEATURE_CACHE_FILE):
        with open(FEATURE_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(FEATURE_CACHE_FILE, "w") as f:
        json.dump(cache, f)

def classify_chunk(filepath):
    cache_key = os.path.basename(filepath)
    features = extract_features(filepath)
    if features is None:
        return cache_key, "unknown"

    prediction = clf.predict([features])[0]
    return cache_key, prediction

def classify_chunks_by_game_parallel(chunk_folder):
    """
    Classify all chunks using parallel processing and cached features.
    """
    print("⚙️ Using multiprocessing game classifier with ffmpeg & cache")

    chunk_paths = [
        os.path.join(chunk_folder, f)
        for f in os.listdir(chunk_folder)
        if f.endswith(".mp4")
    ]

    cache = load_cache()
    chunk_game_map = {}

    # Split chunks into cached and uncached
    uncached = [p for p in chunk_paths if os.path.basename(p) not in cache]

    if uncached:
        print(f"🧠 Extracting features for {len(uncached)} new chunks...")
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(classify_chunk, uncached), total=len(uncached)))

        for chunk_name, prediction in results:
            cache[chunk_name] = prediction

        save_cache(cache)

    # Combine all predictions (cached + new)
    for path in chunk_paths:
        chunk_name = os.path.basename(path)
        chunk_game_map[chunk_name] = cache.get(chunk_name, "unknown")

    return chunk_game_map
