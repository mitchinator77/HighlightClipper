
import os
import time
import logging
from utils.file_utils import prepare_video_chunks, cleanup_temp_files
from utils.process_utils import convert_mkv_to_mp4_batch, run_pipeline_on_chunks
from utils.git_utils import auto_commit_and_push_if_ready
from dotenv import load_dotenv

# Load secrets/config
load_dotenv()
CLIP_SCORE_THRESHOLD = float(os.getenv("CLIP_SCORE_THRESHOLD", 0.6))
CHUNK_LENGTH_MINUTES = int(os.getenv("CHUNK_LENGTH_MINUTES", 10))
ENABLE_AUTO_UPDATE = os.getenv("ENABLE_AUTO_UPDATE", "False").lower() == "true"

# Logging
logging.basicConfig(level=logging.INFO, format="🌀 [%(asctime)s] %(message)s")

def run():
    logging.info("🔄 Starting pipeline.")

    # Step 1: Convert MKV -> MP4 in batches
    converted_videos = convert_mkv_to_mp4_batch(chunk_size=3, delete_original=True)
    if not converted_videos:
        logging.warning("❌ No videos to convert. Exiting.")
        return

    # Step 2: Break into chunks
    chunks = prepare_video_chunks(converted_videos, CHUNK_LENGTH_MINUTES)

    # Step 3: Process in segments
    for idx, chunk_group in enumerate(chunks):
        logging.info(f"🎬 Processing chunk group {idx+1}/{len(chunks)}")
        run_pipeline_on_chunks(chunk_group, CLIP_SCORE_THRESHOLD)

        if ENABLE_AUTO_UPDATE and (idx + 1) % 3 == 0:
            logging.info("🧠 Triggering pipeline self-improvement update.")
            auto_commit_and_push_if_ready()

    # Step 4: Final cleanup
    cleanup_temp_files()
    logging.info("✅ All processing complete.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.error(f"🚨 Pipeline crashed: {e}")
