import logging
from utils.process_utils import convert_mkv_to_mp4_batch, run_pipeline_on_chunks

logging.basicConfig(level=logging.INFO, format="🌀 [%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

try:
    logging.info("🔄 Starting pipeline.")
    run_pipeline_on_chunks()
except Exception as e:
    logging.info(f"🚨 Pipeline crashed: {e}")