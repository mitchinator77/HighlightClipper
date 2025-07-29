from constants.paths import SOURCE_VIDEOS_DIR, CHUNKED_VIDEOS_DIR
from processing.mkv_converter import convert_mkv_to_mp4_batch

def run_pipeline_on_chunks():
    print("[DEBUG] Running full pipeline")
    chunk_size = 3
    delete_original = True
    convert_mkv_to_mp4_batch(SOURCE_VIDEOS_DIR, CHUNKED_VIDEOS_DIR, chunk_size, delete_original)