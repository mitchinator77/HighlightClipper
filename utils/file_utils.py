# utils/file_utils.py
import shutil
import os

def cleanup_temp_files(temp_dirs=None):
    if temp_dirs is None:
        temp_dirs = ['Temp', 'TempAudio']

    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Deleted temp folder: {temp_dir}")