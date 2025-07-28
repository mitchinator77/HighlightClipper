
import shutil
from pathlib import Path

TEMP_DIRS = ["TempAudio", "Storyboard", "Temp"]
for folder in TEMP_DIRS:
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
        print(f"Cleaned {folder}")
