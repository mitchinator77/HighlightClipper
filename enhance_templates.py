import cv2
import os
from pathlib import Path

template_dir = Path("killfeed_templates")
cleaned_dir = Path("cleaned_templates")
cleaned_dir.mkdir(exist_ok=True)

def enhance_and_resize_template(image_path, target_size=(100, 100)):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, target_size)
    return resized

for file in template_dir.iterdir():
    if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        processed = enhance_and_resize_template(file)
        if processed is not None:
            output_path = cleaned_dir / file.name
            cv2.imwrite(str(output_path), processed)
            print(f"✅ Saved: {output_path.name}")
