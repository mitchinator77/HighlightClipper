import cv2
import os
from pathlib import Path
import numpy as np

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    normalized = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def enhance_templates(input_root="templates", output_root="cleaned_templates"):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for category_dir in input_root.iterdir():
        if not category_dir.is_dir():
            continue

        output_category_dir = output_root / category_dir.name
        output_category_dir.mkdir(parents=True, exist_ok=True)

        for img_path in category_dir.glob("*.png"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"⚠️ Could not read: {img_path.name}")
                    continue

                enhanced = enhance_image(image)
                output_name = f"{img_path.stem}_{category_dir.name}.png"
                out_path = output_category_dir / output_name
                cv2.imwrite(str(out_path), enhanced)
                print(f"✅ Saved: {out_path.name}")
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    enhance_templates()
