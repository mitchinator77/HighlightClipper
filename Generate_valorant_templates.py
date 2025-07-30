from PIL import Image
from pathlib import Path

# Input screenshots folder (same as script location)
input_dir = Path(".")
output_dir = Path("hud_templates/valorant")
output_dir.mkdir(parents=True, exist_ok=True)

# Define regions: {output_name: (screenshot_name, (left, top, right, bottom))}
regions = {
    "bottom_hud_abilities.png": ("screenshot_1m45s_standard.png", (700, 960, 1260, 1070)),
    "top_mid_roundtimer.png": ("screenshot_1m45s_standard.png", (610, 20, 750, 70)),
    "bottom_right_ammo.png": ("screenshot_1m45s_standard.png", (1200, 930, 1370, 1060)),

    "scoreboard_top_strip.png": ("screenshot_70s_scoreboard.png", (500, 0, 1300, 100)),
    "scoreboard_full_mid.png": ("screenshot_70s_scoreboard.png", (400, 100, 960, 600)),

    "buyphase_banner.png": ("screenshot_59s_roundstartbuybutton.png", (580, 60, 780, 160)),
}

# Crop and save
for name, (img_file, box) in regions.items():
    try:
        img_path = input_dir / img_file
        with Image.open(img_path) as im:
            cropped = im.crop(box)
            cropped.save(output_dir / name)
            print(f"✅ Saved: {name}")
    except Exception as e:
        print(f"❌ Failed: {name} from {img_file} — {e}")

print(f"\n🎯 All templates saved to: {output_dir.resolve()}")
