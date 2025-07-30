
# Patch for detect_headshot_audio_peaks to save headshot timestamps in a JSON file

import os
import json

def save_headshot_timestamps(timestamps_dict, output_file="Detected/headshot_audio_events.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(timestamps_dict, f, indent=2)
    print(f"✅ Headshot timestamps saved to {output_file}")

# Example usage:
# save_headshot_timestamps({
#     "video1.mp4": [12.4, 88.2],
#     "video2.mp4": [33.3]
# })
