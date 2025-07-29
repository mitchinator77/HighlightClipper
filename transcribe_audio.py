import os
import json
import subprocess
from pathlib import Path

def transcribe_with_whisper(input_path, output_path="Transcripts"):
    Path(output_path).mkdir(exist_ok=True)
    out_json = Path(output_path) / f"{Path(input_path).stem}.json"
    command = f"whisper "{input_path}" --model base --output_format json --output_dir {output_path}"
    print(f"🔤 Transcribing {input_path}...")
    subprocess.run(command, shell=True)
    return out_json