import os
import subprocess

def convert_mkv_to_mp4_batch(source_dir, output_dir, chunk_size=3, delete_original=True):
    print(f"[DEBUG] Running MKV to MP4 batch conversion with chunk size: {chunk_size}")
    os.makedirs(output_dir, exist_ok=True)
    converted_files = []
    for filename in os.listdir(source_dir):
        if filename.endswith('.mkv'):
            input_path = os.path.join(source_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.mp4'
            output_path = os.path.join(output_dir, output_filename)
            cmd = ["ffmpeg", "-i", input_path, "-codec", "copy", output_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if delete_original:
                os.remove(input_path)
            converted_files.append(output_path)
    return converted_files