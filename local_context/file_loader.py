
import os

def load_files_from_directory(directory, extensions={".py", ".json", ".md"}):
    file_data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_data[path] = f.read()
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return file_data
