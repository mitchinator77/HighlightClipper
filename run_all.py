import subprocess
import os
import sys

def run_script(script_name):
    print(f"▶ Running: {script_name}")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script_name}:\n{result.stderr}")

if __name__ == "__main__":
    if os.path.exists("update.py"):
        run_script("update.py")
    else:
        print("⏭ Skipping update: update.py not found.")

    run_script("extract_highlights.py")

    if os.path.exists("tune_config.py"):
        run_script("tune_config.py")
    else:
        print("⏭ Skipping tuning: tune_config.py not found.")