import os
import subprocess

def update_repo():
    try:
        # Ensure git is available
        subprocess.run(["git", "--version"], check=True)

        print("🔄 Pulling latest updates from GitHub...")
        result = subprocess.run(["git", "pull"], check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✅ Update complete.")
    except subprocess.CalledProcessError as e:
        print("❌ Git pull failed:", e.stderr)

if __name__ == "__main__":
    update_repo()