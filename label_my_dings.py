import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import wave
import json
import os

class HeadshotLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔫 Headshot Audio Labeler")
        self.geometry("400x200")
        self.configure(bg="#222222")

        self.file_path = ""
        self.labels = []

        self.create_widgets()
        pygame.mixer.init()

    def create_widgets(self):
        self.load_btn = tk.Button(self, text="🎧 Load WAV File", command=self.load_file, bg="#3498db", fg="white", font=("Arial", 12))
        self.load_btn.pack(pady=10)

        self.play_btn = tk.Button(self, text="▶ Play / Pause", command=self.toggle_play, bg="#2ecc71", fg="white", font=("Arial", 12))
        self.play_btn.pack(pady=10)

        self.label_btn = tk.Button(self, text="💥 Mark Headshot Timestamp", command=self.mark_timestamp, bg="#e74c3c", fg="white", font=("Arial", 12))
        self.label_btn.pack(pady=10)

        self.save_btn = tk.Button(self, text="💾 Save Labels", command=self.save_labels, bg="#f39c12", fg="black", font=("Arial", 12))
        self.save_btn.pack(pady=10)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self.file_path = path
        self.labels = []
        pygame.mixer.music.load(self.file_path)
        pygame.mixer.music.play()
        self.log("Loaded and playing: " + os.path.basename(self.file_path))

    def toggle_play(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    def mark_timestamp(self):
        ms = pygame.mixer.music.get_pos()
        if ms > 0:
            seconds = round(ms / 1000.0, 3)
            self.labels.append(seconds)
            self.log(f"💥 Marked timestamp: {seconds}s")
        else:
            self.log("⚠️ Audio not playing")

    def save_labels(self):
        if not self.file_path or not self.labels:
            messagebox.showwarning("Nothing to Save", "Load a file and mark at least one timestamp.")
            return
        out_path = os.path.splitext(self.file_path)[0] + "_headshots.json"
        with open(out_path, "w") as f:
            json.dump(self.labels, f, indent=2)
        messagebox.showinfo("Saved", f"Saved {len(self.labels)} timestamps to:
{out_path}")
        self.log("✅ Labels saved.")

    def log(self, msg):
        print(msg)

if __name__ == "__main__":
    app = HeadshotLabeler()
    app.mainloop()