
import cv2
import numpy as np
import os
import json
import wave
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import time
from tkinter import filedialog, messagebox
from moviepy.editor import AudioFileClip

class HybridLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("🔁 Headshot Labeler: With Controls Active")
        self.labels = []
        self.video_path = ""
        self.audio_path = ""
        self.current_frame_time = 0
        self.paused = False
        self.cap = None
        self.fps = 30
        self.slider_update = False

        self.canvas = tk.Canvas(self.root, width=640, height=360)
        self.canvas.pack()

        self.controls = tk.Frame(self.root)
        self.controls.pack()

        tk.Button(self.controls, text="🎥 Load Video", command=self.load_video).pack(side="left", padx=5)
        tk.Button(self.controls, text="💥 Mark Headshot", command=self.mark_headshot).pack(side="left", padx=10)
        tk.Button(self.controls, text="💾 Save Labels", command=self.save_labels).pack(side="left", padx=5)
        tk.Button(self.controls, text="▶ Play", command=self.play).pack(side="left", padx=5)
        tk.Button(self.controls, text="⏸ Pause", command=self.pause).pack(side="left", padx=5)
        tk.Button(self.controls, text="⏪ Rewind 5s", command=self.rewind).pack(side="left", padx=5)
        tk.Button(self.controls, text="⏩ Forward 5s", command=self.forward).pack(side="left", padx=5)

        self.slider = tk.Scale(self.root, from_=0, to=100, orient="horizontal", length=640, command=self.seek)
        self.slider.pack()

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack()
        self.fig, self.ax = plt.subplots(figsize=(6, 2), dpi=100)
        self.canvas_wave = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_wave.get_tk_widget().pack()

        self.root.bind("<h>", self.mark_headshot)
        self.root.bind("<q>", lambda e: self.root.quit())

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not path:
            return
        self.video_path = path
        self.audio_path = os.path.splitext(path)[0] + "_audio.wav"

        self.extract_audio()
        self.plot_waveform()
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = self.total_frames / self.fps
        self.slider.config(to=self.duration)
        self.play_video()

    def extract_audio(self):
        audio = AudioFileClip(self.video_path)
        audio.write_audiofile(self.audio_path, verbose=False, logger=None)

    def plot_waveform(self):
        with wave.open(self.audio_path, "rb") as wf:
            frames = wf.readframes(-1)
            samples = np.frombuffer(frames, dtype=np.int16)
            self.ax.clear()
            self.ax.plot(samples, linewidth=0.5)
            self.ax.set_title("Audio Waveform")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas_wave.draw()

    def play_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1.0 / fps
        start_time = time.time()

        def stream():
            current_time = time.time()
            elapsed = current_time - start_time
            expected_frame = int(elapsed * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)

            ret, frame = cap.read()
            if not ret:
                cap.release()
                return

            self.current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = cv2.resize(frame, (640, 360))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.photo = cv2.imencode('.ppm', img)[1].tobytes()
            self.imgtk = tk.PhotoImage(data=self.photo)
            self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)

            self.root.after(1, stream)  # Try again ASAP, let time drift guide us

        stream()

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def rewind(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_MSEC) - 5000
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0, current))

    def forward(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_MSEC) + 5000
            self.cap.set(cv2.CAP_PROP_POS_MSEC, min(current, self.duration * 1000))

    def seek(self, val):
        if self.cap and not self.slider_update:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, float(val) * 1000)

    def mark_headshot(self, event=None):
        ts = round(self.current_frame_time, 3)
        self.labels.append(ts)
        print(f"💥 Headshot marked at {ts}s")

    def save_labels(self):
        if not self.labels or not self.video_path:
            messagebox.showwarning("Nothing to Save", "No labels or video loaded.")
            return

        os.makedirs("headshot_labels", exist_ok=True)
        out_path = os.path.join(
            "headshot_labels",
            os.path.splitext(os.path.basename(self.video_path))[0] + "_headshots.json"
        )
        with open(out_path, "w") as f:
            json.dump(self.labels, f, indent=2)

        messagebox.showinfo("Saved", f"Saved {len(self.labels)} headshots to:\n{out_path}")
        print("✅ Labels saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridLabeler(root)
    root.mainloop()
