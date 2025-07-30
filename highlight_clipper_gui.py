import tkinter as tk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.widgets import Button
import subprocess
import threading
import os
import webbrowser

class HighlightClipperUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎮 HighlightClipper Control Panel")
        self.geometry("800x600")
        self.style = Style("darkly")
        self.configure(bg=self.style.colors.bg)

        self.create_widgets()

    def create_widgets(self):
        frm = tk.Frame(self, bg=self.style.colors.bg)
        frm.pack(pady=20)

        self.run_btn = Button(frm, text="▶ Run HighlightClipper", command=self.run_pipeline, bootstyle=SUCCESS)
        self.run_btn.pack(padx=10, pady=5)

        self.open_btn = Button(frm, text="📂 Open Highlights Folder", command=self.open_highlights, bootstyle=PRIMARY)
        self.open_btn.pack(padx=10, pady=5)

        self.log_box = ScrolledText(self, wrap=tk.WORD, font=("Consolas", 10), height=25)
        self.log_box.pack(padx=20, pady=10, fill="both", expand=True)

    def run_pipeline(self):
        def task():
            self.log("Running pipeline...\n")
            try:
                process = subprocess.Popen(["python", "run_all.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in iter(process.stdout.readline, b''):
                    decoded = line.decode("utf-8", errors="replace")
                    self.log(decoded)
                process.stdout.close()
                process.wait()
                self.log("✅ Pipeline completed.\n")
            except Exception as e:
                self.log(f"❌ Error: {e}\n")

        threading.Thread(target=task, daemon=True).start()

    def log(self, message):
        self.log_box.insert(tk.END, message)
        self.log_box.see(tk.END)

    def open_highlights(self):
        path = os.path.abspath("Highlights")
        if os.path.exists(path):
            webbrowser.open(path)
        else:
            self.log("⚠️ Highlights folder not found. Run the pipeline first.\n")

if __name__ == "__main__":
    app = HighlightClipperUI()
    app.mainloop()