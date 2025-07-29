import os
import cv2
from moviepy.editor import VideoFileClip

def chunk_video(video_path, chunk_length_minutes=10, output_dir='TempChunks'):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)
    duration = video.duration  # in seconds
    chunk_length = chunk_length_minutes * 60
    basename = os.path.splitext(os.path.basename(video_path))[0]

    chunks = []
    for i in range(0, int(duration), chunk_length):
        start = i
        end = min(i + chunk_length, duration)
        chunk_filename = f"{basename}_chunk_{start}-{end}.mp4"
        output_path = os.path.join(output_dir, chunk_filename)
        video.subclip(start, end).write_videofile(output_path, codec="libx264", audio_codec="aac")
        chunks.append(output_path)
    video.close()
    return chunks
