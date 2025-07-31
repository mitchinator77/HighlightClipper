import os
import json

def generate_storyboard(transcript_path, highlights_path):
    with open(transcript_path, "r") as t:
        transcript = t.read()

    with open(highlights_path, "r") as h:
        highlights = json.load(h)

    storyboard = []
    for highlight in highlights:
        timestamp = highlight["timestamp"]
        clip_summary = f"At {timestamp}, highlight occurred: {highlight.get('type', 'event')}"
        storyboard.append({"timestamp": timestamp, "summary": clip_summary})

    return storyboard
