"""Runs all scoring modules and creates a clip report."""
from clip_scorer import score_clip, is_clipworthy
from audio_event_detector import detect_audio_spikes
from transcript_analyzer import score_transcript
from visual_killfeed_detector import score_killfeed_presence

def process_clip(clip_path, audio_path, transcript_path):
    visual_score = score_killfeed_presence(clip_path)
    audio_score = min(detect_audio_spikes(audio_path) / 5.0, 1.0)
    transcript_score = score_transcript(transcript_path)

    final_score = score_clip(visual_score, audio_score, transcript_score)
    worthy = is_clipworthy(final_score)

    return {
        'visual': visual_score,
        'audio': audio_score,
        'transcript': transcript_score,
        'final_score': final_score,
        'clipworthy': worthy
    }