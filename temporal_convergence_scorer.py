# temporal_convergence_scorer.py

from typing import List, Dict, Tuple
import math

# Type alias
Timestamp = float  # seconds

def decay_weight(delta: float, decay_rate: float = 0.8, threshold: float = 2.0) -> float:
    """
    Apply an exponential decay to the score if timestamps are not close.
    """
    if abs(delta) > threshold:
        return decay_rate ** abs(delta - threshold)
    return 1.0

def compute_convergence_score(events: Dict[str, List[Timestamp]]) -> List[Tuple[float, float]]:
    """
    Computes convergence scores based on multiple event timelines.
    Returns a list of tuples: (timestamp, score).
    """
    all_timestamps = sorted(set(events.get("visual", []) + events.get("audio", []) + events.get("headshot", [])))
    scored_moments = []

    for t in all_timestamps:
        score = 0.0

        # Score for visual event
        visual_weight = max(decay_weight(t - tv) for tv in events.get("visual", [])) if events.get("visual") else 0
        score += 1.0 * visual_weight

        # Score for audio spike
        audio_weight = max(decay_weight(t - ta) for ta in events.get("audio", [])) if events.get("audio") else 0
        score += 0.8 * audio_weight

        # Score for headshot ding
        headshot_weight = max(decay_weight(t - th) for th in events.get("headshot", [])) if events.get("headshot") else 0
        score += 1.2 * headshot_weight

        # Normalize score (optionally clip or filter low scores)
        if score >= 1.0:
            scored_moments.append((t, round(score, 3)))

    return scored_moments