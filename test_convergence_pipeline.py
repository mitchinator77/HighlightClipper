# test_convergence_pipeline.py

from temporal_convergence_scorer import compute_convergence_score
from event_utils import normalize_event_timestamps
from scoring_logger import log_scores_to_file

# Example mocked detection data
events = {
    "visual": [10.1, 25.3, 40.0],
    "audio": [10.4, 25.5, 41.2],
    "headshot": [10.3, 40.5]
}

events = normalize_event_timestamps(events)
scores = compute_convergence_score(events)
log_scores_to_file(scores, events, "clip_scoring_log.json")

print("Test complete. Scored highlights:")
for ts, score in scores:
    print(f"  -> {ts:.2f}s  | Score: {score}")