# event_utils.py

from typing import Dict, List

def normalize_event_timestamps(events: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Ensures all event lists contain floats rounded to 3 decimals.
    """
    return {
        k: sorted(round(float(ts), 3) for ts in v)
        for k, v in events.items()
    }