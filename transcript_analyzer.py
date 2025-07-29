"""Parses transcript for clip-worthy keywords."""
HYPE_WORDS = {'ace', 'clutch', 'insane', 'crazy', 'no way', 'quad', 'one tap'}

def score_transcript(path):
    try:
        with open(path, 'r') as f:
            text = f.read().lower()
        hits = sum(1 for word in HYPE_WORDS if word in text)
        return min(hits / len(HYPE_WORDS), 1.0)
    except:
        return 0.0