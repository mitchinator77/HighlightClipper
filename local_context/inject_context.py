
import json
import numpy as np

def load_embeddings(path="embeddings.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def find_similar_files(query, embedder, embeddings, top_k=5):
    query_embed = embedder.model.encode(query)
    scored = [(path, cosine_similarity(query_embed, emb)) for path, emb in embeddings.items()]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
