
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_files(self, file_dict):
        return {path: self.model.encode(content) for path, content in file_dict.items()}

    def save_embeddings(self, embeddings, out_path="embeddings.json"):
        data = {k: v.tolist() for k, v in embeddings.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
