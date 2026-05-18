from __future__ import annotations

import json

import faiss
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DIR


class Retriever:
    def __init__(self) -> None:
        self.chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(PROCESSED_DIR / "faiss.index"))
        self.model_name = (PROCESSED_DIR / "model_name.txt").read_text(encoding="utf-8").strip()
        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index in zip(scores[0], indices[0], strict=True):
            chunk = self.chunks[int(index)].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

