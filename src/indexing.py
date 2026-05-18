from __future__ import annotations

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, PROCESSED_DIR


def build_index(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(PROCESSED_DIR / "faiss.index"))
    np.save(PROCESSED_DIR / "embeddings.npy", embeddings)
    (PROCESSED_DIR / "model_name.txt").write_text(model_name, encoding="utf-8")


if __name__ == "__main__":
    build_index()
    print("Built FAISS index")

