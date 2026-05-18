from __future__ import annotations

import json

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from src.config import PROCESSED_DIR


def build_projection() -> pd.DataFrame:
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")

    if len(embeddings) >= 10:
        reducer = umap.UMAP(n_neighbors=min(15, len(embeddings) - 1), min_dist=0.15, metric="cosine", random_state=42)
        coordinates = reducer.fit_transform(embeddings)
        method = "UMAP"
    else:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        method = "PCA"

    frame = pd.DataFrame(
        {
            "chunk_id": [chunk["chunk_id"] for chunk in chunks],
            "document_id": [chunk["document_id"] for chunk in chunks],
            "title": [chunk["title"] for chunk in chunks],
            "category": [chunk["category"] for chunk in chunks],
            "text": [chunk["text"] for chunk in chunks],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "projection": method,
        }
    )
    frame.to_csv(PROCESSED_DIR / "projection.csv", index=False)
    return frame


if __name__ == "__main__":
    df = build_projection()
    print(f"Saved {df['projection'].iloc[0]} projection for {len(df)} chunks")

