from __future__ import annotations

import json

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.config import PROCESSED_DIR


def _base_frame(chunks: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": [chunk["chunk_id"] for chunk in chunks],
            "document_id": [chunk["document_id"] for chunk in chunks],
            "title": [chunk["title"] for chunk in chunks],
            "category": [chunk["category"] for chunk in chunks],
            "text": [chunk["text"] for chunk in chunks],
        }
    )


def _projection_frame(chunks: list[dict], coordinates: np.ndarray, method: str) -> pd.DataFrame:
    frame = _base_frame(chunks)
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    frame["projection"] = method
    return frame


def build_projection() -> pd.DataFrame:
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    normalized_embeddings = normalize(embeddings)
    frames: list[pd.DataFrame] = []

    pca_coordinates = PCA(n_components=2, random_state=42).fit_transform(normalized_embeddings)
    frames.append(_projection_frame(chunks, pca_coordinates, "PCA"))

    svd_coordinates = TruncatedSVD(n_components=2, random_state=42).fit_transform(normalized_embeddings)
    frames.append(_projection_frame(chunks, svd_coordinates, "Truncated SVD"))

    if len(embeddings) >= 5:
        perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
        tsne_coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(normalized_embeddings)
        frames.append(_projection_frame(chunks, tsne_coordinates, "t-SNE"))

    if len(embeddings) >= 10:
        umap_coordinates = umap.UMAP(
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)
        frames.append(_projection_frame(chunks, umap_coordinates, "UMAP"))

    frame = pd.concat(frames, ignore_index=True)
    frame["projection"] = pd.Categorical(
        frame["projection"],
        categories=["UMAP", "PCA", "t-SNE", "Truncated SVD"],
        ordered=True,
    )
    frame = frame.sort_values(["projection", "chunk_id"]).reset_index(drop=True)
    frame.to_csv(PROCESSED_DIR / "projection.csv", index=False)
    return frame


if __name__ == "__main__":
    df = build_projection()
    methods = ", ".join(df["projection"].drop_duplicates().astype(str))
    print(f"Saved {methods} projections for {df['chunk_id'].nunique()} chunks")
