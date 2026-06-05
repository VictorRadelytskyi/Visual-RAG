from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, PROCESSED_DIR, RAW_DIR


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    metadata_path = RAW_DIR.parent / "raw_documents.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []

    for document in tqdm(metadata, desc="Building chunks"):
        raw_path = RAW_DIR.parent / document["path"]
        text = raw_path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_words(text, chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document['id']}_{index:03d}",
                    "document_id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "url": document["url"],
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Created {len(chunks)} chunks")

