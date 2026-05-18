from __future__ import annotations

import json

from src.config import PROCESSED_DIR


def load_chunks() -> list[dict]:
    return json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))


def get_chunk(chunk_id: str) -> dict | None:
    return next((chunk for chunk in load_chunks() if chunk["chunk_id"] == chunk_id), None)

