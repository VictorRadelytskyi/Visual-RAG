from __future__ import annotations

import json
import re
from pathlib import Path
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR, SOURCE_MANIFEST


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("main") or soup.find("article") or soup.body
    if article is None:
        return ""

    for tag in article.find_all(["script", "style", "nav", "footer", "aside", "figure"]):
        tag.decompose()

    lines: list[str] = []
    for tag in article.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) >= 20 or tag.name.startswith("h"):
            lines.append(text)
    return clean_text("\n\n".join(lines))


def download_documents() -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCE_MANIFEST.read_text())
    metadata: list[dict] = []

    headers = {"User-Agent": "Visual-RAG student project / educational use"}
    for source in tqdm(sources, desc="Downloading documents"):
        response = requests.get(source["url"], headers=headers, timeout=30)
        response.raise_for_status()
        text = extract_article_text(response.text)
        if len(text) < 300:
            raise ValueError(f"Extracted too little text from {source['url']}")

        path = RAW_DIR / f"{source['id']}.txt"
        path.write_text(text, encoding="utf-8")
        metadata.append({**source, "path": str(path.relative_to(RAW_DIR.parent)), "characters": len(text)})

    (RAW_DIR.parent / "raw_documents.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    docs = download_documents()
    print(f"Downloaded {len(docs)} documents to {RAW_DIR}")

