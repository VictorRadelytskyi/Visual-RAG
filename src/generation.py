from __future__ import annotations

import os

from openai import OpenAI


def azure_openai_is_configured() -> bool:
    return all(
        os.getenv(name)
        for name in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_BASE_URL",
            "AZURE_OPENAI_DEPLOYMENT",
        )
    )


def generate_answer(query: str, chunks: list[dict]) -> str | None:
    if not azure_openai_is_configured():
        return None

    client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_BASE_URL"].rstrip("/") + "/",
    )
    context = "\n\n".join(
        f"[{index}] {chunk['title']} ({chunk['chunk_id']}):\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    response = client.responses.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        input=[
            {
                "role": "system",
                "content": (
                    "You answer questions only from the provided context. "
                    "If the context is insufficient, say that the retrieved documents do not contain enough information. "
                    "Keep the answer concise and cite supporting chunks in square brackets, for example [1] or [2]."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nContext:\n{context}",
            },
        ],
    )
    return response.output_text.strip()
