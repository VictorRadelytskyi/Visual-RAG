from __future__ import annotations

import streamlit as st

from src.chunk_store import get_chunk

st.markdown("""
<style>
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Chunk detail", layout="wide")
st.title("Chunk detail")

chunk_id = st.query_params.get("chunk_id")

if not chunk_id:
    st.info("Choose a chunk from the retrieval table in the main app.")
    st.page_link("app.py", label="← Back to search")
    st.stop()

chunk = get_chunk(chunk_id)
if chunk is None:
    st.error(f"Chunk `{chunk_id}` was not found.")
    st.page_link("app.py", label="← Back to search")
    st.stop()

st.page_link("app.py", label="← Back to search")
st.subheader(chunk["title"])

meta_1, meta_2, meta_3 = st.columns(3)
meta_1.metric("Chunk ID", chunk["chunk_id"])
meta_2.metric("Document ID", chunk["document_id"])
meta_3.metric("Category", chunk["category"])

st.markdown("### Description")
st.write(
    "This is one locally generated chunk from a full NASA source document. "
    "It is the exact text unit used for embedding, semantic retrieval, and optional answer generation."
)

st.markdown("### Chunk text")
st.write(chunk["text"])

st.link_button("Open original NASA source", chunk["url"])
