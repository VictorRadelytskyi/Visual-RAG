from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DIR
from src.retrieval import Retriever


st.set_page_config(page_title="Visual RAG: Solar System", layout="wide")
st.title("Visual RAG: Solar System")
st.caption("Semantic search over full NASA solar-system documents, chunked locally and visualized in embedding space.")


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever()


@st.cache_data
def load_projection() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "projection.csv")


retriever = load_retriever()
projection = load_projection()

query = st.text_input(
    "Ask a question",
    value="Which moon may have an ocean beneath its icy surface?",
)
top_k = st.slider("Number of nearest chunks", min_value=3, max_value=10, value=5)
results = retriever.search(query, top_k=top_k) if query.strip() else []
result_ids = {result["chunk_id"] for result in results}

left, right = st.columns([1.25, 1])

with left:
    plot_frame = projection.copy()
    plot_frame["retrieved"] = plot_frame["chunk_id"].isin(result_ids)
    fig = px.scatter(
        plot_frame,
        x="x",
        y="y",
        color="category",
        symbol="retrieved",
        hover_data=["title", "chunk_id"],
        title=f"Embedding space ({plot_frame['projection'].iloc[0]})",
    )
    fig.update_traces(marker={"size": 10})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Nearest neighbors")
    for rank, result in enumerate(results, start=1):
        with st.expander(f"{rank}. {result['title']} · score {result['score']:.3f}", expanded=rank <= 2):
            st.caption(f"{result['category']} · {result['chunk_id']}")
            st.write(result["text"])

st.subheader("Retrieval notes")
st.write(
    """
    Use this dashboard to compare successful and weak queries. Good retrieval should surface chunks
    from the expected object or category. Weak retrieval often appears when a query is vague, when
    several objects share similar properties, or when the answer depends on a fact mentioned only once.
    """
)

