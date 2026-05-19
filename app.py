from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DIR
from src.generation import azure_openai_is_configured, generate_answer
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

st.subheader("Generated answer")
if not azure_openai_is_configured():
    st.info(
        "Azure OpenAI is not configured, so the app is running in retrieval-only mode. "
        "Add the required environment variables to enable answer generation."
    )
elif results:
    if st.button("Generate answer", type="primary"):
        with st.spinner("Generating answer from retrieved context..."):
            st.session_state["generated_answer"] = generate_answer(query, results)
            st.session_state["generated_answer_query"] = query

    if st.session_state.get("generated_answer_query") == query:
        st.write(st.session_state["generated_answer"])
    else:
        st.caption("Click **Generate answer** to send the retrieved chunks to Azure OpenAI.")

st.divider()

projection_methods = projection["projection"].drop_duplicates().tolist()
selected_projection = st.selectbox(
    "Dimensionality reduction method",
    projection_methods,
    index=projection_methods.index("UMAP") if "UMAP" in projection_methods else 0,
)

left, right = st.columns([1.55, 1])

with left:
    st.subheader("Embedding map")
    plot_frame = projection[projection["projection"] == selected_projection].copy()
    plot_frame["retrieved"] = plot_frame["chunk_id"].isin(result_ids)
    fig = px.scatter(
        plot_frame,
        x="x",
        y="y",
        color="category",
        symbol="retrieved",
        hover_data=["title", "chunk_id"],
        title=f"Embedding space ({selected_projection})",
    )
    fig.update_traces(marker={"size": 11})
    fig.update_layout(height=720)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Retrieved entities")
    if results:
        table = pd.DataFrame(
            [
                {
                    "rank": rank,
                    "title": result["title"],
                    "category": result["category"],
                    "score": round(result["score"], 3),
                    "chunk_id": result["chunk_id"],
                    "details": f"./Chunk_Detail?chunk_id={result['chunk_id']}",
                }
                for rank, result in enumerate(results, start=1)
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "details": st.column_config.LinkColumn(
                    "details",
                    display_text="Open chunk",
                )
            },
        )

        st.markdown("### Retrieved text")
        for rank, result in enumerate(results, start=1):
            with st.expander(f"{rank}. {result['title']} · score {result['score']:.3f}", expanded=rank <= 2):
                st.caption(f"{result['category']} · {result['chunk_id']}")
                st.write(result["text"])
    else:
        st.info("Ask a question to retrieve matching chunks.")

st.subheader("Retrieval notes")
st.write(
    """
    Use this dashboard to compare successful and weak queries. Good retrieval should surface chunks
    from the expected object or category. Weak retrieval often appears when a query is vague, when
    several objects share similar properties, or when the answer depends on a fact mentioned only once.
    """
)
