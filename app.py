from __future__ import annotations

import html
import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DIR
from src.generation import azure_openai_is_configured, generate_answer
from src.retrieval import Retriever
 
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

st.set_page_config(page_title="Visual RAG: Solar System", layout="wide")
st.title("Visual RAG: Solar System")
st.caption("Semantic search over full NASA solar-system documents, chunked locally and visualized in embedding space.")


def format_chunk_preview(text: str, line_width: int = 76, max_lines: int = 4) -> str:
    normalized = " ".join(str(text).split())
    lines = textwrap.wrap(normalized, width=line_width, max_lines=max_lines, placeholder="...")
    return "<br>".join(html.escape(line) for line in lines)


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever()


@st.cache_data
def load_projection() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "projection.csv")


retriever = load_retriever()
projection = load_projection()
embedding_count = int(getattr(retriever.index, "ntotal", len(retriever.chunks)))

st.metric("Number of embeddings", f"{embedding_count:,}")

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

plot_frame = projection[projection["projection"] == selected_projection].copy()
plot_frame["retrieved"] = plot_frame["chunk_id"].isin(result_ids)
plot_frame["chunk_preview"] = plot_frame["text"].map(format_chunk_preview)

hover_template = (
    "<b>%{customdata[0]}</b><br>"
    "Chunk: %{customdata[1]}<br><br>"
    "%{customdata[2]}"
    "<extra></extra>"
)

x_range = [plot_frame["x"].min(), plot_frame["x"].max()]
y_range = [plot_frame["y"].min(), plot_frame["y"].max()]
categories = sorted(plot_frame["category"].dropna().unique())
palette = px.colors.qualitative.Plotly
category_colors = {
    category: palette[index % len(palette)]
    for index, category in enumerate(categories)
}

left, right = st.columns(2)

with left:
    st.subheader("Full embedding map")
    fig = px.scatter(
        plot_frame,
        x="x",
        y="y",
        color="category",
        color_discrete_map=category_colors,
        category_orders={"category": categories},
        custom_data=["title", "chunk_id", "chunk_preview"],
        title=f"All chunks ({selected_projection})",
    )
    fig.update_traces(marker={"size": 10}, hovertemplate=hover_template)
    fig.update_layout(height=620, xaxis_range=x_range, yaxis_range=y_range)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Retrieved chunks on the map")
    retrieved_frame = plot_frame[plot_frame["retrieved"]].copy()
    if not retrieved_frame.empty:
        retrieved_fig = px.scatter(
            retrieved_frame,
            x="x",
            y="y",
            color="category",
            color_discrete_map=category_colors,
            category_orders={"category": categories},
            custom_data=["title", "chunk_id", "chunk_preview"],
            text="title",
            title=f"Retrieved chunks only ({selected_projection})",
        )
        retrieved_fig.update_traces(marker={"size": 14}, textposition="top center", hovertemplate=hover_template)
        retrieved_fig.update_layout(height=620, xaxis_range=x_range, yaxis_range=y_range)
        st.plotly_chart(retrieved_fig, use_container_width=True)
    else:
        st.info("Ask a question to retrieve matching chunks.")

st.subheader("Retrieved chunks")
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
