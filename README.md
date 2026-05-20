# Visual Exploration of a RAG System and Document Space

This project builds a **visual, explainable Retrieval-Augmented Generation (RAG) system** over a corpus of **NASA Science articles about the Solar System**. It focuses not only on retrieving relevant information, but also on making the retrieval pipeline understandable through interactive visualization.

The corpus includes raw long-form documents about planets, moons, dwarf planets, asteroids, comets, the Sun, and broader Solar System topics. Instead of starting from pre-chunked data, the project demonstrates the complete workflow from raw document collection to chunk-level search and embedding-space exploration.

## What this project demonstrates

The system walks through the full RAG preparation pipeline:

1. collecting raw documents,
2. cleaning and normalizing source content,
3. splitting documents into chunks,
4. generating embeddings for chunks,
5. storing vectors in a FAISS index,
6. retrieving nearest neighbors for user queries,
7. projecting embeddings into 2D space,
8. visualizing corpus structure and retrieval behavior in Streamlit.

The main goal is **interpretability**. Instead of treating retrieval as a black box, the app helps users see:

- how documents are represented in embedding space,
- which chunks are nearest to a query,
- how semantic clusters form,
- and which fragments are used as answer context.

## Corpus

The current implementation uses **full NASA Science articles about the Solar System** as the knowledge base.

This choice makes the project useful for demonstrating:

- domain-focused semantic search,
- chunking long-form scientific content,
- retrieval quality analysis,
- and visually exploring relationships between related astronomy concepts.

## Features

- End-to-end document preparation pipeline
- Local chunking of raw source articles
- Embedding generation with `sentence-transformers`
- FAISS-based vector retrieval
- 2D embedding projection with techniques such as UMAP/PCA
- Streamlit dashboard for interactive exploration
- Query-to-nearest-neighbor inspection
- Optional Azure OpenAI-based answer generation
- Retrieval-only mode when no LLM credentials are configured

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build_all
streamlit run app.py
```

Recommended Python version: **3.10–3.12**.

The embedding stack depends on `sentence-transformers` and PyTorch, which are generally more reliable on these Python versions than on very recent releases.

## Optional Azure OpenAI answer generation

The application works without a language model. If Azure OpenAI environment variables are not set, the app remains in **retrieval-only mode** and skips answer generation.

```bash
cp .env.example .env
```

Then configure:

```dotenv
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_BASE_URL="https://YOUR-RESOURCE-NAME.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="YOUR_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION="YOUR_API_VERSION"
```

The app loads `.env` automatically at startup.

### Configuration notes

- `AZURE_OPENAI_BASE_URL` must be your **Azure OpenAI resource endpoint**, not the public OpenAI API.
- For the current implementation, use a base resource URL such as `https://YOUR-RESOURCE-NAME.openai.azure.com/`, not a `/openai/v1/` URL.
- `AZURE_OPENAI_DEPLOYMENT` should be your Azure deployment name.
- `AZURE_OPENAI_API_VERSION` must match a version supported by your Azure resource and deployment.

## Project structure

```text
data/
  sources/                 # NASA URL manifest
  raw/                     # downloaded full documents
  processed/               # chunks, embeddings, FAISS index, projection
src/
  download_corpus.py       # fetch + clean raw NASA articles
  chunking.py              # local chunking strategy
  indexing.py              # sentence-transformer embeddings + FAISS
  projection.py            # UMAP/PCA 2D projection
  retrieval.py             # semantic search
app.py                     # Streamlit dashboard
pages/1_Chunk_Detail.py    # drill-down page for a selected chunk
```

## Example questions

You can test the system with questions such as:

- Which moon may have an ocean beneath its icy surface?
- Why is Venus hotter than Mercury?
- Which dwarf planet is located in the asteroid belt?
- What objects are found in the Kuiper Belt?
- Which planet is the largest in the Solar System?
- What are comets made of?

## Visualization goals

A central requirement of this project is the ability to inspect how retrieval behaves.

The dashboard is intended to help users analyze:

- which documents or fragments were retrieved for a query,
- how retrieved chunks are positioned relative to the wider corpus,
- what the nearest semantic neighbors are,
- which fragments were used as context for generated answers,
- and where retrieval quality is strong or weak.

This makes the system useful not only as a demo RAG app, but also as a learning and debugging tool.

## Scope

The project covers the following stages:

1. preparation of a document corpus,
2. chunking documents into smaller fragments,
3. generating embeddings for document fragments,
4. storing embeddings in a vector index,
5. implementing semantic search,
6. optionally generating answers with a language model,
7. building a Streamlit dashboard for interaction and analysis.

## Implementation variants

This kind of system can be extended in multiple directions:

### Basic variant

- Semantic search engine
- Embedding visualization
- No local LLM required

### Intermediate variant

- Full RAG pipeline
- External API or lightweight model for answer generation

### Advanced variant

- Local quantized SLM/LLM
- GPU-backed execution

## Possible extensions

Potential future improvements include comparing:

- different chunking strategies,
- different embedding models,
- different similarity metrics,
- different vector databases,
- and different language models.

These comparisons help evaluate how architectural choices affect retrieval quality, cluster structure, and answer usefulness.

## Example data sources for similar projects

Although this implementation uses NASA Science articles, the same architecture can be applied to other corpora, such as:

- software documentation,
- Wikipedia articles,
- regulations and legal documents,
- scientific publications,
- technical documentation,
- FAQ datasets,
- question-answer collections.

## Suggested technology stack

| Component | Suggested Technologies |
|---|---|
| Backend | Python |
| Dashboard | Streamlit |
| Embeddings | SentenceTransformers / OpenAI Embeddings |
| Vector Database | Chroma, FAISS, Qdrant |
| Visualization | Plotly, Matplotlib |
| Dimensionality Reduction | UMAP, PCA, t-SNE |
| LLM (Optional) | Llama.cpp, Ollama, OpenAI API |

## Expected outcome

The final system should provide:

- an interactive semantic search engine,
- embedding-space visualization,
- insight into retrieval behavior,
- explainability for generated answers,
- and a practical understanding of RAG system architecture.

Overall, the project emphasizes both **engineering implementation** and **retrieval interpretability**.