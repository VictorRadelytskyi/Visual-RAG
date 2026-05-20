# Visual-RAG

Visual-RAG is an interactive Python application for exploring how Retrieval-Augmented Generation (RAG) systems work over a real document corpus.

Built around NASA Science articles about the Solar System, the project combines document preprocessing, semantic retrieval, vector search, and embedding-space visualization in a single end-to-end workflow. The focus is on making retrieval behavior visible and explainable rather than treating it as a black box.

## What this project does

The application demonstrates a complete RAG-oriented pipeline:

- collecting and cleaning raw source documents,
- splitting documents into semantic chunks,
- generating embeddings for each chunk,
- storing vectors in a FAISS index,
- retrieving the most relevant context for a query,
- projecting embeddings into 2D space,
- and visualizing corpus structure and retrieval behavior through Streamlit.

## Key highlights

- End-to-end document preparation and retrieval workflow
- FAISS-based semantic search over long-form documents
- Embedding generation using `sentence-transformers`
- Interactive Streamlit UI for visual exploration
- 2D projection of embedding space with PCA/UMAP-style analysis
- Query-to-nearest-neighbor inspection for explainable retrieval
- Optional Azure OpenAI integration for answer generation
- Retrieval-only mode when LLM credentials are not configured

## Why it stands out

This project is designed to show both implementation skill and practical understanding of modern AI system design:

- building a document-based RAG pipeline from raw data,
- working with embeddings and vector indexes,
- analyzing retrieval quality,
- and turning backend ML behavior into an understandable visual product.

Instead of being only a chatbot demo, Visual-RAG is positioned as a tool for inspecting, debugging, and understanding how document retrieval works in practice.

## Tech stack

- Python
- Streamlit
- FAISS
- sentence-transformers
- PyTorch
- Azure OpenAI (optional)

## Example capabilities

With the current dataset, users can explore questions such as:

- Which moon may have an ocean beneath its icy surface?
- Why is Venus hotter than Mercury?
- Which dwarf planet is located in the asteroid belt?
- What objects are found in the Kuiper Belt?

The UI helps show not only the retrieved result, but also *why* that result was retrieved and how it relates to nearby chunks in the corpus.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build_all
streamlit run app.py
```

Recommended Python version: **3.10–3.12**.

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

## Optional Azure OpenAI setup

The app works without a language model. If Azure OpenAI environment variables are not set, it stays in retrieval-only mode.

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

## Resume-style summary

Visual-RAG is a hands-on AI engineering project that demonstrates:

- RAG pipeline design
- document ingestion and chunking
- vector search and retrieval analysis
- embedding visualization
- interactive ML product prototyping in Python

It is intended as a practical showcase project for recruiters, collaborators, and technical discussions around retrieval systems and explainable AI workflows.
