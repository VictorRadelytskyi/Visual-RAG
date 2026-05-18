# Visual Exploration of a RAG System and Document Space

## Chosen topic

This implementation uses a corpus of **full NASA Science articles about the Solar System**:
planets, moons, dwarf planets, asteroids, comets, the Sun, and overview pages.

The corpus is intentionally made of **whole raw documents**, not pre-split passages, so the project
demonstrates the full RAG preparation workflow:

1. collect raw documents,
2. split them into chunks,
3. embed the chunks,
4. store them in a FAISS vector index,
5. retrieve nearest neighbors for a query,
6. visualize the embedding space in Streamlit.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build_all
streamlit run app.py
```

Recommended Python version: **3.10–3.12**. The embedding stack depends on
`sentence-transformers`/PyTorch, which is usually smoother on those versions than on very new
Python releases.

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
```

## Example questions

- Which moon may have an ocean beneath its icy surface?
- Why is Venus hotter than Mercury?
- Which dwarf planet is located in the asteroid belt?
- What objects are found in the Kuiper Belt?
- Which planet is the largest in the Solar System?
- What are comets made of?

---

## Project Goal

The goal of this project is to develop a **RAG (Retrieval-Augmented Generation)** system for a selected corpus of documents, with a strong emphasis on **visualizing how the system operates**.

The project should not focus solely on running a language model. Instead, its primary objective is to demonstrate:

- the structure of the document corpus,
- semantic search results,
- embedding space relationships,
- and the context fragments used during answer generation.

The system should allow users to understand *why* a particular answer was generated and *which document fragments* influenced it.

---

# Project Scope

The project should include the following stages:

1. Preparation of a document corpus.
2. Splitting documents into smaller chunks/fragments.
3. Generating embeddings for document fragments.
4. Storing embeddings in a vector database.
5. Implementing semantic search.
6. (Optional) Using a language model to generate answers.
7. Building a Streamlit dashboard for visualization and interaction.

A crucial part of the project is the dashboard, which should visually present:

- which documents/fragments were retrieved for a query,
- similarity relationships in embedding space,
- nearest neighbors,
- and the fragments used as context for generated answers.

---

# Functional Requirements

## Core Features

The project must include:

- preparation of a document corpus,
- document chunking,
- embedding generation for document fragments,
- storing embeddings in a vector database such as:
  - Elasticsearch,
  - Chroma,
  - FAISS,
  - Qdrant,
  - or Milvus,
- implementation of a semantic search mechanism,
- optional integration with a language model for answer generation,
- creation of a Streamlit dashboard.

---

# Visualization Requirements

The dashboard should provide:

- 2D visualization of document embeddings using dimensionality reduction techniques such as:
  - UMAP,
  - PCA,
  - t-SNE,
  - TriMAP,
  - or PaCMAP,
- highlighting fragments used as context for answers,
- displaying nearest neighbors for user queries,
- visualization of semantic clusters and relationships between documents.

The system should also support analysis of cases where retrieval works well and cases where irrelevant or poor-quality context is returned.

---

# Implementation Variants

## Basic Variant

- Semantic search engine
- Embedding visualization
- No local LLM required

## Intermediate Variant

- Full RAG pipeline
- Use of an external API or a lightweight language model

## Advanced Variant

- Local quantized SLM/LLM
- Running on NVIDIA GPU hardware

---

# Extended Features

Possible extensions include comparison of:

- different chunking strategies,
- different embedding models,
- different similarity metrics,
- different language models.

This allows evaluation of how architectural decisions affect retrieval quality and answer generation.

---

# Example Data Sources

Possible datasets and corpora include:

- software documentation,
- Wikipedia articles,
- regulations and legal documents,
- scientific publications,
- technical documentation,
- FAQ datasets,
- question-answer collections.

---

# Suggested Technology Stack

| Component | Suggested Technologies |
|---|---|
| Backend | Python |
| Dashboard | Streamlit |
| Embeddings | SentenceTransformers / OpenAI Embeddings |
| Vector Database | Chroma, FAISS, Qdrant |
| Visualization | Plotly, Matplotlib |
| Dimensionality Reduction | UMAP, PCA, t-SNE |
| LLM (Optional) | Llama.cpp, Ollama, OpenAI API |

---

# Expected Outcome

The final system should provide:

- an interactive semantic search engine,
- embedding space visualization,
- insight into retrieval behavior,
- explainability of generated answers,
- and practical understanding of RAG system architecture.

The project should emphasize both the engineering implementation and the interpretability of the retrieval pipeline.
