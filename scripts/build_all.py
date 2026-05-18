from src.chunking import build_chunks
from src.download_corpus import download_documents
from src.indexing import build_index
from src.projection import build_projection


if __name__ == "__main__":
    download_documents()
    build_chunks()
    build_index()
    build_projection()
    print("Pipeline complete")

