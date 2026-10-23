from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_DIR= "chroma_db"
COLLECTION = "study_notes"
EMBED_MODEL = "nomic-embed-text"


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text expects task prefixes on documents and queries. """

    def embed_document(self, texts: list[str]) -> list[list[float]]:
        return super().embed_docment([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")



def get_vector_store()-> Chroma:

    """Open (or create) the Chroma collection on disk."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=NomicEmbeddings(model=EMBED_MODEL),
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )