"""Embedding helpers for DocuMind's local semantic search."""

from sentence_transformers import SentenceTransformer
import numpy as np


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load the existing lightweight embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Create unit-length float32 embeddings suitable for cosine similarity."""
    if not texts:
        return np.empty((0, 0), dtype="float32")
    return np.asarray(
        model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype="float32",
    )


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Create a unit-length query embedding with the same model settings."""
    return embed_texts([query], model)
