"""Small, exact FAISS cosine-similarity index for DocuMind."""

from __future__ import annotations

import re

import faiss
import numpy as np


def _text_key(text: str) -> str:
    return re.sub(r"\W+", "", text).casefold()


def build_faiss_index(embeddings: np.ndarray, chunks: list[dict]) -> tuple:
    """Build an exact inner-product index over unit-normalised embeddings."""
    if len(chunks) == 0 or embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings and chunks must be non-empty and have matching lengths.")
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)  # defensive: cosine search requires unit vectors
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def search_index(
    query_embedding: np.ndarray,
    index,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Return non-duplicate chunks ordered by cosine similarity."""
    if index is None or index.ntotal == 0 or top_k <= 0:
        return []
    query = np.ascontiguousarray(query_embedding.copy(), dtype="float32")
    faiss.normalize_L2(query)
    # Search extra neighbours first so duplicate/overlap chunks do not consume top-k.
    scores, indices = index.search(query, min(index.ntotal, max(top_k * 3, top_k)))

    results: list[dict] = []
    seen_texts: set[str] = set()
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue
        chunk = chunks[int(chunk_index)]
        key = _text_key(chunk.get("text", ""))
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        results.append({**chunk, "similarity": float(score)})
        if len(results) == top_k:
            break
    return results
