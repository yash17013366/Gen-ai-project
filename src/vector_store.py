"""
vector_store.py
----------------
This module manages the FAISS vector database.

WHAT IS FAISS?
FAISS (Facebook AI Similarity Search) is a library that stores
embedding vectors and allows us to quickly find the most similar
vectors to a given query vector.

Think of it like a search engine, but instead of searching for
keywords, it searches for semantic similarity using math (cosine or
L2 distance between vectors).

WHAT DOES THIS MODULE DO?
1. build_faiss_index()
   → Takes all the chunk embeddings, creates a FAISS index,
     and stores the chunk metadata alongside it.

2. search_index()
   → Takes a query embedding, asks FAISS to find the top-k
     most similar chunks, and returns those chunks with their metadata.

HOW SIMILARITY SEARCH WORKS:
  1. We have N chunks, each represented as a 384-dim vector stored in FAISS.
  2. The user asks a question. We convert it to a 384-dim vector.
  3. FAISS computes the distance between the query vector and ALL chunk vectors.
  4. FAISS returns the indices of the top-k closest chunks (the most relevant ones).
  5. We use those indices to look up the original chunk text and metadata.
"""

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray, chunks: list[dict]) -> tuple:
    """
    Builds a FAISS index from the provided embeddings.

    Parameters:
    - embeddings: 2D numpy array of shape (num_chunks, embedding_dim)
                  Each row is the embedding for one chunk.
    - chunks:     List of chunk dicts with "text", "source", "page".

    Returns a tuple: (faiss_index, chunks)
    We return chunks alongside the index because FAISS only stores vectors —
    it doesn't store the original text or metadata. We need to keep
    that separately (in the 'chunks' list) and use the index position
    to look up the metadata when FAISS tells us which chunks are relevant.
    """

    # Determine the size of each embedding vector (should be 384)
    embedding_dimension = embeddings.shape[1]

    # Create a flat (brute-force) L2 distance index.
    # "Flat" means FAISS will check all vectors — no approximation.
    # This is perfectly fine for small-to-medium document collections.
    # L2 = Euclidean distance. Smaller distance = more similar vectors.
    index = faiss.IndexFlatL2(embedding_dimension)

    # Add all chunk embeddings to the FAISS index
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors.")

    return index, chunks


def search_index(query_embedding: np.ndarray, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Searches the FAISS index for the most relevant chunks given a query embedding.

    Parameters:
    - query_embedding: The embedded version of the user's question (shape: 1 x 384).
    - index:           The FAISS index built from our document chunks.
    - chunks:          The list of chunk dicts (parallel to the FAISS index).
    - top_k:           How many top results to return. Default is 5.

    Returns:
    A list of chunk dicts (up to top_k) that are most similar to the query.
    Each dict contains: {"text": "...", "source": "file.pdf", "page": 3}
    """

    # D = distances (how similar each result is — lower = better for L2)
    # I = indices (which chunks in our list are most similar)
    D, I = index.search(query_embedding, top_k)

    # I[0] gives us the list of indices for the first (and only) query row
    relevant_chunks = []
    for idx in I[0]:
        # FAISS returns -1 if no valid neighbor was found (e.g., index is empty)
        if idx == -1:
            continue
        relevant_chunks.append(chunks[idx])

    return relevant_chunks
