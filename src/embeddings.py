"""
embeddings.py
--------------
This module handles converting text into numerical vectors (embeddings).

WHY EMBEDDINGS?
Computers can't directly compare the "meaning" of two pieces of text.
But if we convert text into numbers (vectors), we can calculate how
"close" two pieces of text are in meaning using math.

For example:
  "The cat sat on the mat" → [0.12, 0.87, 0.45, ...]
  "A feline rested on the rug" → [0.11, 0.85, 0.47, ...]
  (These vectors will be very SIMILAR because the sentences mean similar things)

  "The stock market crashed" → [0.92, 0.03, 0.61, ...]
  (This vector will be very DIFFERENT from the cat sentence)

We use the 'sentence-transformers' library which provides pretrained
models that are specifically trained to produce good semantic embeddings.

MODEL USED:
"all-MiniLM-L6-v2" — This is a small, fast model that produces
384-dimensional vectors. It works well for document Q&A tasks.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

# The embedding model we use. This is downloaded once and cached locally.
# It produces 384-dimensional vectors for each piece of text.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the sentence transformer model.

    The model is downloaded from HuggingFace Hub on first use
    and then cached locally so subsequent runs are faster.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Converts a list of text strings into a 2D numpy array of embeddings.

    Parameters:
    - texts: A list of strings to embed.
             e.g., ["The attendance is 75%", "Exams are held in December", ...]
    - model: The loaded SentenceTransformer model.

    Returns:
    - A numpy array of shape (len(texts), 384)
      Each row is the embedding vector for one text.

    The 'show_progress_bar' flag displays a progress bar during encoding,
    which is helpful when processing many chunks.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")
    # float32 is required by FAISS (the vector database we use)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Converts a single user query into an embedding vector.

    Parameters:
    - query: The user's question as a string.
    - model: The loaded SentenceTransformer model.

    Returns:
    - A numpy array of shape (1, 384) — one row for the one query.
      FAISS expects a 2D array even for a single query.
    """
    query_embedding = model.encode([query])  # Encode as a list of 1
    return np.array(query_embedding, dtype="float32")
