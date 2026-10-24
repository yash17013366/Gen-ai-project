"""
rag.py
-------
This is the core RAG (Retrieval-Augmented Generation) module.

WHAT IS RAG?
RAG = Retrieval-Augmented Generation.

Instead of letting the LLM answer from its training data (which can lead
to hallucination), we:
  1. RETRIEVE relevant text chunks from our documents.
  2. AUGMENT the LLM's prompt with that retrieved context.
  3. Ask the LLM to GENERATE an answer using ONLY that context.

This grounded approach means the LLM answers from the documents, not
from its general knowledge — which is what we want for a document Q&A app.

HOW THIS MODULE WORKS:
  build_prompt()
    → Constructs the text we send to the LLM.
      It includes the retrieved document chunks and the user's question.

  query_ollama()
    → Sends the prompt to the Qwen model via Ollama's API and returns
      the generated answer.

  answer_question()
    → Orchestrates the full pipeline:
        query → embed → FAISS search → retrieve chunks →
        build prompt → call Qwen → return answer + sources
"""

import time

import numpy as np
import requests   # For making HTTP requests to Ollama's local REST API
from sentence_transformers import SentenceTransformer
from src.embeddings import embed_query
from src.vector_store import search_index
from validation import calculate_performance_metrics

# -----------------------------------------------------------------
# CONFIGURATION — Change model name here if you switch Qwen versions
# -----------------------------------------------------------------

# The Ollama model name to use for generating answers.
# Run "ollama pull qwen2.5:1.5b" (or another variant) and set that name here.
OLLAMA_MODEL = "qwen2.5:1.5b"

# Ollama runs as a local server on port 11434 by default.
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# How many top-matching chunks to retrieve from FAISS for each question.
TOP_K_RESULTS = 5


def calculate_grounding_proxy(
    answer: str,
    context_chunks: list[dict],
    embedding_model: SentenceTransformer
) -> dict[str, str | int | float] | None:
    """Return the strongest answer-to-context cosine similarity, not an accuracy score."""
    if not answer.strip() or not context_chunks:
        return None

    texts = [answer] + [chunk["text"] for chunk in context_chunks]
    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    similarities = vectors[1:] @ vectors[0]
    best_index = int(np.argmax(similarities))
    best_chunk = context_chunks[best_index]

    return {
        "cosine_similarity": float(np.clip(similarities[best_index], -1.0, 1.0)),
        "source": best_chunk["source"],
        "page": best_chunk["page"],
    }


# -----------------------------------------------------------------
# PROMPT CONSTRUCTION
# -----------------------------------------------------------------

def build_prompt(context_chunks: list[dict], user_question: str) -> str:
    """
    Builds the text prompt that will be sent to the Qwen LLM.

    We include:
    1. System instructions telling Qwen to ONLY use the provided context.
    2. The document context (chunks of text from our PDFs).
    3. The user's question.

    WHY THIS PREVENTS HALLUCINATION:
    By explicitly instructing the model to use ONLY the context and to say
    "not found" when the answer isn't there, we reduce the chance of the
    model making up facts.

    Parameters:
    - context_chunks: List of relevant chunk dicts retrieved from FAISS.
    - user_question:  The user's question string.

    Returns the full prompt as a single string.
    """

    # Build the context block from all retrieved chunks
    # Each chunk shows its source and page so the model can "see" where it's from
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Chunk {i} — Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )

    # Join all chunks into one big context string
    context_text = "\n\n".join(context_parts)

    # Compose the full prompt
    prompt = f"""You are a document question-answering assistant.

Answer the user's question using ONLY the information provided in the DOCUMENT CONTEXT below.

If the answer cannot be found in the provided context, respond with exactly:
"I couldn't find sufficient information about this in the uploaded documents."

Do NOT invent facts. Do NOT use knowledge outside of the provided context.
Provide a concise and clear answer.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{user_question}

ANSWER:"""

    return prompt


# -----------------------------------------------------------------
# OLLAMA COMMUNICATION
# -----------------------------------------------------------------

def query_ollama(
    prompt: str
) -> tuple[str, dict[str, float | int | None]]:
    """
    Sends a prompt to the Ollama API and returns the model's response.

    Ollama runs as a local server at http://localhost:11434.
    We send a POST request with the model name and prompt.

    Returns the answer and the measured Ollama performance metrics.

    Parameters:
    - prompt: The full prompt string to send to the model.

    Returns:
    - A tuple containing the model's answer and its performance metrics.

    Raises:
    - ConnectionError if Ollama is not running.
    - RuntimeError if the model is unavailable or any API error occurs.
    """

    # Prepare the request payload for Ollama
    payload = {
        "model": OLLAMA_MODEL,    # Which model to use
        "prompt": prompt,          # The full prompt text
        "stream": False            # Get the full response at once (no streaming)
    }

    try:
        request_start = time.perf_counter()
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        request_latency = time.perf_counter() - request_start
        # timeout=120 → wait up to 2 minutes for the model to respond
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama. "
            "Please make sure Ollama is running (run 'ollama serve' in your terminal)."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama took too long to respond. Try again.")

    # Check if the HTTP request was successful
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama API error (status {response.status_code}): {response.text}"
        )

    # The Ollama API returns a JSON object. The model's answer is in 'response'.
    result = response.json()
    answer = result.get("response", "").strip()
    performance = calculate_performance_metrics(result, request_latency)

    return answer, performance


# -----------------------------------------------------------------
# FULL RAG PIPELINE
# -----------------------------------------------------------------

def answer_question(
    user_question: str,
    faiss_index,
    chunks: list[dict],
    embedding_model
) -> dict:
    """
    The main RAG pipeline function.

    Given a user's question, this function:
      1. Embeds the question into a vector.
      2. Searches the FAISS index for the most relevant chunks.
      3. Builds a grounding prompt with those chunks.
      4. Calls the Qwen LLM via Ollama.
      5. Returns the answer and source citations.

    Parameters:
    - user_question:   The question the user typed.
    - faiss_index:     The FAISS index built from document chunks.
    - chunks:          The list of chunk dicts (text + source + page).
    - embedding_model: The loaded SentenceTransformer model.

    Returns a dict with the answer, sources, and measured performance:
    {
        "answer": "The minimum attendance is 75%.",
        "sources": [
            {"source": "Rules.pdf", "page": 3},
            {"source": "Handbook.pdf", "page": 7}
        ],
        "performance": {"tokens_per_second": 42.0}
    }
    """

    pipeline_start = time.perf_counter()

    # STEP 1: Convert the user's question into an embedding vector
    embedding_start = time.perf_counter()
    query_vector = embed_query(user_question, embedding_model)
    embedding_seconds = time.perf_counter() - embedding_start

    # STEP 2: Search FAISS for the top-k most similar chunks
    retrieval_start = time.perf_counter()
    relevant_chunks = search_index(query_vector, faiss_index, chunks, top_k=TOP_K_RESULTS)
    retrieval_seconds = time.perf_counter() - retrieval_start

    # If FAISS returns nothing, we can't answer
    if not relevant_chunks:
        return {
            "answer": "I couldn't find sufficient information about this in the uploaded documents.",
            "sources": [],
            "performance": {
                "query_embedding_seconds": embedding_seconds,
                "retrieval_seconds": retrieval_seconds,
                "rag_pipeline_seconds": time.perf_counter() - pipeline_start,
            },
            "quality": {"grounding_proxy": None},
        }

    # STEP 3: Build the prompt using the retrieved chunks
    prompt = build_prompt(relevant_chunks, user_question)

    # STEP 4: Call the Qwen model via Ollama
    answer, performance = query_ollama(prompt)
    grounding_proxy = calculate_grounding_proxy(
        answer, relevant_chunks, embedding_model
    )

    # STEP 5: Compile unique source citations from the retrieved chunks
    # We deduplicate using a set of (source, page) tuples
    seen = set()
    sources = []
    for chunk in relevant_chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk["source"], "page": chunk["page"]})

    return {
        "answer": answer,
        "sources": sources,
        "performance": {
            **performance,
            "query_embedding_seconds": embedding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "rag_pipeline_seconds": time.perf_counter() - pipeline_start,
        },
        "quality": {"grounding_proxy": grounding_proxy},
    }
