"""Evidence-only question answering with validated, claim-level citations."""

from __future__ import annotations

import re
import time

import requests

from src.embeddings import embed_query
from src.vector_store import search_index
from validation import calculate_performance_metrics


# Qwen3 4B was tested first, but Ollama 0.40.17 on this laptop ignores its
# ``think: false`` request and returns reasoning text instead of an answer.
# Qwen2.5 3B is installed, non-thinking, fast, and follows the citation format.
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TOP_K_RESULTS = 5
SUMMARY_CONTEXT_LIMIT = 6
MIN_RELEVANCE_SCORE = 0.25
NOT_FOUND_ANSWER = "I couldn't find sufficient information about this in the uploaded documents."
_CITATION_RE = re.compile(r"\[(C\d+)\]")
_BRACKETED_TOKEN_RE = re.compile(r"\[([^\]]+)\]")
_SUMMARY_REQUEST_RE = re.compile(
    r"\b(?:summar(?:y|ize|ise)|overview|main (?:idea|points)|"
    r"what (?:does|is) (?:this|the) (?:document|doc) (?:say|about))\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "has", "have", "had", "will", "would", "should", "about",
    "into", "than", "then", "their", "there", "they", "your", "only",
}


def _make_evidence(retrieved_chunks: list[dict]) -> list[dict]:
    """Give this request's retrieved chunks short IDs for the model to cite."""
    return [
        {**chunk, "evidence_id": f"C{position}"}
        for position, chunk in enumerate(retrieved_chunks, start=1)
    ]


def is_summary_request(question: str) -> bool:
    """Recognise clear document-overview requests without changing normal Q&A."""
    return bool(_SUMMARY_REQUEST_RE.search(question))


def _evenly_spaced(items: list[dict], count: int) -> list[dict]:
    """Choose representative items without requiring another retrieval model."""
    if count >= len(items):
        return list(items)
    indexes = {
        round(position * (len(items) - 1) / (count - 1))
        for position in range(count)
    } if count > 1 else {0}
    return [item for index, item in enumerate(items) if index in indexes]


def select_summary_chunks(chunks: list[dict], limit: int = SUMMARY_CONTEXT_LIMIT) -> list[dict]:
    """Select page-diverse evidence for a document-level summary.

    One representative passage is first reserved for every document. Remaining
    slots are allocated to larger documents and sampled evenly through pages.
    This avoids weakening factual-question relevance thresholds.
    """
    by_source: dict[str, list[dict]] = {}
    seen_pages: set[tuple[str, int]] = set()
    for chunk in chunks:
        page_key = (chunk["source"], chunk["page"])
        if page_key not in seen_pages:
            by_source.setdefault(chunk["source"], []).append(chunk)
            seen_pages.add(page_key)

    sources = list(by_source)
    if not sources:
        return []
    target = min(limit, sum(len(pages) for pages in by_source.values()))
    allocation = {source: 1 for source in sources}
    while sum(allocation.values()) < target:
        candidates = [
            source for source in sources if allocation[source] < len(by_source[source])
        ]
        if not candidates:
            break
        source = max(
            candidates,
            key=lambda name: (len(by_source[name]) - allocation[name]) / len(by_source[name]),
        )
        allocation[source] += 1

    selected: list[dict] = []
    for source in sources:
        selected.extend(_evenly_spaced(by_source[source], allocation[source]))
    return [{**chunk, "similarity": None} for chunk in selected]


def _meaningful_terms(text: str) -> set[str]:
    """Cheap safeguard against a citation that clearly has no shared evidence."""
    return {
        term.casefold()
        for term in re.findall(r"\b(?:\d+|[A-Za-z]{3,})\b", text)
        if term.casefold() not in _STOP_WORDS
    }


def build_prompt(evidence: list[dict], user_question: str) -> str:
    """Build a compact prompt whose citations can be deterministically checked."""
    context = "\n\n".join(f"[{item['evidence_id']}]\n{item['text']}" for item in evidence)
    return f"""You are DocuMind, an evidence-only document assistant.

Use ONLY the evidence blocks below. Do not use outside knowledge, infer missing
facts, or mention filenames/page numbers yourself. Give a concise answer in at
most four bullet points. Every non-empty answer line must end with one or more
evidence IDs in square brackets, for example: - The deadline is 10 May. [C1]
Use only IDs that are present in the evidence. Cite the evidence that supports
that particular line; do not add a general source list.

If the evidence is not sufficient, reply with exactly this sentence and nothing
else: {NOT_FOUND_ANSWER}

EVIDENCE:
{context}

QUESTION: {user_question}
ANSWER:"""


def build_summary_prompt(evidence: list[dict], user_question: str) -> str:
    """Ask for a concise overview based on page-diverse document evidence."""
    context = "\n\n".join(f"[{item['evidence_id']}]\n{item['text']}" for item in evidence)
    return f"""You are DocuMind, an evidence-only document assistant.

The user is asking for a short overview of the uploaded document collection.
Use ONLY the representative evidence below. Write three to five concise bullets,
with one distinct supported point per bullet. Do not include citations, source
identifiers, filenames, page numbers, a title, or a general source list; DocuMind
will add verified citations itself. Do not use outside knowledge. If the evidence
is insufficient, reply exactly: {NOT_FOUND_ANSWER}

REPRESENTATIVE EVIDENCE:
{context}

QUESTION: {user_question}
ANSWER:"""


def query_ollama(prompt: str) -> tuple[str, dict[str, float | int | None]]:
    """Ask the local Ollama model using conservative generation settings."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 300, "num_ctx": 4096},
    }
    try:
        started = time.perf_counter()
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        request_latency = time.perf_counter() - started
    except requests.exceptions.ConnectionError as error:
        raise ConnectionError(
            "Could not connect to Ollama. Start Ollama, then run `ollama pull qwen2.5:3b`."
        ) from error
    except requests.exceptions.Timeout as error:
        raise TimeoutError("Ollama took too long to respond. Try a shorter question.") from error
    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error (status {response.status_code}): {response.text}")

    result = response.json()
    return result.get("response", "").strip(), calculate_performance_metrics(result, request_latency)


def validate_cited_answer(model_answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Accept only lines with real, request-scoped evidence IDs.

    The LLM never supplies a filename or page number. A fabricated/unknown ID or
    an uncited claim makes the response unverified, so DocuMind abstains instead.
    """
    answer = model_answer.strip()
    if answer == NOT_FOUND_ANSWER:
        return NOT_FOUND_ANSWER, []
    if not answer:
        return NOT_FOUND_ANSWER, []

    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    used_ids: list[str] = []
    verified_lines: list[str] = []
    for line in (line.strip() for line in answer.splitlines()):
        if not line:
            continue
        ids = _CITATION_RE.findall(line)
        bracketed_tokens = _BRACKETED_TOKEN_RE.findall(line)
        if (
            not ids
            or bracketed_tokens != ids
            or any(identifier not in evidence_by_id for identifier in ids)
        ):
            return NOT_FOUND_ANSWER, []
        claim = _CITATION_RE.sub("", line).strip(" -•\t")
        if not claim:
            return NOT_FOUND_ANSWER, []
        claim_terms = _meaningful_terms(claim)
        # This is a lightweight relevance check, not a claim-entailment model.
        # Each cited passage must share at least one meaningful term with its claim.
        if not claim_terms or any(
            not (claim_terms & _meaningful_terms(evidence_by_id[identifier]["text"]))
            for identifier in ids
        ):
            return NOT_FOUND_ANSWER, []
        verified_lines.append(line)
        for identifier in ids:
            if identifier not in used_ids:
                used_ids.append(identifier)

    if not verified_lines:
        return NOT_FOUND_ANSWER, []

    citations = [
        {
            "id": identifier,
            "source": evidence_by_id[identifier]["source"],
            "page": evidence_by_id[identifier]["page"],
            "text": evidence_by_id[identifier]["text"],
            "similarity": evidence_by_id[identifier]["similarity"],
        }
        for identifier in used_ids
    ]
    return "\n".join(verified_lines), citations


def validate_summary_answer(model_answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Attach deterministic evidence IDs to a short model-generated overview.

    Smaller local models often follow summary content instructions but place one
    source list at the end rather than citing every bullet. For summaries only,
    DocuMind removes model-supplied markers and assigns each retained bullet to
    the supplied chunk with the strongest meaningful-term overlap. A claim with
    no matching evidence is not shown.
    """
    answer = model_answer.strip()
    if answer == NOT_FOUND_ANSWER or not answer:
        return NOT_FOUND_ANSWER, []

    used_ids: list[str] = []
    verified_lines: list[str] = []
    seen_claims: set[str] = set()
    for raw_line in answer.splitlines():
        claim = _BRACKETED_TOKEN_RE.sub("", raw_line)
        claim = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", claim).strip()
        claim_terms = _meaningful_terms(claim)
        if not claim_terms:
            continue

        best_item = max(
            evidence,
            key=lambda item: len(claim_terms & _meaningful_terms(item["text"])),
        )
        overlap = claim_terms & _meaningful_terms(best_item["text"])
        normalized_claim = claim.casefold()
        if not overlap or normalized_claim in seen_claims:
            continue

        evidence_id = best_item["evidence_id"]
        verified_lines.append(f"- {claim} [{evidence_id}]")
        seen_claims.add(normalized_claim)
        if evidence_id not in used_ids:
            used_ids.append(evidence_id)
        if len(verified_lines) == 5:
            break

    if not verified_lines:
        return NOT_FOUND_ANSWER, []

    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    citations = [
        {
            "id": identifier,
            "source": evidence_by_id[identifier]["source"],
            "page": evidence_by_id[identifier]["page"],
            "text": evidence_by_id[identifier]["text"],
            "similarity": evidence_by_id[identifier]["similarity"],
        }
        for identifier in used_ids
    ]
    return "\n".join(verified_lines), citations


def _retrieval_summary(
    retrieved: list[dict], citations: list[dict], mode: str = "Semantic search"
) -> dict:
    """Expose useful, non-accuracy diagnostics for the Streamlit interface."""
    return {
        "retrieved_passages": len(retrieved),
        "verified_citations": len(citations),
        "unique_documents": len({chunk["source"] for chunk in retrieved}),
        "top_similarity": retrieved[0]["similarity"] if retrieved else None,
        "mode": mode,
    }


def _not_found_result(
    embedding_seconds: float,
    retrieval_seconds: float,
    started: float,
    retrieved: list[dict] | None = None,
) -> dict:
    return {
        "answer": NOT_FOUND_ANSWER,
        "citations": [],
        "sources": [],
        "retrieval": _retrieval_summary(retrieved or [], []),
        "performance": {
            "query_embedding_seconds": embedding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "rag_pipeline_seconds": time.perf_counter() - started,
        },
    }


def answer_question(user_question: str, faiss_index, chunks: list[dict], embedding_model) -> dict:
    """Retrieve evidence, generate an answer, and return only verified citations."""
    started = time.perf_counter()
    if is_summary_request(user_question):
        selection_started = time.perf_counter()
        retrieved = select_summary_chunks(chunks)
        retrieval_seconds = time.perf_counter() - selection_started
        if not retrieved:
            return _not_found_result(0.0, retrieval_seconds, started)

        evidence = _make_evidence(retrieved)
        raw_answer, ollama_metrics = query_ollama(
            build_summary_prompt(evidence, user_question)
        )
        answer, citations = validate_summary_answer(raw_answer, evidence)
        return {
            "answer": answer,
            "citations": citations,
            "sources": citations,
            "retrieval": _retrieval_summary(
                retrieved, citations, mode="Document overview"
            ),
            "performance": {
                **ollama_metrics,
                "query_embedding_seconds": 0.0,
                "retrieval_seconds": retrieval_seconds,
                "rag_pipeline_seconds": time.perf_counter() - started,
            },
        }

    embedding_started = time.perf_counter()
    query_vector = embed_query(user_question, embedding_model)
    embedding_seconds = time.perf_counter() - embedding_started

    retrieval_started = time.perf_counter()
    retrieved = search_index(query_vector, faiss_index, chunks, top_k=TOP_K_RESULTS)
    retrieval_seconds = time.perf_counter() - retrieval_started
    if not retrieved or retrieved[0]["similarity"] < MIN_RELEVANCE_SCORE:
        return _not_found_result(embedding_seconds, retrieval_seconds, started, retrieved)

    evidence = _make_evidence(retrieved)
    raw_answer, ollama_metrics = query_ollama(build_prompt(evidence, user_question))
    answer, citations = validate_cited_answer(raw_answer, evidence)

    return {
        "answer": answer,
        "citations": citations,
        "sources": citations,
        "retrieval": _retrieval_summary(retrieved, citations),
        "performance": {
            **ollama_metrics,
            "query_embedding_seconds": embedding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "rag_pipeline_seconds": time.perf_counter() - started,
        },
    }
