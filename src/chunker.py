"""Sentence-aware chunking with source metadata preserved on every chunk."""

from __future__ import annotations

import re


def _normalise_whitespace(text: str) -> str:
    """Make extracted PDF text readable without changing its meaning."""
    return re.sub(r"\s+", " ", text).strip()


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an unusually long sentence at a word boundary, with overlap."""
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def split_text_into_chunks(
    text: str, chunk_size: int = 700, overlap: int = 120
) -> list[str]:
    """Return mostly sentence-complete chunks, retaining boundary context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    clean_text = _normalise_whitespace(text)
    if not clean_text:
        return []

    # The long-text fallback below also handles tables and pages with no periods.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(sentence, chunk_size, overlap))
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        boundary_context = current[-overlap:].strip() if overlap else ""
        current = f"{boundary_context} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


def chunk_pages(
    pages_data: list[dict], chunk_size: int = 700, overlap: int = 120
) -> list[dict]:
    """Chunk page text and attach a stable ID and original location to each chunk."""
    all_chunks: list[dict] = []
    for page in pages_data:
        for chunk_number, chunk_text in enumerate(
            split_text_into_chunks(page.get("text", ""), chunk_size, overlap), start=1
        ):
            all_chunks.append(
                {
                    "chunk_id": f"chunk-{len(all_chunks) + 1:04d}",
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_number": chunk_number,
                }
            )
    return all_chunks
