"""
chunker.py
-----------
This module is responsible for splitting large pages of text into
smaller, overlapping chunks.

WHY DO WE CHUNK?
Large documents can have pages with thousands of words.
When we search for relevant content, we want to find the SPECIFIC
paragraph or section that answers the question — not the entire page.

Smaller chunks = more precise retrieval.

HOW IT WORKS:
We take a block of text and split it into pieces of a fixed size
(e.g., 500 characters). To avoid cutting off a sentence mid-thought,
we add an "overlap" — the last few characters of one chunk are
repeated at the start of the next chunk.

Example (chunk_size=20, overlap=5):
  Text:    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  Chunk 1: "ABCDEFGHIJKLMNOPQRST"   (chars 0–19)
  Chunk 2: "PQRSTUVWXYZ..."          (chars 15–34, overlapping chars 15–19)

This overlap ensures that context is not lost at chunk boundaries.
"""


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits a long text string into overlapping chunks.

    Parameters:
    - text:       The full text to split.
    - chunk_size: How many characters each chunk should contain.
    - overlap:    How many characters to repeat from the end of one chunk
                  at the start of the next chunk (context continuity).

    Returns a list of text strings (the chunks).
    """

    chunks = []
    start = 0  # Start position for the current chunk

    while start < len(text):
        # Determine where this chunk ends
        end = start + chunk_size

        # Grab the chunk
        chunk = text[start:end]

        # Only save the chunk if it has meaningful content
        if chunk.strip():
            chunks.append(chunk)

        # Move the start position forward, but step back by 'overlap'
        # so the next chunk repeats the last 'overlap' characters
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages_data: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Takes the list of extracted pages (from pdf_reader.py) and creates
    smaller chunks from each page's text.

    For EACH chunk, we keep the SAME metadata as the parent page:
    - source filename
    - page number

    This is how citations work — every chunk "remembers" where it came from.

    Parameters:
    - pages_data: List of page dicts from pdf_reader.py
                  Each dict: {"text": "...", "source": "file.pdf", "page": 3}
    - chunk_size: Characters per chunk.
    - overlap:    Characters to repeat between chunks.

    Returns a list of chunk dicts:
    [
        {"text": "chunk text...", "source": "file.pdf", "page": 3},
        ...
    ]
    """

    all_chunks = []

    for page in pages_data:
        # Split this page's text into smaller pieces
        text_chunks = split_text_into_chunks(page["text"], chunk_size, overlap)

        # For each chunk, attach the original source metadata
        for chunk_text in text_chunks:
            all_chunks.append({
                "text": chunk_text,
                "source": page["source"],   # e.g., "Academic_Rules.pdf"
                "page": page["page"]        # e.g., 3
            })

    return all_chunks
