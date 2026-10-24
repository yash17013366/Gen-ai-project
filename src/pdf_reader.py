"""
pdf_reader.py
--------------
This module is responsible for extracting text from PDF files.

We use PyMuPDF (imported as 'fitz') for this.

PyMuPDF can open a PDF file, iterate over each page,
and extract all the text content from that page.

For each piece of text we extract, we keep track of:
  - The text itself
  - The page number it came from
  - The filename of the PDF it came from

This metadata is CRITICAL because it allows us to show citations later.
"""

import pymupdf as fitz


def extract_text_from_pdf(pdf_path: str, filename: str) -> list[dict]:
    """
    Opens a PDF file and extracts text from every page.

    Returns a list of dictionaries. Each dictionary represents one page:
    {
        "text": "The text content of this page...",
        "source": "Academic_Rules.pdf",
        "page": 3
    }

    Parameters:
    - pdf_path: The full file path to the PDF on disk.
    - filename:  The original name of the PDF (e.g., "Academic_Rules.pdf").
                 We pass this separately so the displayed name is clean.
    """

    pages_data = []  # We will collect all pages in this list

    # Open the PDF using PyMuPDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF '{filename}': {e}")

    # fitz numbers pages starting from 0, but readers expect page 1 to mean
    # the first page, so we add 1 when storing the page number.
    for page_index in range(len(doc)):
        page = doc[page_index]

        # Extract all the plain text from this page
        page_text = page.get_text()

        # Skip pages that have no text (e.g., image-only pages)
        if not page_text.strip():
            continue

        # Store the page text along with its source metadata
        pages_data.append({
            "text": page_text,
            "source": filename,
            "page": page_index + 1  # Convert 0-indexed to 1-indexed page number
        })

    doc.close()

    return pages_data


def extract_text_from_multiple_pdfs(pdf_files: list[tuple]) -> list[dict]:
    """
    Processes multiple PDFs and combines all their extracted pages.

    Parameters:
    - pdf_files: A list of tuples, each tuple is (pdf_path, filename)
                 For example: [("/tmp/file1.pdf", "Rules.pdf"), ...]

    Returns a flat list of all page dictionaries from all PDFs.
    """

    all_pages = []

    for pdf_path, filename in pdf_files:
        pages = extract_text_from_pdf(pdf_path, filename)
        all_pages.extend(pages)  # Add this PDF's pages to our combined list

    return all_pages
