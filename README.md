# DocuMind

DocuMind is a deliberately small local multi-document RAG app. Upload up to five
text-based PDFs, ask a question, and receive a concise answer with only the
supporting passages that the answer actually cites.

## What makes the citations trustworthy

1. PyMuPDF extracts each PDF page separately and retains its original, 1-based
   page number and original filename.
2. Sentence-aware chunks keep this metadata plus a stable internal chunk ID.
3. Unit-normalised `all-MiniLM-L6-v2` embeddings are searched by exact cosine
   similarity in FAISS. Exact duplicate passages are removed from the results.
4. Retrieved passages receive short request-scoped IDs (`C1`, `C2`, …). The LLM
   sees only those IDs and the passages—not filenames or pages.
5. Every non-empty generated claim must contain valid supplied IDs and share a
   meaningful term with every cited passage. An unknown, fabricated, missing, or
   clearly unrelated citation rejects the whole generated answer and shows the
   standard abstention instead.
6. DocuMind resolves a valid ID itself to filename, page, and exact passage.

This validation proves that a displayed citation is from the retrieved evidence.
It cannot prove the LLM's wording is a perfect entailment of that passage, so the
app intentionally abstains when an answer is not correctly cited.

### Document overview requests

Requests such as “summarize this document,” “give me an overview,” or “what does
this document say?” use a dedicated overview path. It selects representative
pages across the uploaded documents rather than lowering the factual-question
similarity threshold. The overview remains concise and uses the same validated
`[C#]` citations as ordinary answers.

## Setup (Windows)

Install Python 3.10+ and [Ollama](https://ollama.com/download), then from the
project directory run:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

In a second terminal:

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

## Model and laptop configuration

Qwen3 4B was tested first. On the installed Ollama 0.40.17 server it ignored
the API's non-thinking setting and returned internal reasoning in the answer
field, which correctly failed DocuMind's citation validator. The app therefore
uses the already-installed non-thinking `qwen2.5:3b` with low temperature
(`0.1`), a 300-token answer cap, and a 4096-token context. It returned a valid
`[C1]`-cited answer in the same live Ollama test.

The machine inspected for this project has an RTX 3050 Laptop GPU with 4 GB VRAM
and 16 GB system RAM. The 3B fallback is a safer fit than Qwen3 4B for this GPU.
Keep documents small for the demo and close GPU-heavy applications.

Run the real Ollama/model/throughput check after installation:

```powershell
python validation.py
```

The live model check is still useful after any driver, model, or Ollama update.

## Tests

The repeatable test suite creates three sample PDFs and checks extraction,
page-number preservation, sentence chunking, duplicate removal, retrieval among
similar passages, multi-document citation resolution, missing-information
abstention, and fabricated-citation rejection:

```powershell
python -m unittest discover -s tests -v
```

These tests do not need Ollama: they use a deterministic test embedder and mock
only generation so source/page validation is directly testable. For a jury demo,
also run the same questions in the Streamlit app and record the
actual answers and displayed passages.

## Suggested jury demo questions

Use three short PDFs whose facts you know and test:

- One-PDF fact: “What is the attendance requirement?”
- Two-PDF synthesis: “State the deadline from one document and the budget from another.”
- Three-PDF synthesis: ask for one known fact from each PDF.
- Missing fact: ask about a topic absent from all PDFs.
- Similar passages: include “application deadline” and “assignment deadline.”
- Contradiction: include two different values; the answer should cite each claim
  separately rather than silently inventing a reconciliation.

For every displayed `[C#]`, open its supporting-passage panel and verify the
shown filename, original page number, and exact text.

## What the interface shows

- Collection size: uploaded documents, readable pages, and searchable passages.
- Per-answer model measurements: actual input/output token counts, generation
  speed, and end-to-end RAG time reported by Ollama and the application.
- Retrieval details: number of passages searched, verified citation count, number
  of represented documents, and top cosine relevance. Relevance is explicitly
  labelled as a retrieval signal, not an answer-accuracy score.

## Project layout

```text
app.py                 Streamlit interface and document workflow
src/pdf_reader.py      Page-by-page PDF extraction
src/chunker.py         Sentence-aware chunking and chunk metadata
src/embeddings.py      Normalised MiniLM embeddings
src/vector_store.py    Exact cosine FAISS search and deduplication
src/rag.py             Prompting, abstention, and citation validation
tests/test_documind.py Repeatable reliability checks
validation.py          Real local Ollama model/throughput check
```
