# DocuMind 📄
### Multi-Document RAG Assistant with Citations

> **College Project — Topic 183: Multi-Document RAG with Citations**

DocuMind is a local, fully offline RAG (Retrieval-Augmented Generation) application.  
Upload multiple PDFs, ask questions, and receive grounded answers with exact source citations.

---

## Project Structure

```
DocuMind/
│
├── app.py                  ← Main Streamlit application (UI + pipeline orchestration)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── data/                   ← (Optional) Store test PDF files here
│
└── src/
    ├── __init__.py         ← Makes src/ a Python package
    ├── pdf_reader.py       ← Extracts text + page numbers from PDFs using PyMuPDF
    ├── chunker.py          ← Splits pages into smaller overlapping text chunks
    ├── embeddings.py       ← Converts text to numerical vectors using Sentence Transformers
    ├── vector_store.py     ← Builds and queries the FAISS vector index
    └── rag.py              ← RAG pipeline: prompt builder, Ollama caller, answer_question()
```

---

## How the RAG Pipeline Works

```
User uploads PDFs
      ↓
pdf_reader.py   → Extract text per page (with filename + page number metadata)
      ↓
chunker.py      → Split pages into 500-char overlapping chunks (overlap = 100 chars)
      ↓
embeddings.py   → Convert each chunk to a 384-dimensional vector (all-MiniLM-L6-v2)
      ↓
vector_store.py → Store all vectors in a FAISS index

─── User asks a question ───

User question
      ↓
embeddings.py   → Convert question to a 384-dim vector
      ↓
vector_store.py → Search FAISS for top-5 most similar chunk vectors
      ↓
rag.py          → Build a grounding prompt with the retrieved chunks
      ↓
Ollama / Qwen   → Generate a grounded answer
      ↓
app.py          → Display answer + source citations (from retrieval metadata)
```

---

## Setup Instructions

### Step 1 — Install Python Dependencies

Make sure you have Python 3.10 or later installed.

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — the web UI framework
- `PyMuPDF` — PDF text extraction
- `sentence-transformers` — local embedding model
- `faiss-cpu` — vector similarity search
- `numpy` — array math
- `requests` — HTTP calls to Ollama

---

### Step 2 — Install Ollama

Ollama is a tool that lets you run LLMs locally on your machine.

**Windows / macOS / Linux:**
Download from: https://ollama.com/download

After installing, verify Ollama works:
```bash
ollama --version
```

---

### Step 3 — Pull the Qwen Model

We use Qwen 2.5 (1.5 billion parameter version) which is fast and lightweight.

```bash
ollama pull qwen2.5:1.5b
```

> **Note:** Other Qwen variants you can use:
> - `qwen2.5:3b` — Better quality, needs more RAM
> - `qwen2.5:7b` — Best quality, needs ~8GB RAM
> - `qwen2:1.5b` — Older generation

To change the model, open `src/rag.py` and update this line:
```python
OLLAMA_MODEL = "qwen2.5:1.5b"   ← Change this to any ollama model name
```

---

### Step 4 — Start Ollama (keep it running)

```bash
ollama serve
```

This starts the Ollama server at `http://localhost:11434`.  
Keep this terminal open while using DocuMind.

---

### Step 5 — Run DocuMind

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## How to Use

1. **Upload PDFs** — Click the upload area and select 1–5 PDF files.
2. **Process Documents** — Click the "⚙️ Process Documents" button.
3. **Wait** — The app will extract text, chunk it, embed it, and build the FAISS index.
4. **Ask questions** — Type your question in the chat box at the bottom.
5. **View citations** — Each answer shows which PDF file and page number the answer came from.
6. **Reset** — Click "🗑️ Clear & Reset" to start a new session with different documents.

---

## LLM Performance Metrics

Expand **How performance is calculated** under the chat heading for the metric definitions. Each generated answer also has a **Performance for this answer** panel showing available measurements:

| Metric | Source / calculation |
|---|---|
| Request latency | Client-side `time.perf_counter()` around the Ollama HTTP request |
| Ollama total, model load, prompt processing, generation | Ollama's nanosecond durations divided by 1,000,000,000 |
| Prompt and output tokens | Token counts reported by Ollama |
| Prompt throughput | Prompt tokens ÷ prompt-processing seconds |
| Generation throughput | Output tokens ÷ generation seconds |
| Query embedding, document retrieval | Independently timed with `time.perf_counter()` |
| RAG pipeline total | Time for the complete question-answer path, including Ollama generation |

These measurements describe latency and throughput; they do **not** measure answer accuracy.

### Answer quality and accuracy

- **Answer grounding (proxy)** compares the generated answer embedding with each retrieved passage embedding and displays the highest cosine similarity, along with that passage's citation. This is a semantic similarity signal only; it is not a probability, entailment check, or proof that the answer is factually supported.
- **Reference benchmark** lets you upload a JSON array of questions and verified expected answers. The app runs the RAG pipeline for each question and reports:
  - **Exact-match accuracy**: the share of answers whose lowercased, punctuation-insensitive word sequence exactly matches the reference. Valid paraphrases can score as incorrect.
  - **Mean token F1**: the average per-answer F1 based on overlapping words, useful for partial matches but not a substitute for human review.

Use representative questions and carefully verified references for a meaningful benchmark. These scores describe only that dataset and run; review individual answers and citations before drawing conclusions.

JSON format:

```json
[
  {
    "question": "What is the attendance requirement?",
    "expected_answer": "The minimum attendance is 75%."
  }
]
```

Timings vary with hardware, model warm-up, and prompt length.

---

## Configuration

All key parameters are at the top of their respective files:

| Parameter | File | Default | Description |
|---|---|---|---|
| `OLLAMA_MODEL` | `src/rag.py` | `qwen2.5:1.5b` | The Ollama model to use |
| `CHUNK_SIZE` | `app.py` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `app.py` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `src/rag.py` | `5` | How many chunks to retrieve per query |
| `EMBEDDING_MODEL_NAME` | `src/embeddings.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `MAX_FILES` | `app.py` | `5` | Maximum PDFs per session |

---

## Demo Test Cases

Use these to demonstrate your project to the jury:

**Test 1 — Single document answer:**
Upload one PDF and ask a question whose answer is clearly on a specific page.
The answer should cite that exact page.

**Test 2 — Multi-document answer:**
Upload 2+ PDFs where the question requires combining information from both.
The sources list should show pages from multiple documents.

**Test 3 — Out-of-context question (hallucination test):**
Ask a question that is NOT related to any uploaded document.
Expected response: *"I couldn't find sufficient information about this in the uploaded documents."*

---

## Key Libraries Explained

| Library | Purpose |
|---|---|
| **Streamlit** | Turns a Python script into an interactive web UI |
| **PyMuPDF (fitz)** | Reads PDF files and extracts text page by page |
| **Sentence Transformers** | Downloads and runs a local model to convert text into vectors |
| **FAISS** | Facebook's library for fast vector similarity search |
| **NumPy** | Handles the numerical arrays (vectors) |
| **Requests** | Sends HTTP calls to the local Ollama API |
| **Ollama** | Runs the Qwen LLM locally without any cloud API |

---

## Troubleshooting

**"Could not connect to Ollama"**
→ Run `ollama serve` in a separate terminal and keep it running.

**"Model not found"**
→ Run `ollama pull qwen2.5:1.5b` to download the model.

**"No text could be extracted"**
→ Your PDF may be a scanned image. PyMuPDF only extracts digital text.

**Slow embedding generation**
→ This is normal on the first run. The embedding model downloads once (~90MB) and is cached.

**Out of memory**
→ Try using a smaller Qwen model like `qwen2.5:1.5b` or reduce `CHUNK_SIZE`.

---

## Performance and Accuracy — Quick Guide

This section is a plain-language summary of the numbers DocuMind shows.

### How fast is it?

- **Request latency**: How long the app waits for Ollama to return an answer.
- **Model load time**: How long Ollama takes to get the model ready. This can be higher the first time.
- **Prompt processing**: How long the model takes to read the question and document passages.
- **Generation time**: How long the model takes to write its answer.
- **Prompt and output tokens**: The amount of text the model read and generated, counted in tokens (small pieces of text).
- **Prompt and generation speed**: How many tokens the model reads or writes per second.
- **Question embedding and document retrieval**: How long it takes to turn the question into a vector and find related passages in the PDFs.
- **RAG pipeline total**: The overall time for the app to search the documents and generate an answer.

These are speed measurements, not a rating of whether an answer is right. They can change depending on the computer, model warm-up, and question length.

### Is the answer supported by the PDFs?

The app shows **answer grounding similarity** by comparing the answer with the passages it retrieved. It displays the most similar passage and its source. A higher similarity means the wording is more alike; it does **not** prove that the answer is correct or supported by the passage.

### How do we measure answer accuracy?

Use **Evaluate answer accuracy with a reference benchmark** in the app. Upload a JSON file with questions and their verified expected answers. The app answers each question and compares its answer with the reference:

- **Exact-match accuracy**: The percentage of answers that match the expected answer after ignoring capitalization and punctuation. A correct answer written differently may not count as a match.
- **Mean token F1**: A partial-match score. It gives credit when the generated answer and expected answer share words, even if they are not identical.

The benchmark scores apply only to the questions and expected answers you provide. Use a good set of representative questions, and review the answers yourself. DocuMind does not have one built-in accuracy percentage; an accuracy result is available only after you run it with a verified benchmark.

Needed Commands -
cd "D:\Gen AI Jury\Gen-ai-project"
dir Gen-ai-project
python -m streamlit run app.py