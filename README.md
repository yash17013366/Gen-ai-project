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
