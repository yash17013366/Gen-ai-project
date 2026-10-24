"""
app.py
-------
DocuMind — Multi-Document RAG Assistant with Citations

This is the main application file. It runs the Streamlit UI and
orchestrates the entire RAG pipeline.

WHAT THIS FILE DOES:
1. Renders the upload interface so users can upload PDF files.
2. When the user clicks "Process Documents", it:
   a. Saves the uploaded files temporarily to disk.
   b. Extracts text from each PDF (pdf_reader.py).
   c. Splits text into chunks (chunker.py).
   d. Converts chunks to embeddings (embeddings.py).
   e. Builds a FAISS vector index (vector_store.py).
3. Renders a chat interface for the user to ask questions.
4. For each question, calls the full RAG pipeline (rag.py)
   and displays the answer with source citations.

HOW TO RUN:
  streamlit run app.py
"""

import json
import os
import tempfile
import streamlit as st

# Import our custom modules
from src.pdf_reader import extract_text_from_multiple_pdfs
from src.chunker import chunk_pages
from src.embeddings import load_embedding_model, embed_texts
from src.vector_store import build_faiss_index
from src.rag import answer_question, OLLAMA_MODEL
from validation import calculate_answer_scores, summarize_benchmark

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

MAX_FILES = 5          # Maximum number of PDFs the user can upload
CHUNK_SIZE = 500       # Characters per chunk
CHUNK_OVERLAP = 100    # Overlapping characters between chunks


# -----------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind — Multi-Document RAG",
    page_icon="📄",
    layout="centered"
)


# -----------------------------------------------------------------
# CUSTOM CSS — Makes the UI look clean and professional
# -----------------------------------------------------------------

st.markdown("""
<style>
    /* ---- Global font and background ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Main container ---- */
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ---- App title block ---- */
    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .app-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
        margin-top: 0.3rem;
        letter-spacing: 0.05em;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        border-left: 3px solid #a78bfa;
        padding-left: 0.6rem;
    }

    /* ---- File list badges ---- */
    .file-badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #c4b5fd;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 3px 3px;
        font-size: 0.82rem;
    }

    /* ---- Chat messages ---- */
    .chat-user {
        background: rgba(96, 165, 250, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #bfdbfe;
        font-size: 0.95rem;
    }

    .chat-assistant {
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 2px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #d1fae5;
        font-size: 0.95rem;
    }

    /* ---- Sources block ---- */
    .sources-block {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
    }

    .sources-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .source-item {
        font-size: 0.82rem;
        color: #a78bfa;
        margin: 2px 0;
    }

    /* ---- Status messages ---- */
    .status-success {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        color: #6ee7b7;
        font-size: 0.9rem;
        margin: 8px 0;
    }

    .status-info {
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 8px;
        padding: 10px 16px;
        color: #93c5fd;
        font-size: 0.9rem;
        margin: 8px 0;
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }

    /* ---- Streamlit overrides ---- */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: opacity 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(167, 139, 250, 0.4);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(167, 139, 250, 0.04);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# SESSION STATE INITIALIZATION
# Session state is Streamlit's way of persisting variables across
# UI interactions (since Streamlit re-runs the entire script on
# each user action).
# -----------------------------------------------------------------

def init_session_state():
    """Initialize all session state variables if they don't exist yet."""

    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None       # The FAISS index object

    if "chunks" not in st.session_state:
        st.session_state.chunks = []              # All text chunks with metadata

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None   # Loaded SentenceTransformer

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []     # Names of successfully processed PDFs

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []        # List of {role, content, sources}

    if "documents_ready" not in st.session_state:
        st.session_state.documents_ready = False  # Whether documents have been processed

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None


init_session_state()


def render_performance_metrics(performance: dict):
    """Display measured timings and token throughput for one answer."""
    metric_specs = (
        ("Request latency", "request_latency_seconds", "Client wall-clock time for the Ollama API request.", "seconds"),
        ("Ollama total", "ollama_total_seconds", "Total processing time reported by Ollama.", "seconds"),
        ("Model load", "model_load_seconds", "Time Ollama spent loading the model.", "seconds"),
        ("Prompt processing", "prompt_processing_seconds", "Time spent processing input tokens.", "seconds"),
        ("Prompt tokens", "prompt_tokens", "Input tokens counted by Ollama.", "count"),
        ("Prompt throughput", "prompt_tokens_per_second", "Prompt tokens divided by prompt-processing seconds.", "tokens/sec"),
        ("Generation", "generation_seconds", "Time spent generating output tokens.", "seconds"),
        ("Output tokens", "output_tokens", "Output tokens counted by Ollama.", "count"),
        ("Generation throughput", "tokens_per_second", "Output tokens divided by generation seconds.", "tokens/sec"),
        ("Query embedding", "query_embedding_seconds", "Time to embed the user's question.", "seconds"),
        ("Document retrieval", "retrieval_seconds", "Time to search the FAISS index.", "seconds"),
        ("RAG pipeline total", "rag_pipeline_seconds", "Embedding, retrieval, prompt creation, and generation.", "seconds"),
    )
    available_metrics = [
        (label, performance.get(key), help_text, unit)
        for label, key, help_text, unit in metric_specs
        if performance.get(key) is not None
    ]

    with st.expander("📈 Performance for this answer"):
        for start in range(0, len(available_metrics), 3):
            columns = st.columns(3)
            for column, (label, value, help_text, unit) in zip(
                columns, available_metrics[start:start + 3]
            ):
                if unit == "seconds":
                    display_value = f"{value:.3f} s"
                elif unit == "tokens/sec":
                    display_value = f"{value:.2f} tokens/s"
                else:
                    display_value = str(value)
                column.metric(label, display_value, help=help_text)


def render_quality_metrics(quality: dict):
    """Display the embedding-based grounding proxy with its strongest source."""
    grounding = quality.get("grounding_proxy")
    if grounding is None:
        return

    with st.expander("🔎 Answer grounding (proxy, not accuracy)"):
        st.metric(
            "Best context cosine similarity",
            f"{grounding['cosine_similarity']:.3f}",
            help=(
                "Cosine similarity between the answer and each retrieved chunk; "
                "the highest score is shown. Similarity does not prove the answer "
                "is entailed by or factually correct according to the document."
            ),
        )
        st.caption(
            "Closest retrieved passage: "
            f"{grounding['source']} — Page {grounding['page']}. "
            "Use the reference-answer benchmark below to measure exact-match "
            "accuracy on questions with verified answers."
        )


# -----------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------

st.markdown("""
<div class="title-block">
    <h1 class="app-title">📄 DocuMind</h1>
    <p class="app-subtitle">Multi-Document RAG Assistant with Citations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# -----------------------------------------------------------------
# SECTION 1 — DOCUMENT UPLOAD
# -----------------------------------------------------------------

st.markdown('<div class="section-header">📂 Upload Documents</div>', unsafe_allow_html=True)

st.caption(f"Upload up to **{MAX_FILES}** PDF documents. Only PDF files are accepted.")

# Streamlit file uploader — accepts multiple files, PDF only
uploaded_files = st.file_uploader(
    label="Choose PDF files",
    type=["pdf"],              # Only allow PDF uploads
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# -----------------------------------------------------------------
# FILE VALIDATION
# -----------------------------------------------------------------

# Check if the user has uploaded more than the maximum allowed
if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.error(
        f"⚠️ You uploaded {len(uploaded_files)} files, but the maximum is {MAX_FILES}. "
        f"Please remove some files before processing."
    )
    uploaded_files = []  # Clear the uploaded files to prevent processing

# Show the uploaded file names as badges
if uploaded_files:
    st.markdown(
        " ".join([f'<span class="file-badge">✓ {f.name}</span>' for f in uploaded_files]),
        unsafe_allow_html=True
    )
    st.caption(f"{len(uploaded_files)} file(s) selected")

st.markdown("")  # Spacing


# -----------------------------------------------------------------
# PROCESS DOCUMENTS BUTTON
# -----------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    process_button = st.button("⚙️ Process Documents", use_container_width=True)

with col2:
    reset_button = st.button("🗑️ Clear & Reset", use_container_width=True)


# -----------------------------------------------------------------
# RESET LOGIC
# Clears everything so the user can start a fresh session.
# -----------------------------------------------------------------

if reset_button:
    st.session_state.faiss_index = None
    st.session_state.chunks = []
    st.session_state.embedding_model = None
    st.session_state.processed_files = []
    st.session_state.chat_history = []
    st.session_state.documents_ready = False
    st.session_state.benchmark_results = None
    st.success("✅ Session cleared. You can now upload new documents.")
    st.rerun()


# -----------------------------------------------------------------
# DOCUMENT PROCESSING LOGIC
# This runs when the user clicks "Process Documents"
# -----------------------------------------------------------------

if process_button:

    # Guard: make sure at least one file was uploaded
    if not uploaded_files:
        st.error("❌ Please upload at least one PDF file before processing.")

    else:
        # We use a Streamlit spinner to show activity during processing
        with st.spinner("🔄 Processing documents... This may take a moment."):

            try:
                # --------------------------------------------------
                # STEP 1: Save uploaded files to a temporary directory
                # Streamlit gives us file-like objects, but PyMuPDF
                # needs actual file paths on disk.
                # We use Python's tempfile module to create temporary files.
                # --------------------------------------------------
                temp_dir = tempfile.mkdtemp()
                pdf_file_paths = []  # Will hold (path_on_disk, original_filename)

                for uploaded_file in uploaded_files:
                    # Write the uploaded content to a temp file
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    pdf_file_paths.append((temp_path, uploaded_file.name))

                # --------------------------------------------------
                # STEP 2: Extract text from all PDFs
                # pdf_reader.py opens each PDF and extracts text
                # from every page, keeping track of page numbers.
                # --------------------------------------------------
                st.info("📖 Extracting text from PDFs...")
                all_pages = extract_text_from_multiple_pdfs(pdf_file_paths)

                if not all_pages:
                    st.error("❌ No text could be extracted from the uploaded PDFs. "
                             "They may be image-only or scanned documents.")
                    st.stop()

                # --------------------------------------------------
                # STEP 3: Split pages into smaller chunks
                # chunker.py splits large page text into smaller
                # overlapping pieces for more precise retrieval.
                # --------------------------------------------------
                st.info("✂️ Splitting text into chunks...")
                all_chunks = chunk_pages(all_pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

                if not all_chunks:
                    st.error("❌ No text chunks were created. The PDFs may be empty.")
                    st.stop()

                # --------------------------------------------------
                # STEP 4: Load the embedding model (only load once)
                # We cache it in session_state so we don't reload
                # it every time (it takes a few seconds to load).
                # --------------------------------------------------
                if st.session_state.embedding_model is None:
                    st.info("🧠 Loading embedding model (first time may take a minute)...")
                    st.session_state.embedding_model = load_embedding_model()

                # --------------------------------------------------
                # STEP 5: Generate embeddings for all chunks
                # We extract just the text from each chunk dict
                # and pass it to the embedding model.
                # --------------------------------------------------
                st.info("🔢 Generating embeddings for all chunks...")
                chunk_texts = [chunk["text"] for chunk in all_chunks]
                embeddings = embed_texts(chunk_texts, st.session_state.embedding_model)

                # --------------------------------------------------
                # STEP 6: Build the FAISS index
                # FAISS stores all embedding vectors and allows
                # us to quickly find the most similar ones to a query.
                # --------------------------------------------------
                st.info("📦 Building FAISS vector index...")
                faiss_index, indexed_chunks = build_faiss_index(embeddings, all_chunks)

                # --------------------------------------------------
                # STEP 7: Save results in session state
                # --------------------------------------------------
                st.session_state.faiss_index = faiss_index
                st.session_state.chunks = indexed_chunks
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.documents_ready = True
                st.session_state.chat_history = []  # Clear old chat on new upload
                st.session_state.benchmark_results = None

                # --------------------------------------------------
                # SUCCESS: Show summary
                # --------------------------------------------------
                num_docs = len(uploaded_files)
                num_chunks = len(all_chunks)

                st.markdown(f"""
<div class="status-success">
    ✅ <strong>{num_docs} document(s) processed successfully!</strong><br>
    📊 Total chunks created: <strong>{num_chunks}</strong><br>
    🔍 Ready to answer your questions.
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")


# -----------------------------------------------------------------
# SHOW CURRENTLY PROCESSED FILES
# -----------------------------------------------------------------

if st.session_state.documents_ready and st.session_state.processed_files:
    st.markdown("")
    st.markdown(
        f'<div class="status-info">📁 <strong>{len(st.session_state.processed_files)} document(s)</strong> available for questions: '
        + ", ".join([f"<em>{f}</em>" for f in st.session_state.processed_files])
        + "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------
# SECTION 2 — CHAT INTERFACE
# Only shown once documents have been processed.
# -----------------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)

if not st.session_state.documents_ready:
    st.markdown(
        '<div class="status-info">ℹ️ Please upload and process PDF documents to start asking questions.</div>',
        unsafe_allow_html=True
    )

else:
    st.caption(f"Using model: **{OLLAMA_MODEL}** via Ollama")
    with st.expander("How performance is calculated"):
        st.markdown(
            """
These are **speed and throughput measurements**, not an answer-accuracy score.

- **Request latency** uses Python's `time.perf_counter()` around the Ollama HTTP request.
- **Ollama total, model load, prompt processing, and generation times** come from Ollama's response in nanoseconds and are converted to seconds by dividing by 1,000,000,000.
- **Prompt/output token counts** come from Ollama's tokenizer counters.
- **Prompt throughput** = prompt tokens ÷ prompt-processing seconds.
- **Generation throughput** = output tokens ÷ generation seconds.
- **Query embedding** and **document retrieval** are timed separately; **RAG pipeline total** measures the complete question-answer path.

Timings vary with hardware, model warm-up, and prompt length. The answer-grounding cosine similarity is a semantic similarity proxy, not a probability or proof of factual support. Accuracy is measured only by comparing answers to verified references in the benchmark below.
"""
        )

    with st.expander("🧪 Evaluate answer accuracy with a reference benchmark"):
        st.markdown(
            "Create a JSON file with questions and **verified expected answers**. "
            "Exact-match accuracy is the percentage of generated answers that match "
            "after lowercasing and ignoring punctuation. Token F1 gives partial credit "
            "for overlapping words."
        )
        benchmark_template = json.dumps(
            [
                {
                    "question": "Replace with a question answerable from your PDFs",
                    "expected_answer": "Replace with the verified answer from your PDFs",
                }
            ],
            indent=2,
        )
        st.download_button(
            "Download benchmark JSON template",
            data=benchmark_template,
            file_name="documind_benchmark_template.json",
            mime="application/json",
        )
        benchmark_file = st.file_uploader(
            "Upload your benchmark JSON",
            type=["json"],
            key="benchmark_uploader",
        )

        if st.button("Run answer accuracy benchmark", disabled=benchmark_file is None):
            try:
                benchmark_cases = json.loads(benchmark_file.getvalue())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                st.error(f"Invalid benchmark JSON: {error}")
            else:
                valid_cases = (
                    isinstance(benchmark_cases, list)
                    and bool(benchmark_cases)
                    and all(
                        isinstance(case, dict)
                        and isinstance(case.get("question"), str)
                        and bool(case["question"].strip())
                        and isinstance(case.get("expected_answer"), str)
                        and bool(case["expected_answer"].strip())
                        for case in benchmark_cases
                    )
                )
                if not valid_cases:
                    st.error(
                        "The benchmark must be a non-empty JSON array of objects, "
                        "each with non-empty string fields 'question' and "
                        "'expected_answer'."
                    )
                else:
                    results = []
                    try:
                        with st.spinner(
                            f"Evaluating {len(benchmark_cases)} benchmark questions..."
                        ):
                            for case in benchmark_cases:
                                result = answer_question(
                                    user_question=case["question"],
                                    faiss_index=st.session_state.faiss_index,
                                    chunks=st.session_state.chunks,
                                    embedding_model=st.session_state.embedding_model,
                                )
                                scores = calculate_answer_scores(
                                    result["answer"], case["expected_answer"]
                                )
                                results.append({
                                    "question": case["question"],
                                    "expected_answer": case["expected_answer"],
                                    "answer": result["answer"],
                                    **scores,
                                    "grounding_proxy": result["quality"][
                                        "grounding_proxy"
                                    ],
                                })
                    except (ConnectionError, TimeoutError, RuntimeError) as error:
                        st.error(f"Benchmark stopped because answer generation failed: {error}")
                    else:
                        st.session_state.benchmark_results = {
                            "file_name": benchmark_file.name,
                            "results": results,
                            "summary": summarize_benchmark(results),
                        }

        benchmark_results = st.session_state.benchmark_results
        if benchmark_results:
            st.caption(f"Results for `{benchmark_results['file_name']}`")
            summary = benchmark_results["summary"]
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Exact-match accuracy",
                f"{summary['exact_match_accuracy']:.1%}",
                help="Exact normalized answer matches divided by benchmark questions.",
            )
            metric_columns[1].metric(
                "Mean token F1",
                f"{summary['mean_token_f1']:.1%}",
                help="Macro-average of per-answer token precision/recall F1.",
            )
            metric_columns[2].metric("Questions", summary["question_count"])
            st.caption(
                "These scores apply only to this user-provided benchmark and model run. "
                "Review each answer; exact match can mark valid paraphrases as incorrect."
            )
            st.table([
                {
                    "Question": item["question"],
                    "Expected answer": item["expected_answer"],
                    "Generated answer": item["answer"],
                    "Exact match": item["exact_match"],
                    "Token F1": f"{item['token_f1']:.1%}",
                    "Grounding similarity (proxy)": (
                        f"{item['grounding_proxy']['cosine_similarity']:.3f}"
                        if item["grounding_proxy"] is not None
                        else "N/A"
                    ),
                }
                for item in benchmark_results["results"]
            ])

    # ------------------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # We iterate over the chat history stored in session state and
    # render each message with appropriate styling.
    # ------------------------------------------------------------------
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 <strong>You:</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # Assistant message
            st.markdown(
                f'<div class="chat-assistant">🤖 <strong>DocuMind:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )

            # If this message has source citations, display them
            if message.get("sources"):
                sources_html = '<div class="sources-block"><div class="sources-title">📎 Sources</div>'
                for i, src in enumerate(message["sources"], start=1):
                    sources_html += f'<div class="source-item">{i}. {src["source"]} — Page {src["page"]}</div>'
                sources_html += "</div>"
                st.markdown(sources_html, unsafe_allow_html=True)

            if message.get("performance"):
                render_performance_metrics(message["performance"])
            if message.get("quality"):
                render_quality_metrics(message["quality"])

    # ------------------------------------------------------------------
    # QUESTION INPUT
    # ------------------------------------------------------------------
    st.markdown("")
    user_question = st.chat_input("Ask a question about your uploaded documents...")

    if user_question:

        # Validate: non-empty question
        if not user_question.strip():
            st.warning("Please enter a valid question.")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question,
                "sources": []
            })

            # ----------------------------------------------------------
            # Run the RAG pipeline
            # ----------------------------------------------------------
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    result = answer_question(
                        user_question=user_question,
                        faiss_index=st.session_state.faiss_index,
                        chunks=st.session_state.chunks,
                        embedding_model=st.session_state.embedding_model
                    )

                    answer = result["answer"]
                    sources = result["sources"]
                    performance = result.get("performance")
                    quality = result.get("quality")

                except ConnectionError as e:
                    answer = f"⚠️ Ollama connection error: {str(e)}"
                    sources = []
                    performance = None
                    quality = None

                except TimeoutError as e:
                    answer = f"⏱️ Timeout: {str(e)}"
                    sources = []
                    performance = None
                    quality = None

                except RuntimeError as e:
                    answer = f"❌ Model error: {str(e)}"
                    sources = []
                    performance = None
                    quality = None

                except Exception as e:
                    answer = f"❌ Unexpected error: {str(e)}"
                    sources = []
                    performance = None
                    quality = None

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "performance": performance,
                "quality": quality,
            })

            # Rerun to show the updated chat history
            st.rerun()


# -----------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.75rem;">'
    'DocuMind · Multi-Document RAG · Powered by FAISS + Sentence Transformers + Qwen via Ollama'
    '</p>',
    unsafe_allow_html=True
)
