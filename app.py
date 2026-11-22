"""DocuMind: a small, local multi-PDF RAG application with verified citations."""

from __future__ import annotations

import html
import os
import tempfile

import streamlit as st

from assets.styles import inject_css
from src.chunker import chunk_pages
from src.embeddings import EMBEDDING_MODEL_NAME, embed_texts, load_embedding_model
from src.pdf_reader import extract_text_from_multiple_pdfs
from src.rag import OLLAMA_MODEL, answer_question
from src.vector_store import build_faiss_index


MAX_FILES = 5
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="centered")


def initialise_state() -> None:
    defaults = {
        "faiss_index": None,
        "chunks": [],
        "embedding_model": None,
        "processed_files": [],
        "page_count": 0,
        "chat_history": [],
        "documents_ready": False,
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)


def reset_documents() -> None:
    st.session_state.faiss_index = None
    st.session_state.chunks = []
    st.session_state.processed_files = []
    st.session_state.chat_history = []
    st.session_state.page_count = 0
    st.session_state.documents_ready = False


def process_files(uploaded_files) -> tuple[list[dict], object, list[str], object, list[str], int]:
    """Extract, chunk, embed, and index uploaded PDFs without retaining files."""
    with tempfile.TemporaryDirectory(prefix="documind_") as temp_dir:
        file_paths: list[tuple[str, str]] = []
        for position, uploaded_file in enumerate(uploaded_files):
            # Do not use the browser-provided name as a filesystem path.
            temp_path = os.path.join(temp_dir, f"upload_{position}.pdf")
            with open(temp_path, "wb") as destination:
                destination.write(uploaded_file.getvalue())
            file_paths.append((temp_path, os.path.basename(uploaded_file.name)))

        pages = extract_text_from_multiple_pdfs(file_paths)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    if not chunks:
        raise ValueError("No readable text was found. The PDFs may be empty or image-only.")

    model = st.session_state.embedding_model or load_embedding_model()
    embeddings = embed_texts([chunk["text"] for chunk in chunks], model)
    index, indexed_chunks = build_faiss_index(embeddings, chunks)
    files = [os.path.basename(item.name) for item in uploaded_files]
    readable_sources = {page["source"] for page in pages}
    empty_files = [name for name in files if name not in readable_sources]
    return indexed_chunks, index, files, model, empty_files, len(pages)


def render_citations(citations: list[dict]) -> None:
    """Render verified source passages as a dedicated evidence section."""
    if not citations:
        return
    st.markdown('<div class="evidence-heading">Sources &amp; Evidence</div>', unsafe_allow_html=True)
    st.caption("Each source below was cited in the answer and verified against retrieved evidence.")
    for citation in citations:
        passage = " ".join(citation["text"].split())
        excerpt = passage if len(passage) <= 280 else f"{passage[:277].rstrip()}…"
        with st.container(border=True):
            st.markdown(f"**📄 {citation['source']}**  ·  Page {citation['page']}  ·  [{citation['id']}]")
            st.markdown('<span class="verified-badge">● Verified evidence</span>', unsafe_allow_html=True)
            st.caption(excerpt)
            with st.expander("View full supporting passage"):
                st.write(citation["text"])


def _metric_value(value, suffix: str = "", decimals: int | None = None) -> str:
    """Format an optional measured value without inventing unavailable data."""
    if value is None:
        return "—"
    if decimals is not None:
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def render_answer_metrics(performance: dict | None, retrieval: dict | None) -> None:
    """Show concise, real diagnostics directly below an answer."""
    if not performance or not retrieval:
        return
    st.caption("Answer run metrics")
    primary = st.columns(4)
    primary[0].metric("Input tokens", _metric_value(performance.get("prompt_tokens")))
    primary[1].metric("Output tokens", _metric_value(performance.get("output_tokens")))
    primary[2].metric(
        "Generation speed",
        _metric_value(performance.get("tokens_per_second"), " tok/s", 1),
    )
    primary[3].metric(
        "End-to-end time",
        _metric_value(performance.get("rag_pipeline_seconds"), " s", 2),
    )

    with st.expander("Retrieval and model details"):
        retrieval_columns = st.columns(3)
        retrieval_columns[0].metric("Retrieved passages", retrieval.get("retrieved_passages", 0))
        retrieval_columns[1].metric("Verified citations", retrieval.get("verified_citations", 0))
        retrieval_columns[2].metric(
            "Top relevance",
            _metric_value(retrieval.get("top_similarity"), decimals=3),
            help="Cosine similarity between the question and the highest-ranked passage. It is a retrieval signal, not an accuracy score.",
        )
        st.caption(
            f"Context mode: {retrieval.get('mode', 'Semantic search')} · "
            f"Model: {OLLAMA_MODEL} · Documents represented: "
            f"{retrieval.get('unique_documents', 0)} · Temperature: 0.1 · Top-K: 5"
        )


def render_document_metrics() -> None:
    """Show the size of the current evidence collection before questions begin."""
    metrics = st.columns(4)
    metrics[0].metric("Documents", len(st.session_state.processed_files))
    metrics[1].metric("Readable pages", st.session_state.page_count)
    metrics[2].metric("Searchable passages", len(st.session_state.chunks))
    metrics[3].metric("Embedding model", "MiniLM")


initialise_state()

inject_css()
st.markdown('<div class="documind-title">📄 DocuMind</div>', unsafe_allow_html=True)
st.markdown('<div class="documind-subtitle">Intelligent answers from your documents.</div>', unsafe_allow_html=True)
st.markdown('<div class="documind-tagline">● Evidence verified · Private and local</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Documents")
    if st.session_state.processed_files:
        for name in st.session_state.processed_files:
            st.write(f"• {name}")
        st.caption(f"{len(st.session_state.chunks)} searchable passages")
    else:
        st.caption("No documents processed yet.")
    st.divider()
    st.caption(f"Answer model: `{OLLAMA_MODEL}`")
    st.caption("Temperature: `0.1` · Retrieval: `Top-K 5`")
    if st.button("Clear documents and chat", use_container_width=True):
        reset_documents()
        st.rerun()

st.subheader("Your documents")
st.caption("Upload up to five PDFs. DocuMind keeps every answer grounded in the documents you provide.")
uploaded_files = st.file_uploader(
    "⇧ Add your PDFs", type=["pdf"], accept_multiple_files=True,
    help="Upload up to five text-based PDF documents for a single evidence collection.",
)
if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.error(f"Please select at most {MAX_FILES} PDFs.")
elif uploaded_files:
    selected_files = "".join(
        '<span class="file-chip">📄 '
        + html.escape(os.path.basename(file.name))
        + "</span>"
        for file in uploaded_files
    )
    st.markdown(f'<div class="selected-files">{selected_files}</div>', unsafe_allow_html=True)

process_disabled = not uploaded_files or len(uploaded_files) > MAX_FILES
if st.button("Process documents", type="primary", disabled=process_disabled):
    try:
        with st.status("Processing documents…", expanded=True) as status:
            st.write("Extracting text page by page…")
            chunks, index, files, model, empty_files, page_count = process_files(uploaded_files)
            st.write("Building the searchable index…")
            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.processed_files = files
            st.session_state.embedding_model = model
            st.session_state.page_count = page_count
            st.session_state.documents_ready = True
            st.session_state.chat_history = []
            if empty_files:
                st.write("No extractable text (skipped): " + ", ".join(empty_files))
            status.update(label=f"Processed {len(files)} document(s)", state="complete")
    except Exception as error:
        st.error(f"Document processing failed: {error}")

st.divider()
st.subheader("Ask a question")
if not st.session_state.documents_ready:
    st.info("Upload and process one or more text-based PDFs to start an evidence-based conversation.")
else:
    render_document_metrics()
    st.caption(
        f"Model: `{OLLAMA_MODEL}` · Embeddings: `{EMBEDDING_MODEL_NAME}` · "
        f"Cosine retrieval · Top-K: 5"
    )
    clear_chat, _ = st.columns([1, 4])
    if clear_chat.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))
                render_answer_metrics(
                    message.get("performance"), message.get("retrieval")
                )

    st.markdown('<div class="suggestion-label">Suggested questions</div>', unsafe_allow_html=True)
    suggestion_columns = st.columns(4)
    suggestions = (
        ("✦ Explore this document", "What is this document about?"),
        ("≡ Short summary", "Give me a short summary."),
        ("✓ Key rules", "What are the key rules?"),
        ("⌕ Find a detail", "What are the important deadlines?"),
    )
    selected_question = None
    for column, (label, question_text) in zip(suggestion_columns, suggestions):
        if column.button(label, key=f"suggestion_{question_text}", use_container_width=True):
            selected_question = question_text

    question = selected_question or st.chat_input("Ask anything about your documents…")
    if question and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question.strip()})
        with st.chat_message("user"):
            st.markdown(question.strip())
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents…"):
                try:
                    result = answer_question(
                        question.strip(),
                        st.session_state.faiss_index,
                        st.session_state.chunks,
                        st.session_state.embedding_model,
                    )
                    answer = result["answer"]
                    citations = result["citations"]
                    performance = result.get("performance")
                    retrieval = result.get("retrieval")
                except (ConnectionError, TimeoutError, RuntimeError) as error:
                    answer = f"Unable to generate an answer: {error}"
                    citations = []
                    performance = None
                    retrieval = None
                except Exception as error:
                    answer = f"Unexpected error: {error}"
                    citations = []
                    performance = None
                    retrieval = None
            st.markdown(answer)
            render_citations(citations)
            render_answer_metrics(performance, retrieval)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "performance": performance,
                "retrieval": retrieval,
            }
        )
