"""Small repeatable checks for DocuMind's extraction, retrieval, and citations.

They use generated PDFs and a deterministic test embedder, so no Ollama model is
needed to verify source/page mapping.  Run with: python -m unittest discover -s tests
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf

from src.chunker import chunk_pages, split_text_into_chunks
from src.embeddings import embed_texts
from src.pdf_reader import extract_text_from_multiple_pdfs
from src.rag import (
    NOT_FOUND_ANSWER,
    answer_question,
    is_summary_request,
    select_summary_chunks,
    validate_cited_answer,
    validate_summary_answer,
)
from src.vector_store import build_faiss_index, search_index


class KeywordEmbedder:
    """A tiny predictable embedder for retrieval tests, not used by the app."""

    vocabulary = (
        "attendance",
        "seventyfive",
        "submission",
        "deadline",
        "may",
        "budget",
        "five",
        "thousand",
        "application",
        "assignment",
        "contradictory",
        "monday",
        "friday",
    )

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            lower = text.lower().replace("75%", "seventyfive").replace("5000", "five thousand")
            rows.append([float(word in lower) for word in self.vocabulary])
        return np.asarray(rows, dtype="float32")


def make_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class DocuMindPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.rules = root / "rules.pdf"
        cls.schedule = root / "schedule.pdf"
        cls.finance = root / "finance.pdf"
        make_pdf(cls.rules, ["Attendance is 75%.", "The review is Monday."])
        make_pdf(cls.schedule, ["The assignment submission deadline is 10 May.", "The application deadline is 30 May."])
        make_pdf(cls.finance, ["The project budget is 5000 rupees.", "The schedule is contradictory: review is Friday."])
        pages = extract_text_from_multiple_pdfs(
            [(str(cls.rules), cls.rules.name), (str(cls.schedule), cls.schedule.name), (str(cls.finance), cls.finance.name)]
        )
        cls.chunks = chunk_pages(pages, chunk_size=120, overlap=20)
        cls.model = KeywordEmbedder()
        cls.index, cls.chunks = build_faiss_index(embed_texts([item["text"] for item in cls.chunks], cls.model), cls.chunks)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_pdf_pages_keep_original_numbers_and_skip_empty_page(self):
        empty = Path(self.temp_dir.name) / "empty.pdf"
        make_pdf(empty, ["", "Page two has text."])
        pages = extract_text_from_multiple_pdfs([(str(empty), empty.name)])
        self.assertEqual([(page["source"], page["page"]) for page in pages], [("empty.pdf", 2)])

    def test_sentence_chunks_keep_boundary_context_and_metadata(self):
        chunks = split_text_into_chunks("First sentence ends. Second sentence is important. Third ends.", 45, 12)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Second sentence", " ".join(chunks))
        page_chunks = chunk_pages([{"text": "One. Two.", "source": "a.pdf", "page": 7}], 20, 5)
        self.assertEqual(page_chunks[0]["source"], "a.pdf")
        self.assertEqual(page_chunks[0]["page"], 7)
        self.assertTrue(page_chunks[0]["chunk_id"].startswith("chunk-"))

    def test_retrieval_prefers_assignment_over_similar_application_passage(self):
        results = search_index(embed_texts(["What is the assignment deadline?"], self.model), self.index, self.chunks, top_k=3)
        self.assertIn("assignment submission", results[0]["text"].lower())
        self.assertEqual(results[0]["page"], 1)

    def test_duplicate_chunk_text_is_not_returned_twice(self):
        duplicate_chunks = [
            {"chunk_id": "chunk-1", "source": "a.pdf", "page": 1, "text": "Attendance is 75%."},
            {"chunk_id": "chunk-2", "source": "b.pdf", "page": 1, "text": "Attendance is 75%."},
        ]
        index, duplicate_chunks = build_faiss_index(embed_texts([item["text"] for item in duplicate_chunks], self.model), duplicate_chunks)
        results = search_index(embed_texts(["attendance"], self.model), index, duplicate_chunks, top_k=2)
        self.assertEqual(len(results), 1)

    def test_validator_resolves_each_claim_to_the_correct_source_page_and_passage(self):
        evidence = [
            {"evidence_id": "C1", "source": "rules.pdf", "page": 1, "text": "Attendance is 75%.", "similarity": 0.9},
            {"evidence_id": "C2", "source": "schedule.pdf", "page": 1, "text": "The assignment submission deadline is 10 May.", "similarity": 0.8},
            {"evidence_id": "C3", "source": "finance.pdf", "page": 1, "text": "The project budget is 5000 rupees.", "similarity": 0.8},
        ]
        answer, citations = validate_cited_answer(
            "- Attendance is 75%. [C1]\n- Deadline is 10 May. [C2]\n- Budget is 5000 rupees. [C3]", evidence
        )
        self.assertIn("[C2]", answer)
        self.assertEqual([(item["source"], item["page"]) for item in citations], [("rules.pdf", 1), ("schedule.pdf", 1), ("finance.pdf", 1)])
        self.assertEqual(citations[1]["text"], evidence[1]["text"])

    def test_two_document_answer_keeps_each_claim_citation_separate(self):
        evidence = [
            {"evidence_id": "C1", "source": "rules.pdf", "page": 1, "text": "Attendance is 75%.", "similarity": 0.9},
            {"evidence_id": "C2", "source": "schedule.pdf", "page": 1, "text": "The assignment deadline is 10 May.", "similarity": 0.8},
        ]
        answer, citations = validate_cited_answer(
            "- Attendance is 75%. [C1]\n- Assignment deadline is 10 May. [C2]", evidence
        )
        self.assertEqual(len(citations), 2)
        self.assertIn("[C1]", answer.splitlines()[0])
        self.assertIn("[C2]", answer.splitlines()[1])

    def test_contradictory_sources_remain_separately_cited(self):
        evidence = [
            {"evidence_id": "C1", "source": "rules.pdf", "page": 2, "text": "The review is Monday.", "similarity": 0.9},
            {"evidence_id": "C2", "source": "finance.pdf", "page": 2, "text": "The review is Friday.", "similarity": 0.8},
        ]
        answer, citations = validate_cited_answer(
            "- The documents conflict: the review is Monday [C1] and Friday [C2]", evidence
        )
        self.assertNotEqual(answer, NOT_FOUND_ANSWER)
        self.assertEqual([(item["source"], item["page"]) for item in citations], [("rules.pdf", 2), ("finance.pdf", 2)])

    def test_validator_rejects_fabricated_and_uncited_claims(self):
        evidence = [{"evidence_id": "C1", "source": "rules.pdf", "page": 1, "text": "Attendance is 75%.", "similarity": 0.9}]
        for response in ("The answer is 75%.", "The answer is 75%. [C99]", "The answer is 75%. [D1]"):
            answer, citations = validate_cited_answer(response, evidence)
            self.assertEqual(answer, NOT_FOUND_ANSWER)
            self.assertEqual(citations, [])

    def test_pipeline_abstains_when_question_is_absent(self):
        with patch("src.rag.query_ollama") as generate:
            result = answer_question("What is the weather on Mars?", self.index, self.chunks, self.model)
        self.assertEqual(result["answer"], NOT_FOUND_ANSWER)
        self.assertEqual(result["citations"], [])
        generate.assert_not_called()

    def test_summary_request_uses_page_diverse_evidence_without_semantic_threshold(self):
        self.assertTrue(is_summary_request("Give me a very short summary of this doc."))
        self.assertTrue(is_summary_request("What does this document say?"))
        self.assertFalse(is_summary_request("What is the attendance requirement?"))

        selected = select_summary_chunks(self.chunks, limit=6)
        self.assertEqual(len(selected), 6)
        self.assertEqual({chunk["source"] for chunk in selected}, {"rules.pdf", "schedule.pdf", "finance.pdf"})
        self.assertTrue(all(chunk["similarity"] is None for chunk in selected))

        with patch(
            "src.rag.query_ollama",
            return_value=("- Attendance is 75%. [C1]", {}),
        ) as generate:
            result = answer_question("Please summarize this document.", self.index, self.chunks, self.model)
        self.assertEqual(result["retrieval"]["mode"], "Document overview")
        self.assertEqual(result["citations"][0]["source"], "rules.pdf")
        generate.assert_called_once()

    def test_summary_validator_assigns_evidence_to_uncited_bullets(self):
        evidence = [
            {"evidence_id": "C1", "source": "rules.pdf", "page": 1, "text": "Students must maintain 75 percent attendance in lectures.", "similarity": None},
            {"evidence_id": "C2", "source": "schedule.pdf", "page": 1, "text": "Assignments and laboratory projects are due by prescribed deadlines.", "similarity": None},
            {"evidence_id": "C3", "source": "rules.pdf", "page": 5, "text": "Academic misconduct includes plagiarism and copying.", "similarity": None},
        ]
        answer, citations = validate_summary_answer(
            "- Students must maintain 75 percent attendance in lectures.\n"
            "- Assignments and projects must meet deadlines.\n"
            "- Plagiarism and copying are academic misconduct.",
            evidence,
        )
        self.assertIn("[C1]", answer)
        self.assertIn("[C2]", answer)
        self.assertIn("[C3]", answer)
        self.assertEqual([citation["id"] for citation in citations], ["C1", "C2", "C3"])

    def test_pipeline_rejects_invalid_model_citation(self):
        with patch("src.rag.query_ollama", return_value=("A made-up fact. [C99]", {})):
            result = answer_question("What is the attendance requirement?", self.index, self.chunks, self.model)
        self.assertEqual(result["answer"], NOT_FOUND_ANSWER)
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
