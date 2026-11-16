"""Tests for the RAG prompt template (src/prompts.py)."""

from src.prompts import SYSTEM_TEMPLATE


def test_prompt_has_required_placeholders():
    for placeholder in ("{context}", "{chat_history}", "{question}"):
        assert placeholder in SYSTEM_TEMPLATE


def test_prompt_restricts_answers_to_context():
    assert "Answer ONLY using the provided context" in SYSTEM_TEMPLATE
    assert "Do NOT use outside knowledge" in SYSTEM_TEMPLATE


def test_prompt_instructs_spanish_answer():
    assert "answer in Spanish" in SYSTEM_TEMPLATE


def test_prompt_has_insufficient_context_fallback():
    assert "El documento no contiene información suficiente" in SYSTEM_TEMPLATE


def test_prompt_instructs_page_citation():
    assert "Cite page numbers" in SYSTEM_TEMPLATE
