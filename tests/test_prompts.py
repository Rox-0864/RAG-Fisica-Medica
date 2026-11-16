"""Tests for the RAG prompt templates (src/prompts.py)."""

from src.prompts import CONDENSE_TEMPLATE, SYSTEM_TEMPLATE


def test_qa_prompt_has_context_placeholder():
    # The document-combining chain fills {context} with retrieved chunks.
    assert "{context}" in SYSTEM_TEMPLATE


def test_qa_prompt_restricts_answers_to_context():
    assert "Answer ONLY using the provided context" in SYSTEM_TEMPLATE
    assert "Do NOT use outside knowledge" in SYSTEM_TEMPLATE


def test_qa_prompt_instructs_spanish_answer():
    assert "answer in Spanish" in SYSTEM_TEMPLATE


def test_qa_prompt_has_insufficient_context_fallback():
    assert "El documento no contiene información suficiente" in SYSTEM_TEMPLATE


def test_qa_prompt_instructs_page_citation():
    assert "Cite page numbers" in SYSTEM_TEMPLATE


def test_condense_prompt_instructs_reformulation():
    assert "rephrase the question" in CONDENSE_TEMPLATE
    assert "Do NOT answer the question" in CONDENSE_TEMPLATE


def test_condense_prompt_has_no_context_placeholder():
    # Condense runs BEFORE retrieval, so it must not reference {context}.
    assert "{context}" not in CONDENSE_TEMPLATE
