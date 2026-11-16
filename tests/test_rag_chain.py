"""Tests for RAG chain orchestration helpers (src/rag_chain.py)."""

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_chain import format_chat_history


def test_format_converts_roles_and_skips_greeting():
    messages = [
        {"role": "bot", "content": "greeting"},  # initial system message, skipped
        {"role": "user", "content": "pregunta 1"},
        {"role": "bot", "content": "respuesta 1"},
    ]
    history = format_chat_history(messages)
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "pregunta 1"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "respuesta 1"


def test_format_drops_trailing_unpaired_question():
    # The current question (last user message) must be excluded.
    messages = [
        {"role": "bot", "content": "greeting"},
        {"role": "user", "content": "pregunta 1"},
        {"role": "bot", "content": "respuesta 1"},
        {"role": "user", "content": "pregunta 2"},  # unpaired current question
    ]
    history = format_chat_history(messages)
    assert len(history) == 2
    assert history[-1].content == "respuesta 1"


def test_format_returns_empty_with_only_greeting():
    messages = [{"role": "bot", "content": "greeting"}]
    assert format_chat_history(messages) == []
