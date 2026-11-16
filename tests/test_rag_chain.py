"""Tests for RAG chain orchestration helpers (src/rag_chain.py)."""

from src.rag_chain import format_chat_history


def test_format_pairs_user_and_bot_messages():
    messages = [
        {"role": "bot", "content": "greeting"},  # initial system message, skipped
        {"role": "user", "content": "pregunta 1"},
        {"role": "bot", "content": "respuesta 1"},
        {"role": "user", "content": "pregunta 2"},
        {"role": "bot", "content": "respuesta 2"},
    ]
    assert format_chat_history(messages) == [
        ("pregunta 1", "respuesta 1"),
        ("pregunta 2", "respuesta 2"),
    ]


def test_format_skips_unpaired_user_message():
    # The current question has no answer yet, so it must be excluded.
    messages = [
        {"role": "bot", "content": "greeting"},
        {"role": "user", "content": "pregunta 1"},
        {"role": "bot", "content": "respuesta 1"},
        {"role": "user", "content": "pregunta 2"},  # unpaired
    ]
    assert format_chat_history(messages) == [("pregunta 1", "respuesta 1")]


def test_format_returns_empty_with_only_greeting():
    messages = [{"role": "bot", "content": "greeting"}]
    assert format_chat_history(messages) == []
