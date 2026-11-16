"""Tests for the centralized configuration (src/config.py)."""

import os
from pathlib import Path
from unittest.mock import patch

from src import config


def test_int_parses_valid_value():
    with patch.dict(os.environ, {"CHUNK_SIZE": "1234"}):
        assert config._int("CHUNK_SIZE", 1000) == 1234


def test_int_falls_back_to_default_when_absent():
    with patch.dict(os.environ, {"CHUNK_SIZE": ""}):
        assert config._int("CHUNK_SIZE", 1000) == 1000


def test_int_falls_back_on_invalid_value():
    with patch.dict(os.environ, {"CHUNK_SIZE": "not-a-number"}):
        assert config._int("CHUNK_SIZE", 1000) == 1000


def test_float_parses_valid_value():
    with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.7"}):
        assert config._float("LLM_TEMPERATURE", 0.2) == 0.7


def test_float_falls_back_on_invalid_value():
    with patch.dict(os.environ, {"LLM_TEMPERATURE": "abc"}):
        assert config._float("LLM_TEMPERATURE", 0.2) == 0.2


def test_path_resolves_relative_to_project_root():
    with patch.dict(os.environ, {"VECTOR_STORE_DIR": "my_store"}):
        assert config._path("VECTOR_STORE_DIR", "vector_store") == (
            config.PROJECT_ROOT / "my_store"
        )


def test_path_keeps_absolute_value():
    with patch.dict(os.environ, {"VECTOR_STORE_DIR": "/tmp/abs_store"}):
        assert config._path("VECTOR_STORE_DIR", "vector_store") == Path("/tmp/abs_store")


def test_path_falls_back_to_default_when_absent():
    with patch.dict(os.environ, {"VECTOR_STORE_DIR": ""}):
        assert config._path("VECTOR_STORE_DIR", "vector_store") == (
            config.PROJECT_ROOT / "vector_store"
        )


def test_resolve_device_returns_explicit_value():
    with patch.object(config, "EMBEDDING_DEVICE", "cpu"):
        assert config.resolve_embedding_device() == "cpu"
    with patch.object(config, "EMBEDDING_DEVICE", "cuda"):
        assert config.resolve_embedding_device() == "cuda"


def test_resolve_device_auto_returns_valid_device():
    with patch.object(config, "EMBEDDING_DEVICE", "auto"):
        assert config.resolve_embedding_device() in {"cpu", "cuda"}


def test_defaults_reflect_current_deployment():
    assert config.OLLAMA_MODEL == "llama3.1:8b"
    assert config.EMBEDDING_MODEL == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert config.RETRIEVAL_K == 8
    assert config.CHUNK_SIZE == 1000
    assert config.CHUNK_OVERLAP == 200
