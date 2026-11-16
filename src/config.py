"""
Centralized configuration loaded from environment variables.

This module is the single source of truth for all runtime settings.
Values are read once from a `.env` file (via python-dotenv) and/or the
process environment, with sensible defaults matching the current
deployment (llama3.1:8b + multilingual embeddings).

Modules import from here instead of hardcoding their own constants, so
changing a model, chunk size, or vector-store path only requires editing
`.env` (or the defaults below) — never the source code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of the `src/` package).
PROJECT_ROOT = Path(__file__).parent.parent

# Load `.env` once. Existing environment variables take precedence over
# the file (dotenv's default behaviour).
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to `default` on absence/error."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    """Parse a float env var, falling back to `default` on absence/error."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    """Resolve a path env var relative to PROJECT_ROOT when not absolute."""
    raw = os.getenv(name)
    value = raw if raw else default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Ollama (LLM)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2)
NUM_CTX = _int("NUM_CTX", 8192)
NUM_PREDICT = _int("NUM_PREDICT", 2048)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# "auto" (detect CUDA), "cuda", or "cpu".
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_K = _int("RETRIEVAL_K", 8)

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

VECTOR_STORE_DIR = _path("VECTOR_STORE_DIR", "vector_store")
VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "medphys_rag")


def resolve_embedding_device() -> str:
    """
    Resolve the embedding device.

    When EMBEDDING_DEVICE is "auto", detect whether CUDA is available
    (torch is a transitive dependency of sentence-transformers) and fall
    back to CPU. An explicit "cuda"/"cpu" value is returned as-is.
    """
    device = EMBEDDING_DEVICE.lower().strip()
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"
