"""
Embedding model wrapper using HuggingFace sentence-transformers.

Uses paraphrase-multilingual-MiniLM-L12-v2 (470MB, 384-dimension vectors),
which aligns Spanish and English in the same semantic space. The device is
auto-detected (CUDA when available, CPU otherwise) via src/config.py.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL, resolve_embedding_device

logger = logging.getLogger(__name__)

# Default embedding model (single source of truth in src/config.py)
DEFAULT_MODEL = EMBEDDING_MODEL

# Singleton instance to avoid reloading the model
_embedding_model = None


def get_embeddings(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Get or create the HuggingFace embedding model.

    The model is loaded once and reused. On first call it downloads
    ~470MB from HuggingFace Hub (cached locally for subsequent runs).
    The device is auto-detected: CUDA when available, CPU otherwise.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    device = resolve_embedding_device()
    logger.info(f"Loading embedding model: {model_name} (device={device})")
    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully")
    return _embedding_model
