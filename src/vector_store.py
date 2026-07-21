"""
Vector store management using ChromaDB.

Handles creation, persistence, and loading of the vector store.
The store is created once during initial indexing and loaded
from disk on subsequent runs.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Default paths
VECTOR_STORE_DIR = Path(__file__).parent.parent / "vector_store"
STORE_NAME = "medphys_rag"


def get_store_path(store_name: str = STORE_NAME) -> Path:
    """Get the full path for a vector store."""
    return VECTOR_STORE_DIR / store_name


def initialize_vector_store(
    documents: List[Document],
    store_name: str = STORE_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    """
    Create a new vector store from documents.

    This is expensive — it computes embeddings for every chunk.
    Should only run once per document set.

    Args:
        documents: List of LangChain Documents to index.
        store_name: Name for the persisted store directory.
        embedding_model: Optional pre-loaded embedding model.

    Returns:
        Initialized Chroma vector store.
    """
    if embedding_model is None:
        embedding_model = get_embeddings()

    store_path = get_store_path(store_name)
    store_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating vector store at: {store_path}")
    logger.info(f"Indexing {len(documents)} documents...")

    # Extract text from documents for Chroma
    # Note: Chroma.from_documents preserves metadata automatically
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(store_path),
    )

    logger.info(f"Vector store created with {vector_store._collection.count()} embeddings")
    return vector_store


def load_vector_store(
    store_name: str = STORE_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> Optional[Chroma]:
    """
    Load an existing vector store from disk.

    Args:
        store_name: Name of the persisted store.
        embedding_model: Optional pre-loaded embedding model.

    Returns:
        Loaded Chroma vector store, or None if it doesn't exist.
    """
    store_path = get_store_path(store_name)

    if not store_path.exists() or not any(store_path.iterdir()):
        logger.info(f"Vector store not found at: {store_path}")
        return None

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info(f"Loading vector store from: {store_path}")
    vector_store = Chroma(
        persist_directory=str(store_path),
        embedding_function=embedding_model,
    )

    try:
        count = vector_store._collection.count()
        logger.info(f"Vector store loaded: {count} embeddings")
    except Exception:
        logger.warning("Could not get collection count")

    return vector_store


def store_exists(store_name: str = STORE_NAME) -> bool:
    """Check if a vector store already exists on disk."""
    store_path = get_store_path(store_name)
    return store_path.exists() and any(store_path.iterdir())
