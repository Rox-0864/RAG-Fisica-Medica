"""Tests for document loading and chunking (src/document_loader.py)."""

from src import config
from src.document_loader import split_into_chunks


def test_split_returns_non_empty_chunks():
    text = "palabra " * 500  # ~4000 chars, must split into multiple chunks
    chunks = split_into_chunks(text)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_chunks_do_not_exceed_chunk_size():
    text = "Una frase sobre física médica. " * 200
    chunks = split_into_chunks(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= config.CHUNK_SIZE for chunk in chunks)


def test_short_text_returns_single_chunk():
    chunks = split_into_chunks("Texto corto que no se divide.")
    assert len(chunks) == 1
    assert chunks[0] == "Texto corto que no se divide."


def test_overlap_preserves_boundary_content():
    # A unique token near a chunk boundary must survive in the overlapping
    # region so retrieval can recover it.
    filler = "x " * 490  # ~980 chars, near CHUNK_SIZE
    text = filler + " TOKEN_UNICO_MEDIO " + filler
    chunks = split_into_chunks(text)
    assert any("TOKEN_UNICO_MEDIO" in chunk for chunk in chunks)
