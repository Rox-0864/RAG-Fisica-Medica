"""
Document loader for medical physics PDF.

Extracts text and tables from "Essential Physics of Medical Imaging"
and splits content into overlapping chunks for vector embedding.
"""

import logging
from pathlib import Path
from typing import List

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Documents ready for embedding.

    Each Document contains:
        - page_content: text chunk (~1000 chars with 200-char overlap)
        - metadata: page number, source filename

    The overlap ensures that information split across chunk boundaries
    is still recoverable during retrieval.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Loading PDF: {file_path.name} ({file_path.stat().st_size / 1e6:.1f} MB)")

    pages_data = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Total pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Extract tables and convert to readable text format
            tables = page.extract_tables()
            table_text = ""
            if tables:
                table_parts = []
                for i, table in enumerate(tables):
                    if table:
                        rows = []
                        for row in table:
                            clean_row = [
                                str(cell).strip() if cell else ""
                                for cell in row
                            ]
                            rows.append(" | ".join(clean_row))
                        table_parts.append(f"[Table {i + 1}]\n" + "\n".join(rows))
                table_text = "\n\n".join(table_parts)

            # Combine text and tables
            full_text = text.strip()
            if table_text:
                full_text += f"\n\n{table_text}"

            pages_data.append({
                "text": full_text,
                "page": page_num,
                "source": file_path.name,
            })

    logger.info(f"Extracted text from {len(pages_data)} pages")

    # Split into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    documents = []
    for page_data in pages_data:
        chunks = text_splitter.split_text(page_data["text"])

        for chunk in chunks:
            if not chunk.strip():
                continue

            doc = Document(
                page_content=chunk,
                metadata={
                    "page": page_data["page"],
                    "source": page_data["source"],
                }
            )
            documents.append(doc)

    logger.info(f"Created {len(documents)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return documents


def print_stats(documents: List[Document]) -> None:
    """Print summary statistics about the loaded documents."""
    if not documents:
        print("No documents loaded.")
        return

    pages = sorted(set(doc.metadata["page"] for doc in documents))
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chars = total_chars / len(documents) if documents else 0

    print(f"\n=== Document Loading Stats ===")
    print(f"Total chunks:   {len(documents)}")
    print(f"Pages spanned:  {pages[0]} - {pages[-1]} ({len(pages)} pages with content)")
    print(f"Total chars:    {total_chars:,}")
    print(f"Avg chunk size: {avg_chars:.0f} chars")
    print(f"Chunk settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    # Show a sample chunk
    print(f"\n--- Sample chunk (page {documents[0].metadata['page']}) ---")
    print(documents[0].page_content[:300] + "...")
