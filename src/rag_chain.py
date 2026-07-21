"""
RAG chain orchestration.

Coordinates the full RAG pipeline:
1. Load vector store and LLM
2. Retrieve relevant context for the query
3. Format prompt with context + history + question
4. Generate answer via Ollama
"""

import logging
from typing import List, Tuple, Dict, Any

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from .vector_store import load_vector_store
from .embeddings import get_embeddings
from .llm import get_llm
from .prompts import SYSTEM_TEMPLATE

logger = logging.getLogger(__name__)

# Number of chunks to retrieve per query
RETRIEVAL_K = 10


def create_rag_chain():
    """
    Create and return a configured RAG chain.

    The chain handles retrieval + generation in one call.
    Uses ConversationalRetrievalChain which:
        - Reformulates questions considering chat history
        - Retrieves K most relevant chunks
        - Formats prompt with context, history, and question
        - Generates answer via LLM

    Returns:
        Configured ConversationalRetrievalChain ready for .invoke()
    """
    # Load components
    llm = get_llm()
    embedding_model = get_embeddings()
    vector_store = load_vector_store(embedding_model=embedding_model)

    if vector_store is None:
        raise RuntimeError(
            "Vector store not found. Run document indexing first:\n"
            "  python -m src.document_loader"
        )

    # Configure retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    # Build prompt template
    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "chat_history", "question"],
    )

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )

    logger.info(f"RAG chain ready (retriever k={RETRIEVAL_K})")
    return chain


def format_chat_history(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Convert Streamlit message format to LangChain chat history format.

    Streamlit uses: [{"role": "user", "content": "..."}, {"role": "bot", "content": "..."}]
    LangChain needs: [("user question", "bot answer"), ...]

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        List of (user_message, bot_message) tuples.
    """
    formatted = []
    # Skip the initial system message (index 0)
    chat_pairs = messages[1:]

    for i in range(0, len(chat_pairs) - 1, 2):
        if i + 1 < len(chat_pairs):
            user_msg = chat_pairs[i]
            bot_msg = chat_pairs[i + 1]
            if user_msg["role"] == "user" and bot_msg["role"] == "bot":
                formatted.append((user_msg["content"], bot_msg["content"]))

    return formatted


def query_rag(chain, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Execute a RAG query and return the answer with sources.

    Args:
        chain: Configured ConversationalRetrievalChain.
        question: User's question in Spanish.
        chat_history: Previous conversation pairs.

    Returns:
        Dict with keys: answer, source_documents
    """
    logger.info(f"Query: {question[:100]}...")

    try:
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        logger.info(f"Retrieved {len(result.get('source_documents', []))} source chunks")
        return result

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": "Lo siento, ocurrió un error al procesar tu pregunta. "
                      "Por favor, intentá de nuevo.",
            "source_documents": [],
        }
