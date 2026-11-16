"""
RAG chain orchestration.

Coordinates the full RAG pipeline using LangChain's composable LCEL
pattern (create_history_aware_retriever + create_stuff_documents_chain +
create_retrieval_chain), replacing the deprecated ConversationalRetrievalChain.

Steps:
1. Load the vector store and the Ollama chat model.
2. Reformulate the question into a standalone query using chat history.
3. Retrieve the K most relevant chunks.
4. Generate the answer from the retrieved context.
"""

import logging
from typing import Any, Dict, List

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import OLLAMA_MODEL, RETRIEVAL_K
from .embeddings import get_embeddings
from .llm import get_llm
from .prompts import CONDENSE_TEMPLATE, SYSTEM_TEMPLATE
from .vector_store import load_vector_store

logger = logging.getLogger(__name__)


def create_rag_chain(model: str = OLLAMA_MODEL):
    """
    Create and return a configured conversational RAG chain.

    The chain is built from three explicit, composable pieces:
    1. A history-aware retriever that reformulates the question using
       chat history before retrieving (so follow-ups are self-contained).
    2. A document-combining chain that answers from the retrieved context.
    3. A retrieval chain that ties the two together.

    Args:
        model: Ollama model name (e.g. "llama3.2:3b", "llama3.1:8b").

    Returns:
        A Runnable that accepts {"input", "chat_history"} and returns
        {"answer", "context"}.
    """
    llm = get_llm(model=model)
    embedding_model = get_embeddings()
    vector_store = load_vector_store(embedding_model=embedding_model)

    if vector_store is None:
        raise RuntimeError(
            "Vector store not found. Run document indexing first:\n"
            "  python -m src.document_loader"
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # 1. Reformulate the question using chat history, then retrieve.
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_prompt
    )

    # 2. Answer from the retrieved context.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Full conversational RAG chain.
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logger.info(f"RAG chain ready (retriever k={RETRIEVAL_K})")
    return chain


def format_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """
    Convert Streamlit message format to a list of LangChain messages.

    Streamlit uses: [{"role": "user"|"bot", "content": "..."}]
    LangChain needs a flat list of HumanMessage / AIMessage objects.

    The initial greeting (index 0) and the trailing current question
    (an unpaired user message) are excluded.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        List of HumanMessage / AIMessage objects representing prior turns.
    """
    history = []
    for msg in messages[1:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "bot":
            history.append(AIMessage(content=msg["content"]))

    # Drop the current question, which is being asked right now (unpaired).
    if history and isinstance(history[-1], HumanMessage):
        history.pop()

    return history


def query_rag(chain, question: str, chat_history: List[Any]) -> Dict[str, Any]:
    """
    Execute a RAG query and return the answer with sources.

    Args:
        chain: Configured RAG chain.
        question: User's question in Spanish.
        chat_history: Previous conversation as a list of LangChain messages.

    Returns:
        Dict with keys: answer, source_documents
    """
    logger.info(f"Query: {question[:100]}...")

    try:
        result = chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })
        sources = result.get("context", [])
        logger.info(f"Retrieved {len(sources)} source chunks")
        return {
            "answer": result.get("answer", ""),
            "source_documents": sources,
        }

    except Exception as e:
        logger.exception("RAG query failed")
        return {
            "answer": "Lo siento, ocurrió un error al procesar tu pregunta. "
                      "Por favor, intentá de nuevo.",
            "source_documents": [],
        }
