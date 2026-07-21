"""
Streamlit application for the Medical Physics RAG agent.

Provides a chat interface where users can ask questions
in Spanish about the "Essential Physics of Medical Imaging" book.

On first run, indexes the PDF and creates the vector store.
On subsequent runs, loads the existing vector store instantly.
"""

import logging
from pathlib import Path

import streamlit as st

from src.document_loader import load_pdf
from src.vector_store import (
    initialize_vector_store,
    load_vector_store,
    store_exists,
)
from src.embeddings import get_embeddings
from src.llm import get_llm, list_available_models
from src.rag_chain import create_rag_chain, format_chat_history, query_rag

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Radiodiagnostico",
    page_icon="🩻",
    layout="wide",
)

# PDF path — relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Essential Physics Medical Imaging.pdf"

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [{
            "role": "bot",
            "content": (
                "Hola. Soy un agente especializado en fisica medica e "
                "imagenologia diagnostica. Puedo responder tus preguntas "
                "basandome en el libro **Essential Physics of Medical Imaging**.\n\n"
                "El libro esta en ingles, pero yo respondo en espanol. "
                "Podes preguntarme sobre rayos X, CT, MRI, ultrasonido, "
                "medicina nuclear, proteccion radiologica, y mas.\n\n"
                "Que te gustaria saber?"
            ),
        }],
        "chain_ready": False,
        "chain": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Vector Store Initialization
# ---------------------------------------------------------------------------

def initialize_or_load_store() -> bool:
    """
    Initialize the vector store if it doesn't exist, or load it.

    Returns True if the store is ready, False if initialization failed.
    """
    if store_exists():
        logger.info("Vector store found — loading existing")
        return True

    # Need to index the PDF
    if not PDF_PATH.exists():
        st.error(f"PDF no encontrado en: {PDF_PATH}")
        st.info(
            "Coloca el archivo 'Essential Physics Medical Imaging.pdf' "
            "en la carpeta 'data/' y reinicia la aplicacion."
        )
        return False

    # Show indexing progress
    status_container = st.empty()

    try:
        with status_container.status("Indexando el documento...", expanded=True) as status:
            st.write("**Paso 1/3:** Extrayendo texto del PDF...")
            documents = load_pdf(str(PDF_PATH))

            # Show stats in a formatted way
            pages = sorted(set(d.metadata["page"] for d in documents))
            total_chars = sum(len(d.page_content) for d in documents)
            st.write(
                f"- {len(documents)} chunks creados\n"
                f"- {len(pages)} paginas procesadas\n"
                f"- {total_chars:,} caracteres extraidos"
            )

            st.write("**Paso 2/3:** Generando embeddings (esto toma unos minutos)...")
            embedding_model = get_embeddings()

            st.write("**Paso 3/3:** Guardando vector store en ChromaDB...")
            initialize_vector_store(
                documents=documents,
                embedding_model=embedding_model,
            )

            status.update(label="Indexacion completada", state="complete")

        st.success("Documento indexado correctamente. El agente esta listo.")
        return True

    except Exception as e:
        status_container.error(f"Error durante la indexacion: {e}")
        logger.exception("Indexing failed")
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with configuration info."""
    with st.sidebar:
        st.title("Configuracion")

        # Model selector
        st.subheader("Modelo LLM")
        models = list_available_models()
        model_names = list(models.keys())
        default_idx = model_names.index("llama3.2:3b") if "llama3.2:3b" in model_names else 0

        selected_model = st.selectbox(
            "Selecciona el modelo",
            options=model_names,
            index=default_idx,
            format_func=lambda m: f"{m} — {models[m].split(' — ')[0]}",
            key="selected_model",
        )
        st.caption(models[selected_model].split(" — ")[1])

        # Reload chain button when model changes
        if st.button("Aplicar cambio de modelo"):
            st.session_state.chain_ready = False
            st.session_state.chain = None
            st.rerun()

        st.markdown("---")
        st.subheader("Embedding")
        st.text("all-MiniLM-L6-v2 (CPU)")
        st.text("Vector Store: ChromaDB")

        st.markdown("---")
        st.subheader("Documento")
        if PDF_PATH.exists():
            size_mb = PDF_PATH.stat().st_size / 1e6
            st.text(f"PDF: {PDF_PATH.name}")
            st.text(f"Tamano: {size_mb:.1f} MB")
        else:
            st.warning("PDF no encontrado en data/")

        st.markdown("---")
        st.subheader("Acerca de")
        st.markdown(
            "Agente RAG 100% local y gratuito. "
            "No utiliza APIs externas de pago. "
            "Todo el procesamiento ocurre en tu maquina."
        )

        st.markdown("---")
        st.caption("Repositorio: [GitHub](https://github.com/Rox-0864/AI-RAG-Radiodiagnostico)")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application."""
    init_session_state()
    render_sidebar()

    st.title("RAG Radiodiagnostico")
    st.caption("Agente de IA para consultas sobre fisica medica e imagenologia")

    # Initialize or load the vector store
    if not st.session_state.chain_ready:
        with st.spinner("Verificando vector store..."):
            ready = initialize_or_load_store()
            if not ready:
                st.stop()

            # Create the RAG chain
            try:
                model = st.session_state.get("selected_model", "llama3.2:3b")
                st.session_state.chain = create_rag_chain(model=model)
                st.session_state.chain_ready = True
                logger.info(f"RAG chain initialized with model: {model}")
            except Exception as e:
                st.error(f"Error al inicializar el agente: {e}")
                logger.exception("Chain initialization failed")
                st.stop()

    # Chat interface
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Escribi tu pregunta sobre fisica medica..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("bot"):
            with st.spinner("Buscando en el documento..."):
                chat_history = format_chat_history(st.session_state.messages)
                result = query_rag(
                    st.session_state.chain,
                    prompt,
                    chat_history,
                )

                answer = result.get("answer", "Error al generar respuesta.")
                st.markdown(answer)

                # Show sources in an expander
                sources = result.get("source_documents", [])
                if sources:
                    with st.expander("Fuentes consultadas"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "?")
                            st.caption(f"Pagina {page} — fragmento {i}")
                            st.text(doc.page_content[:300] + "...")
                            st.markdown("---")

        # Save response to history
        st.session_state.messages.append({"role": "bot", "content": answer})


if __name__ == "__main__":
    main()
