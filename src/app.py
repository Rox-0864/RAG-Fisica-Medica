"""
Streamlit application for the Medical Physics RAG agent.

Provides a chat interface where users can ask questions
in Spanish about the "Essential Physics of Medical Imaging" book.

On first run, indexes the PDF and creates the vector store.
On subsequent runs, loads the existing vector store instantly.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Silence noisy transformers import warnings from Streamlit's file watcher
# (torchvision is an optional dep we don't need — the watcher just tries to inspect everything)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add project root to Python path so 'src' package is importable
# when running via 'streamlit run src/app.py'
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    page_title="La Fisica de las Imagenes Medicas",
    page_icon="🩻",
    layout="wide",
)

# PDF path — relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
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
                "Hola. Soy un agente especializado en física médica e "
                "imagenología diagnóstica. Puedo responder tus preguntas "
                "basándome en el libro **Essential Physics of Medical Imaging**.\n\n"
                "El libro está en inglés, pero yo respondo en español. "
                "Podés preguntarme sobre rayos X, CT, MRI, ultrasonido, "
                "medicina nuclear, protección radiológica, y más.\n\n"
                "¿Qué te gustaría saber?"
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
        st.title("Configuración")

        st.markdown("---")
        st.subheader("Modelo")
        st.text("LLM: llama3.2:3b (Ollama)")
        st.text("Embedding: all-MiniLM-L6-v2 (CPU)")
        st.text("Vector Store: ChromaDB")

        st.markdown("---")
        st.subheader("Documento")
        if PDF_PATH.exists():
            size_mb = PDF_PATH.stat().st_size / 1e6
            st.text(f"PDF: {PDF_PATH.name}")
            st.text(f"Tamaño: {size_mb:.1f} MB")
        else:
            st.warning("PDF no encontrado en data/")

        st.markdown("---")
        st.subheader("Acerca de")
        st.markdown(
            "Agente RAG 100% local y gratuito. "
            "No utiliza APIs externas de pago. "
            "Todo el procesamiento ocurre en tu máquina."
        )

        st.markdown("---")
        st.caption("Repositorio: [GitHub](https://github.com/Rox-0864/RAG-Fisica-Medica)")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application."""
    init_session_state()
    render_sidebar()

    # Header with image and title
    col1, col2 = st.columns([1, 3])
    with col1:
        banner_path = ASSETS_DIR / "radiodiagnostico.jpeg"
        if banner_path.exists():
            st.image(str(banner_path), use_container_width=True)
    with col2:
        st.title("La Física de las Imágenes Médicas")
        st.markdown(
            "<h4 style='margin-top: 0; color: #666;'>"
            "Agente de IA para consultas sobre física médica "
            "e imagenología diagnóstica"
            "</h4>",
            unsafe_allow_html=True,
        )

    # Initialize or load the vector store
    if not st.session_state.chain_ready:
        with st.spinner("Verificando vector store..."):
            ready = initialize_or_load_store()
            if not ready:
                st.stop()

            # Create the RAG chain
            try:
                st.session_state.chain = create_rag_chain(model="llama3.2:3b")
                st.session_state.chain_ready = True
                logger.info("RAG chain initialized")
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
