# AI-RAG-Radiodiagnostico

Agente RAG (Retrieval-Augmented Generation) para responder preguntas sobre el libro
"Essential Physics of Medical Imaging" usando IA 100% local y gratuita.

## Stack

- **LLM**: Ollama + llama3.2:3b
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **PDF Processing**: pdfplumber
- **UI**: Streamlit
- **Orchestration**: LangChain

## Requisitos

- Python 3.10+
- Ollama instalado y corriendo
- 8GB+ RAM (recomendado 16GB)

## Instalación

```bash
# 1. Clonar el repo
git clone git@github.com:Rox-0864/AI-RAG-Radiodiagnostico.git
cd AI-RAG-Radiodiagnostico

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env

# 5. Bajar modelo de Ollama
ollama pull llama3.2:3b

# 6. Ejecutar
streamlit run src/app.py
```

## Uso

1. Colocar el PDF en la carpeta `data/`
2. Al iniciar, el sistema indexa el documento (solo la primera vez)
3. Hacer preguntas en español sobre el contenido del libro
4. El agente busca en el documento (en inglés) y responde en español

## Arquitectura

```
PDF -> pdfplumber -> Chunks -> Embeddings (HuggingFace) -> ChromaDB
                                                              |
Pregunta (español) -> Retrieval -> Contexto (inglés) -> Ollama -> Respuesta (español)
```
