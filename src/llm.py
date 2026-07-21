"""
LLM interface using Ollama.

Connects to a local Ollama instance. Supports multiple models:
- llama3.2:3b (2GB) — fast, good Spanish, default
- llama3.1:8b (4.9GB) — better accuracy & Spanish, needs ~6-7GB RAM
"""

import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Available models (user can switch at runtime)
AVAILABLE_MODELS = {
    "llama3.2:3b": "Llama 3.2 3B — rápido, buen español (2GB RAM)",
    "llama3.1:8b": "Llama 3.1 8B — más preciso, mejor español (6-7GB RAM)",
}

# Cached instance per model to avoid reconnecting every query
_llm_instances: dict[str, OllamaLLM] = {}


def get_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    force_reload: bool = False,
) -> OllamaLLM:
    """
    Get or create an Ollama LLM instance.

    Caches one instance per model name. Call with force_reload=True
    to force a new connection (useful when switching models).

    Args:
        model: Ollama model name.
        base_url: Ollama server URL.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
        force_reload: If True, discards cached instance and creates a new one.

    Returns:
        Configured OllamaLLM instance.
    """
    global _llm_instances

    if force_reload:
        _llm_instances.pop(model, None)

    if model in _llm_instances:
        return _llm_instances[model]

    logger.info(f"Connecting to Ollama: {base_url} (model: {model})")
    llm = OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=4096,
        num_predict=1024,
    )
    _llm_instances[model] = llm
    logger.info(f"Ollama LLM ready: {model}")
    return llm


def list_available_models() -> dict[str, str]:
    """Return dict of model_name -> description for models known to work."""
    return AVAILABLE_MODELS.copy()
