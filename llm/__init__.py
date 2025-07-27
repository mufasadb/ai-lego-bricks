"""
LLM abstraction layer for unified model access.

This package provides:
- Generation services for one-shot LLM calls
- Conversation services for multi-turn conversations
- Streaming support for real-time responses
- Multiple provider support (Gemini, Anthropic, Ollama)
"""

from .generation_service import (
    quick_generate_gemini,
    quick_generate_anthropic,
    quick_generate_ollama,
    quick_generate_ollama_stream,
    create_generation_service,
)

# Note: direct text client classes available in .text_clients module

from .llm_types import (
    LLMProvider,
    GenerationConfig,
)

__all__ = [
    "quick_generate_gemini",
    "quick_generate_anthropic",
    "quick_generate_ollama",
    "quick_generate_ollama_stream",
    "create_generation_service",
    "LLMProvider",
    "GenerationConfig",
]
