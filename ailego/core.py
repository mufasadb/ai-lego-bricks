"""
Core factory functions for AI Lego Bricks components.

This module provides simplified factory functions for creating various services
with automatic provider detection and sensible defaults.
"""

import os
from typing import Any, Dict


def create_memory_service(provider: str = "auto", **kwargs) -> Any:
    """
    Create a memory service with automatic provider detection.

    Args:
        provider: Memory provider ("auto", "supabase", "neo4j")
        **kwargs: Additional configuration options

    Returns:
        Memory service instance
    """
    from memory import create_memory_service as _create_memory_service

    return _create_memory_service(provider, **kwargs)


def create_generation_service(provider: str = "auto", **kwargs) -> Any:
    """
    Create a generation service for one-shot LLM calls.

    Args:
        provider: LLM provider ("auto", "gemini", "anthropic", "ollama")
        **kwargs: Additional configuration options

    Returns:
        Generation service instance
    """
    from llm.generation_service import (
        create_generation_service as _create_generation_service,
    )

    return _create_generation_service(provider, **kwargs)


def create_conversation_service(provider: str = "auto", **kwargs) -> Any:
    """
    Create a conversation service for multi-turn conversations.

    Args:
        provider: LLM provider ("auto", "gemini", "anthropic", "ollama")
        **kwargs: Additional configuration options

    Returns:
        Conversation service instance
    """
    from chat.conversation_service import (
        create_conversation_service as _create_conversation_service,
    )

    return _create_conversation_service(provider, **kwargs)


def create_tts_service(provider: str = "auto", **kwargs) -> Any:
    """
    Create a text-to-speech service.

    Args:
        provider: TTS provider ("auto", "openai", "google", "coqui")
        **kwargs: Additional configuration options

    Returns:
        TTS service instance
    """
    from tts import create_tts_service as _create_tts_service

    return _create_tts_service(provider, **kwargs)


def create_prompt_service(provider: str = "auto", **kwargs) -> Any:
    """
    Create a prompt management service.

    Args:
        provider: Storage provider ("auto", "supabase", "neo4j")
        **kwargs: Additional configuration options

    Returns:
        Prompt service instance
    """
    from prompt import create_prompt_service as _create_prompt_service

    return _create_prompt_service(provider, **kwargs)


def get_version() -> str:
    """Get the current version of AI Lego Bricks."""
    from ailego import __version__

    return __version__


def get_available_providers() -> Dict[str, list]:
    """
    Get a list of available providers for each service type.

    Returns:
        Dictionary mapping service types to available providers
    """
    providers = {"memory": [], "llm": [], "tts": [], "prompt": []}

    # Check for memory providers
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        providers["memory"].append("supabase")
    if os.getenv("NEO4J_URI") and os.getenv("NEO4J_USERNAME"):
        providers["memory"].append("neo4j")

    # Check for LLM providers
    if os.getenv("GOOGLE_API_KEY"):
        providers["llm"].append("gemini")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers["llm"].append("anthropic")
    if os.getenv("OLLAMA_URL") or os.path.exists("/usr/local/bin/ollama"):
        providers["llm"].append("ollama")

    # Check for TTS providers
    if os.getenv("OPENAI_API_KEY"):
        providers["tts"].append("openai")
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        providers["tts"].append("google")
    providers["tts"].append("coqui")  # Usually available

    # Prompt providers typically use same as memory
    providers["prompt"] = providers["memory"]

    return providers
