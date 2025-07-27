"""
Chat service package for LLM integrations.

This package provides:
- Conversation services for multi-turn conversations
- Chat interfaces for different LLM providers
- Conversation state management and persistence
"""

from .conversation_service import (
    create_conversation,
    create_gemini_conversation,
    create_anthropic_conversation,
    create_ollama_conversation,
)

from .chat_service import (
    create_chat_service,
    get_available_chat_services,
)

__all__ = [
    "create_conversation",
    "create_gemini_conversation",
    "create_anthropic_conversation",
    "create_ollama_conversation",
    "create_chat_service",
    "get_available_chat_services",
]
