"""
AI Lego Bricks - A modular LLM agent system.

This package provides building blocks for intelligent AI workflows with advanced
memory capabilities, JSON-driven agent orchestration, and multi-modal processing.
"""

__version__ = "0.1.0"
__author__ = "Daniel Beach"
__email__ = "callmebeachy@gmail.com"

# Core imports for easy access
from .core import (
    create_memory_service,
    create_generation_service,
    create_conversation_service,
    create_tts_service,
    create_prompt_service,
)

# Agent orchestration
from agent_orchestration import AgentOrchestrator

__all__ = [
    "__version__",
    "create_memory_service",
    "create_generation_service", 
    "create_conversation_service",
    "create_tts_service",
    "create_prompt_service",
    "AgentOrchestrator",
]