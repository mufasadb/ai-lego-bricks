"""
Prompt management service for AI Lego Bricks

Provides externalized prompt management with versioning, templating, 
evaluation, and training data collection capabilities.
"""

from .prompt_service import PromptService, create_prompt_service
from .prompt_models import (
    Prompt, PromptVersion, PromptTemplate, PromptExecution,
    PromptMetadata, PromptRole, PromptStatus
)
from .prompt_registry import PromptRegistry
from .evaluation_service import EvaluationService

__all__ = [
    "PromptService",
    "create_prompt_service", 
    "Prompt",
    "PromptVersion",
    "PromptTemplate",
    "PromptExecution",
    "PromptMetadata",
    "PromptRole",
    "PromptStatus",
    "PromptRegistry",
    "EvaluationService"
]