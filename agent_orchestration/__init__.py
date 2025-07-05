"""
Agent Orchestration System

A JSON-driven system for orchestrating AI agents using the existing building blocks:
- LLM clients (text, vision, embedding)
- Memory services (Supabase, Neo4j)
- Document processing (PDF extraction, chunking)
- Chat interfaces

This allows building complex agent workflows by combining these components
in a declarative, configuration-driven manner.
"""

from .orchestrator import AgentOrchestrator, WorkflowExecutor
from .models import WorkflowConfig, StepConfig, StepType
from .step_handlers import StepHandlerRegistry

__all__ = [
    "AgentOrchestrator",
    "WorkflowExecutor", 
    "WorkflowConfig",
    "StepConfig",
    "StepType",
    "StepHandlerRegistry"
]