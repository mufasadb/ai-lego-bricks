"""
Tools module for AI Lego Bricks - Universal tool calling service.
"""
from .tool_types import (
    ToolSchema, ToolCall, ToolResult, ToolChoice, Tool, ToolExecutor,
    ToolParameter, ToolChoiceType, ParameterType
)
from .tool_registry import ToolRegistry, get_global_registry, register_tool_globally, get_tool_globally
from .provider_adapters import (
    ProviderAdapter, OpenAIAdapter, AnthropicAdapter, GeminiAdapter, OllamaAdapter,
    AdapterFactory
)
from .tool_service import ToolService, get_global_tool_service
from .secure_tool_executor import (
    SecureToolExecutor, APIToolExecutor, DatabaseToolExecutor, WebhookToolExecutor,
    create_tool_with_credentials
)

__all__ = [
    # Core types
    "ToolSchema",
    "ToolCall", 
    "ToolResult",
    "ToolChoice",
    "Tool",
    "ToolExecutor",
    "ToolParameter",
    "ToolChoiceType",
    "ParameterType",
    
    # Registry
    "ToolRegistry",
    "get_global_registry",
    "register_tool_globally",
    "get_tool_globally",
    
    # Adapters
    "ProviderAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter", 
    "GeminiAdapter",
    "OllamaAdapter",
    "AdapterFactory",
    
    # Main service
    "ToolService",
    "get_global_tool_service",
    
    # Secure tools
    "SecureToolExecutor",
    "APIToolExecutor",
    "DatabaseToolExecutor", 
    "WebhookToolExecutor",
    "create_tool_with_credentials",
]