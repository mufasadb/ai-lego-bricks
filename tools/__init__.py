"""
Tools module for AI Lego Bricks - Universal tool calling service.
"""

from .tool_types import (
    ToolSchema,
    ToolCall,
    ToolResult,
    ToolChoice,
    Tool,
    ToolExecutor,
    ToolParameter,
    ToolChoiceType,
    ParameterType,
)
from .tool_registry import (
    ToolRegistry,
    get_global_registry,
    register_tool_globally,
    get_tool_globally,
)
from .provider_adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    OllamaAdapter,
    AdapterFactory,
)
from .tool_service import ToolService, get_global_tool_service
from .secure_tool_executor import (
    SecureToolExecutor,
    APIToolExecutor,
    DatabaseToolExecutor,
    WebhookToolExecutor,
    create_tool_with_credentials,
)
from .mcp_types import (
    MCPServerConfig,
    MCPTransport,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPToolInfo,
    MCPMethods,
    MCPErrorCodes,
)
from .mcp_server_manager import (
    MCPServerProcess,
    MCPServerManager,
    get_global_mcp_manager,
)
from .mcp_tool_executor import (
    MCPToolExecutor,
    MCPToolDiscovery,
    register_mcp_tools_globally,
)
from .mcp_config import (
    MCPConfigManager,
    initialize_mcp_servers_from_config,
    get_global_mcp_config_manager,
    create_example_config,
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
    # MCP integration
    "MCPServerConfig",
    "MCPTransport",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPToolInfo",
    "MCPMethods",
    "MCPErrorCodes",
    "MCPServerProcess",
    "MCPServerManager",
    "get_global_mcp_manager",
    "MCPToolExecutor",
    "MCPToolDiscovery",
    "register_mcp_tools_globally",
    "MCPConfigManager",
    "initialize_mcp_servers_from_config",
    "get_global_mcp_config_manager",
    "create_example_config",
]
