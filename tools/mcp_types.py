"""
MCP (Model Context Protocol) types and interfaces for tool integration.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class MCPTransport(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str
    command: List[str]
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    env_credentials: Optional[Dict[str, str]] = None  # Maps env var names to credential keys
    transport: MCPTransport = MCPTransport.STDIO
    timeout: Optional[int] = 30
    auto_restart: bool = True
    working_directory: Optional[str] = None
    required_credentials: Optional[List[str]] = None  # List of required credential keys

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None

class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""
    jsonrpc: str = "2.0"
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None

class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPResource(BaseModel):
    """MCP resource information."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

class MCPPrompt(BaseModel):
    """MCP prompt template."""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None

class MCPServerCapabilities(BaseModel):
    """MCP server capabilities."""
    experimental: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None

class MCPClientCapabilities(BaseModel):
    """MCP client capabilities."""
    experimental: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None

class MCPInitializeParams(BaseModel):
    """MCP initialize request parameters."""
    protocolVersion: str = "2024-11-05"
    capabilities: MCPClientCapabilities
    clientInfo: Dict[str, str]

class MCPInitializeResult(BaseModel):
    """MCP initialize response result."""
    protocolVersion: str
    capabilities: MCPServerCapabilities
    serverInfo: Dict[str, str]

class MCPCallToolParams(BaseModel):
    """Parameters for calling an MCP tool."""
    name: str
    arguments: Optional[Dict[str, Any]] = None

class MCPToolResult(BaseModel):
    """Result from calling an MCP tool."""
    content: List[Dict[str, Any]]
    isError: Optional[bool] = None

class MCPListToolsResult(BaseModel):
    """Result from listing MCP tools."""
    tools: List[MCPToolInfo]

class MCPListResourcesResult(BaseModel):
    """Result from listing MCP resources."""
    resources: List[MCPResource]

class MCPListPromptsResult(BaseModel):
    """Result from listing MCP prompts."""
    prompts: List[MCPPrompt]

# MCP Method Names
class MCPMethods:
    """Standard MCP method names."""
    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    PING = "ping"
    
    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    
    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"
    
    # Completion
    COMPLETION_COMPLETE = "completion/complete"
    
    # Notifications
    NOTIFICATIONS_CANCELLED = "notifications/cancelled"
    NOTIFICATIONS_PROGRESS = "notifications/progress"
    NOTIFICATIONS_MESSAGE = "notifications/message"
    NOTIFICATIONS_RESOURCES_LIST_CHANGED = "notifications/resources/list_changed"
    NOTIFICATIONS_RESOURCES_UPDATED = "notifications/resources/updated"
    NOTIFICATIONS_TOOLS_LIST_CHANGED = "notifications/tools/list_changed"
    NOTIFICATIONS_PROMPTS_LIST_CHANGED = "notifications/prompts/list_changed"

# Common MCP Error Codes
class MCPErrorCodes:
    """Standard MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    TOOL_NOT_FOUND = -32000
    RESOURCE_NOT_FOUND = -32001
    PROMPT_NOT_FOUND = -32002
    SERVER_ERROR = -32003