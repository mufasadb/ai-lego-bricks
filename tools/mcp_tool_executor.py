"""
MCP tool executor that integrates with the existing tool system.
"""
import logging
from typing import Dict, List, Optional, Any
from .tool_types import ToolCall, ToolResult, ToolSchema, ParameterType, ToolParameter, Tool
from .secure_tool_executor import SecureToolExecutor
from .mcp_server_manager import get_global_mcp_manager

logger = logging.getLogger(__name__)

class MCPToolExecutor(SecureToolExecutor):
    """Tool executor that calls tools on MCP servers."""
    
    def __init__(self, server_name: str, tool_name: str, 
                 credential_manager=None):
        """
        Initialize MCP tool executor.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool on the server
            credential_manager: Optional credential manager
        """
        self.server_name = server_name
        self.tool_name = tool_name
        
        # No specific credentials required for MCP tools by default
        # Server-specific credentials should be handled in server config
        super().__init__(credential_manager, required_credentials=[])
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute the MCP tool call."""
        try:
            # Get the MCP server manager
            manager = await get_global_mcp_manager()
            
            # Get the specific server
            server = await manager.get_server(self.server_name)
            if not server:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=None,
                    error=f"MCP server '{self.server_name}' not found"
                )
            
            if not server.is_initialized:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=None,
                    error=f"MCP server '{self.server_name}' is not initialized"
                )
            
            # Call the tool on the server
            result = await manager.call_tool_on_server(
                self.server_name, 
                self.tool_name, 
                tool_call.parameters
            )
            
            # Extract content from MCP result
            content = result.get("content", [])
            is_error = result.get("isError", False)
            
            if is_error:
                error_message = self._extract_error_message(content)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=None,
                    error=error_message
                )
            else:
                # Convert MCP content to a more usable format
                processed_result = self._process_mcp_content(content)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=processed_result,
                    error=None
                )
        
        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.tool_name}' on server '{self.server_name}': {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    def _extract_error_message(self, content: List[Dict[str, Any]]) -> str:
        """Extract error message from MCP content."""
        if not content:
            return "Unknown MCP tool error"
        
        error_parts = []
        for item in content:
            if item.get("type") == "text":
                error_parts.append(item.get("text", ""))
        
        return " ".join(error_parts) if error_parts else "Unknown MCP tool error"
    
    def _process_mcp_content(self, content: List[Dict[str, Any]]) -> Any:
        """Process MCP content into a more usable format."""
        if not content:
            return None
        
        # If single text content, return just the text
        if len(content) == 1 and content[0].get("type") == "text":
            return content[0].get("text")
        
        # If single JSON content, return the parsed data
        if len(content) == 1 and content[0].get("type") == "application/json":
            try:
                import json
                return json.loads(content[0].get("data", "{}"))
            except (json.JSONDecodeError, KeyError):
                pass
        
        # For mixed or complex content, return the full structure
        return content


class MCPToolDiscovery:
    """Discovers and converts MCP tools to our tool format."""
    
    @staticmethod
    async def discover_tools_from_server(server_name: str) -> List[Tool]:
        """Discover all tools from an MCP server and convert them to our Tool format."""
        tools = []
        
        try:
            manager = await get_global_mcp_manager()
            server = await manager.get_server(server_name)
            
            if not server or not server.is_initialized:
                logger.warning(f"MCP server '{server_name}' is not available for tool discovery")
                return tools
            
            # Get tools from server
            mcp_tools = await server.list_tools()
            
            for mcp_tool_info in mcp_tools:
                try:
                    tool = MCPToolDiscovery._convert_mcp_tool_to_tool(server_name, mcp_tool_info)
                    tools.append(tool)
                except Exception as e:
                    logger.error(f"Failed to convert MCP tool '{mcp_tool_info.get('name', 'unknown')}': {e}")
            
            logger.info(f"Discovered {len(tools)} tools from MCP server '{server_name}'")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from MCP server '{server_name}': {e}")
        
        return tools
    
    @staticmethod
    def _convert_mcp_tool_to_tool(server_name: str, mcp_tool_info: Dict[str, Any]) -> Tool:
        """Convert an MCP tool info to our Tool format."""
        name = mcp_tool_info["name"]
        description = mcp_tool_info.get("description", "")
        input_schema = mcp_tool_info.get("inputSchema", {})
        
        # Convert MCP input schema to our ToolParameter format
        parameters = MCPToolDiscovery._convert_schema_to_tool_parameter(input_schema)
        
        # Create tool schema
        schema = ToolSchema(
            name=f"mcp_{server_name}_{name}",  # Prefix with server name to avoid conflicts
            description=f"[MCP:{server_name}] {description}",
            parameters=parameters
        )
        
        # Create executor
        executor = MCPToolExecutor(server_name, name)
        
        return Tool(schema=schema, executor=executor)
    
    @staticmethod
    def _convert_schema_to_tool_parameter(schema: Dict[str, Any]) -> Optional[ToolParameter]:
        """Convert JSON schema to ToolParameter."""
        if not schema:
            return None
        
        schema_type = schema.get("type", "object")
        
        # Map JSON schema types to our ParameterType
        type_mapping = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "number": ParameterType.NUMBER,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT
        }
        
        param_type = type_mapping.get(schema_type, ParameterType.OBJECT)
        
        # Build parameter
        param = ToolParameter(
            type=param_type,
            description=schema.get("description"),
            enum=schema.get("enum"),
            format=schema.get("format"),
            nullable=schema.get("nullable")
        )
        
        # Handle array items
        if param_type == ParameterType.ARRAY and "items" in schema:
            param.items = MCPToolDiscovery._convert_schema_to_tool_parameter(schema["items"])
        
        # Handle object properties
        if param_type == ParameterType.OBJECT and "properties" in schema:
            properties = {}
            for prop_name, prop_schema in schema["properties"].items():
                properties[prop_name] = MCPToolDiscovery._convert_schema_to_tool_parameter(prop_schema)
            param.properties = properties
            param.required = schema.get("required", [])
        
        return param
    
    @staticmethod
    async def discover_all_tools() -> Dict[str, List[Tool]]:
        """Discover tools from all available MCP servers."""
        all_tools = {}
        
        manager = await get_global_mcp_manager()
        server_names = await manager.list_servers()
        
        for server_name in server_names:
            tools = await MCPToolDiscovery.discover_tools_from_server(server_name)
            all_tools[server_name] = tools
        
        return all_tools


async def register_mcp_tools_globally(server_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Register MCP tools with the global tool registry.
    
    Args:
        server_names: Specific servers to register tools from, or None for all servers
    
    Returns:
        Registration results
    """
    from .tool_registry import get_global_registry
    
    registry = await get_global_registry()
    results = {
        "registered": [],
        "failed": [],
        "total": 0
    }
    
    if server_names is None:
        manager = await get_global_mcp_manager()
        server_names = await manager.list_servers()
    
    for server_name in server_names:
        try:
            tools = await MCPToolDiscovery.discover_tools_from_server(server_name)
            
            for tool in tools:
                try:
                    await registry.register_tool(tool, f"mcp_{server_name}")
                    results["registered"].append({
                        "name": tool.schema.name,
                        "server": server_name,
                        "original_name": tool.executor.tool_name
                    })
                    results["total"] += 1
                except Exception as e:
                    results["failed"].append({
                        "name": tool.schema.name,
                        "server": server_name,
                        "error": str(e)
                    })
                    results["total"] += 1
            
        except Exception as e:
            logger.error(f"Failed to register tools from MCP server '{server_name}': {e}")
            results["failed"].append({
                "server": server_name,
                "error": str(e)
            })
    
    logger.info(f"MCP tool registration complete: {len(results['registered'])} registered, {len(results['failed'])} failed")
    return results