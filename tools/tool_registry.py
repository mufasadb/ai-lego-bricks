"""
Tool registry for managing and discovering tools.
"""
import asyncio
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager
from .tool_types import Tool, ToolSchema, ToolCall, ToolResult

class ToolRegistry:
    """Central registry for managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def register_tool(self, tool: Tool, category: Optional[str] = None) -> None:
        """Register a tool in the registry."""
        async with self._lock:
            self._tools[tool.schema.name] = tool
            
            if category:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(tool.schema.name)
    
    async def unregister_tool(self, name: str) -> bool:
        """Unregister a tool from the registry."""
        async with self._lock:
            if name in self._tools:
                del self._tools[name]
                
                # Remove from categories
                for category_tools in self._categories.values():
                    category_tools.discard(name)
                
                return True
            return False
    
    async def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    async def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """Get a tool schema by name."""
        tool = self._tools.get(name)
        return tool.schema if tool else None
    
    async def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List all tools, optionally filtered by category."""
        if category:
            return list(self._categories.get(category, set()))
        return list(self._tools.keys())
    
    async def get_all_schemas(self, category: Optional[str] = None) -> List[ToolSchema]:
        """Get all tool schemas, optionally filtered by category."""
        tool_names = await self.list_tools(category)
        return [self._tools[name].schema for name in tool_names if name in self._tools]
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"Tool '{tool_call.name}' not found"
            )
        
        try:
            return await tool.executor.execute(tool_call)
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    async def execute_multiple_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute_tool(tool_call) for tool_call in tool_calls]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._categories.keys())
    
    async def search_tools(self, query: str, category: Optional[str] = None) -> List[ToolSchema]:
        """Search for tools by name or description."""
        schemas = await self.get_all_schemas(category)
        query_lower = query.lower()
        
        matching_schemas = []
        for schema in schemas:
            if (query_lower in schema.name.lower() or 
                query_lower in schema.description.lower()):
                matching_schemas.append(schema)
        
        return matching_schemas
    
    @asynccontextmanager
    async def batch_register(self):
        """Context manager for batch registering tools."""
        async with self._lock:
            yield self
    
    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_tool_count(self) -> int:
        """Get the total number of registered tools."""
        return len(self._tools)

# Global registry instance
_global_registry = ToolRegistry()

async def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry

async def register_tool_globally(tool: Tool, category: Optional[str] = None) -> None:
    """Register a tool in the global registry."""
    registry = await get_global_registry()
    await registry.register_tool(tool, category)

async def get_tool_globally(name: str) -> Optional[Tool]:
    """Get a tool from the global registry."""
    registry = await get_global_registry()
    return await registry.get_tool(name)