"""
Generic tool interface for LLM services
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json


class ToolParameterType(str, Enum):
    """Tool parameter types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
    properties: Optional[Dict[str, 'ToolParameter']] = None  # For object types
    items: Optional['ToolParameter'] = None  # For array types


class ToolDefinition(BaseModel):
    """Tool definition for LLM registration"""
    name: str
    description: str
    parameters: List[ToolParameter]
    category: str = "general"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = self._param_to_openai_schema(param)
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = self._param_to_anthropic_schema(param)
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = self._param_to_gemini_schema(param)
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def _param_to_openai_schema(self, param: ToolParameter) -> Dict[str, Any]:
        """Convert parameter to OpenAI schema"""
        schema = {
            "type": param.type.value,
            "description": param.description
        }
        
        if param.enum:
            schema["enum"] = param.enum
        if param.default is not None:
            schema["default"] = param.default
        if param.type == ToolParameterType.OBJECT and param.properties:
            schema["properties"] = {
                name: self._param_to_openai_schema(prop)
                for name, prop in param.properties.items()
            }
        if param.type == ToolParameterType.ARRAY and param.items:
            schema["items"] = self._param_to_openai_schema(param.items)
        
        return schema
    
    def _param_to_anthropic_schema(self, param: ToolParameter) -> Dict[str, Any]:
        """Convert parameter to Anthropic schema"""
        schema = {
            "type": param.type.value,
            "description": param.description
        }
        
        if param.enum:
            schema["enum"] = param.enum
        if param.type == ToolParameterType.OBJECT and param.properties:
            schema["properties"] = {
                name: self._param_to_anthropic_schema(prop)
                for name, prop in param.properties.items()
            }
        if param.type == ToolParameterType.ARRAY and param.items:
            schema["items"] = self._param_to_anthropic_schema(param.items)
        
        return schema
    
    def _param_to_gemini_schema(self, param: ToolParameter) -> Dict[str, Any]:
        """Convert parameter to Gemini schema"""
        schema = {
            "type": param.type.value,
            "description": param.description
        }
        
        if param.enum:
            schema["enum"] = param.enum
        if param.type == ToolParameterType.OBJECT and param.properties:
            schema["properties"] = {
                name: self._param_to_gemini_schema(prop)
                for name, prop in param.properties.items()
            }
        if param.type == ToolParameterType.ARRAY and param.items:
            schema["items"] = self._param_to_gemini_schema(param.items)
        
        return schema


class ToolResult(BaseModel):
    """Result of tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM registration"""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute tool with given parameters"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters"""
        pass
    
    def get_category(self) -> str:
        """Get tool category"""
        return self.get_definition().category


class UniversalToolRegistry:
    """Registry that adapts tools to any LLM provider"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_groups: Dict[str, List[str]] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: Tool, group: str = "default"):
        """Register a tool in the registry"""
        definition = tool.get_definition()
        self.tools[definition.name] = tool
        
        # Add to group
        if group not in self.tool_groups:
            self.tool_groups[group] = []
        self.tool_groups[group].append(definition.name)
        
        # Add to category
        category = tool.get_category()
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(definition.name)
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool from all groups"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            
            # Remove from groups
            for group_tools in self.tool_groups.values():
                if tool_name in group_tools:
                    group_tools.remove(tool_name)
            
            # Remove from categories
            for category_tools in self.categories.values():
                if tool_name in category_tools:
                    category_tools.remove(tool_name)
    
    def get_tools_for_provider(self, provider: str, group: str = "default") -> List[Dict[str, Any]]:
        """Get tools formatted for specific LLM provider"""
        tools = []
        tool_names = self.tool_groups.get(group, [])
        
        for tool_name in tool_names:
            if tool_name not in self.tools:
                continue
                
            tool = self.tools[tool_name]
            definition = tool.get_definition()
            
            if provider.lower() in ["openai", "ollama"]:
                tools.append(definition.to_openai_format())
            elif provider.lower() == "anthropic":
                tools.append(definition.to_anthropic_format())
            elif provider.lower() == "gemini":
                tools.append(definition.to_gemini_format())
            else:
                # Default to OpenAI format
                tools.append(definition.to_openai_format())
        
        return tools
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools by category"""
        return self.categories.get(category, [])
    
    def get_available_groups(self) -> List[str]:
        """Get available tool groups"""
        return list(self.tool_groups.keys())
    
    def get_available_categories(self) -> List[str]:
        """Get available tool categories"""
        return list(self.categories.keys())
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute tool by name"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        
        # Validate parameters
        if not tool.validate_parameters(parameters):
            return ToolResult(
                success=False,
                error=f"Invalid parameters for tool '{tool_name}'"
            )
        
        try:
            return await tool.execute(parameters)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name"""
        if tool_name in self.tools:
            return self.tools[tool_name].get_definition()
        return None


class ToolEnabledLLMService(ABC):
    """Base class for LLM services with tool support"""
    
    def __init__(self, tool_registry: UniversalToolRegistry):
        self.tool_registry = tool_registry
        self.provider_name = self._get_provider_name()
    
    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get provider name for tool formatting"""
        pass
    
    def get_available_tools(self, tool_group: str = "default") -> List[Dict[str, Any]]:
        """Get tools formatted for this provider"""
        return self.tool_registry.get_tools_for_provider(self.provider_name, tool_group)
    
    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls from LLM response"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            parameters = tool_call.get("parameters", {})
            
            result = await self.tool_registry.execute_tool(tool_name, parameters)
            
            results.append({
                "tool_call_id": tool_call.get("id", ""),
                "name": tool_name,
                "success": result.success,
                "result": result.result,
                "error": result.error
            })
        
        return results
    
    @abstractmethod
    async def chat_with_tools(self, message: str, tool_group: str = "default") -> str:
        """Chat with tool support"""
        pass


# Example tool implementations
class MemorySearchTool(Tool):
    """Tool for searching through stored memories"""
    
    def __init__(self, memory_service):
        self.memory_service = memory_service
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_search",
            description="Search through stored memories and knowledge",
            category="memory",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query to find relevant memories",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type=ToolParameterType.INTEGER,
                    description="Maximum number of results to return",
                    required=False,
                    default=10
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            query = parameters["query"]
            limit = parameters.get("limit", 10)
            
            # This would need to be async in real implementation
            results = await self._search_memory(query, limit)
            
            return ToolResult(
                success=True,
                result={
                    "results": results,
                    "count": len(results)
                },
                metadata={
                    "query": query,
                    "limit": limit
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def _search_memory(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search memory service (placeholder for async implementation)"""
        # In real implementation, this would be async
        if hasattr(self.memory_service, 'search_async'):
            return await self.memory_service.search_async(query, limit=limit)
        else:
            # Fallback to sync method
            return self.memory_service.search(query, limit=limit)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return "query" in parameters and isinstance(parameters["query"], str)


class DocumentProcessingTool(Tool):
    """Tool for processing documents"""
    
    def __init__(self, pdf_service):
        self.pdf_service = pdf_service
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="process_document",
            description="Extract and process text from documents",
            category="documents",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type=ToolParameterType.STRING,
                    description="Path to the document file",
                    required=True
                ),
                ToolParameter(
                    name="extract_images",
                    type=ToolParameterType.BOOLEAN,
                    description="Whether to extract images from the document",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            file_path = parameters["file_path"]
            extract_images = parameters.get("extract_images", False)
            
            # This would need to be async in real implementation
            result = await self._process_document(file_path, extract_images)
            
            return ToolResult(
                success=True,
                result={
                    "text": result.get("text", ""),
                    "images": result.get("images", []) if extract_images else [],
                    "metadata": result.get("metadata", {})
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def _process_document(self, file_path: str, extract_images: bool) -> Dict[str, Any]:
        """Process document (placeholder for async implementation)"""
        # In real implementation, this would be async
        if hasattr(self.pdf_service, 'extract_text_async'):
            return await self.pdf_service.extract_text_async(file_path, extract_images=extract_images)
        else:
            # Fallback to sync method
            return self.pdf_service.extract_text(file_path, extract_images=extract_images)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return "file_path" in parameters and isinstance(parameters["file_path"], str)


class WebSearchTool(Tool):
    """Tool for web search"""
    
    def __init__(self, web_service):
        self.web_service = web_service
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for information",
            category="web",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type=ToolParameterType.INTEGER,
                    description="Number of results to return",
                    required=False,
                    default=10
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            query = parameters["query"]
            num_results = parameters.get("num_results", 10)
            
            results = await self._search_web(query, num_results)
            
            return ToolResult(
                success=True,
                result={
                    "results": results,
                    "count": len(results)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def _search_web(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search web (placeholder for async implementation)"""
        # This would be implemented with actual web search service
        return []
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return "query" in parameters and isinstance(parameters["query"], str)


# Export main classes and functions
__all__ = [
    "Tool",
    "ToolDefinition",
    "ToolParameter",
    "ToolParameterType",
    "ToolResult",
    "UniversalToolRegistry",
    "ToolEnabledLLMService",
    "MemorySearchTool",
    "DocumentProcessingTool",
    "WebSearchTool"
]