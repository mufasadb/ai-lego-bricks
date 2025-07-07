"""
Example tool implementations for demonstration purposes.
"""
import asyncio
import json
from typing import Dict, Any
from datetime import datetime
from .tool_types import ToolSchema, ToolParameter, ParameterType, ToolExecutor, ToolCall, ToolResult, Tool

class WeatherExecutor(ToolExecutor):
    """Example weather tool executor."""
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute weather lookup (mock implementation)."""
        try:
            location = tool_call.parameters.get("location", "Unknown")
            
            # Mock weather data
            weather_data = {
                "location": location,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "45%",
                "wind": "10 km/h",
                "timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=weather_data
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )

class CalculatorExecutor(ToolExecutor):
    """Example calculator tool executor."""
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute mathematical operations."""
        try:
            expression = tool_call.parameters.get("expression", "")
            
            # Basic safety check - only allow basic math operations
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains invalid characters")
            
            # Evaluate expression safely
            try:
                result = eval(expression)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={"expression": expression, "result": result}
                )
            except Exception as e:
                raise ValueError(f"Invalid mathematical expression: {str(e)}")
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )

class FileManagerExecutor(ToolExecutor):
    """Example file management tool executor."""
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute file operations (mock implementation)."""
        try:
            operation = tool_call.parameters.get("operation", "list")
            path = tool_call.parameters.get("path", ".")
            
            if operation == "list":
                # Mock file listing
                files = [
                    {"name": "document.txt", "size": 1024, "type": "file"},
                    {"name": "images", "size": 0, "type": "directory"},
                    {"name": "config.json", "size": 512, "type": "file"}
                ]
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={"path": path, "files": files}
                )
            elif operation == "read":
                # Mock file reading
                content = f"Mock content from file: {path}"
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={"path": path, "content": content}
                )
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )

class WebSearchExecutor(ToolExecutor):
    """Example web search tool executor."""
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute web search (mock implementation)."""
        try:
            query = tool_call.parameters.get("query", "")
            limit = tool_call.parameters.get("limit", 5)
            
            # Mock search results
            results = [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/result-{i+1}",
                    "snippet": f"This is a mock search result snippet for query '{query}'"
                }
                for i in range(min(limit, 5))
            ]
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={"query": query, "results": results}
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )

def create_example_tools():
    """Create example tools for demonstration."""
    
    # Weather tool
    weather_schema = ToolSchema(
        name="get_weather",
        description="Get current weather information for a location",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "location": ToolParameter(
                    type=ParameterType.STRING,
                    description="The city or location to get weather for"
                )
            },
            required=["location"]
        )
    )
    weather_tool = Tool(schema=weather_schema, executor=WeatherExecutor())
    
    # Calculator tool
    calculator_schema = ToolSchema(
        name="calculate",
        description="Perform mathematical calculations",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "expression": ToolParameter(
                    type=ParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                )
            },
            required=["expression"]
        )
    )
    calculator_tool = Tool(schema=calculator_schema, executor=CalculatorExecutor())
    
    # File manager tool
    file_manager_schema = ToolSchema(
        name="file_manager",
        description="Manage and interact with files and directories",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "operation": ToolParameter(
                    type=ParameterType.STRING,
                    description="Operation to perform",
                    enum=["list", "read", "write"]
                ),
                "path": ToolParameter(
                    type=ParameterType.STRING,
                    description="File or directory path"
                ),
                "content": ToolParameter(
                    type=ParameterType.STRING,
                    description="Content to write (for write operation)"
                )
            },
            required=["operation", "path"]
        )
    )
    file_manager_tool = Tool(schema=file_manager_schema, executor=FileManagerExecutor())
    
    # Web search tool
    web_search_schema = ToolSchema(
        name="web_search",
        description="Search the web for information",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "query": ToolParameter(
                    type=ParameterType.STRING,
                    description="Search query"
                ),
                "limit": ToolParameter(
                    type=ParameterType.INTEGER,
                    description="Maximum number of results to return"
                )
            },
            required=["query"]
        )
    )
    web_search_tool = Tool(schema=web_search_schema, executor=WebSearchExecutor())
    
    return {
        "weather": weather_tool,
        "calculator": calculator_tool,
        "file_manager": file_manager_tool,
        "web_search": web_search_tool
    }

async def register_example_tools():
    """Register example tools globally."""
    from .tool_registry import register_tool_globally
    
    tools = create_example_tools()
    
    # Register tools with categories
    await register_tool_globally(tools["weather"], "utilities")
    await register_tool_globally(tools["calculator"], "utilities")
    await register_tool_globally(tools["file_manager"], "system")
    await register_tool_globally(tools["web_search"], "information")
    
    print("Example tools registered successfully!")
    print(f"Registered {len(tools)} tools:")
    for name, tool in tools.items():
        print(f"  - {tool.schema.name}: {tool.schema.description}")

if __name__ == "__main__":
    asyncio.run(register_example_tools())