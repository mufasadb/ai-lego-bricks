"""
Tool types and interfaces for the tool service.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class ToolChoiceType(str, Enum):
    """Tool choice options for different providers."""
    AUTO = "auto"
    ANY = "any"
    NONE = "none"
    SPECIFIC = "specific"

class ParameterType(str, Enum):
    """Supported parameter types for tool schemas."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

class ToolParameter(BaseModel):
    """Represents a tool parameter with type and validation."""
    type: ParameterType
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    items: Optional["ToolParameter"] = None
    properties: Optional[Dict[str, "ToolParameter"]] = None
    required: Optional[List[str]] = None
    format: Optional[str] = None
    nullable: Optional[bool] = None

class ToolSchema(BaseModel):
    """Universal tool schema that can be converted to any provider format."""
    name: str
    description: str
    parameters: Optional[ToolParameter] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameter_to_json_schema(self.parameters) if self.parameters else {}
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tools format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._parameter_to_json_schema(self.parameters) if self.parameters else {}
        }
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Google Gemini function declarations format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._parameter_to_json_schema(self.parameters) if self.parameters else {}
        }
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama tools format (similar to OpenAI)."""
        return self.to_openai_format()
    
    def _parameter_to_json_schema(self, param: Optional[ToolParameter]) -> Dict[str, Any]:
        """Convert ToolParameter to JSON schema format."""
        if not param:
            return {}
        
        schema = {"type": param.type.value}
        
        if param.description:
            schema["description"] = param.description
        if param.enum:
            schema["enum"] = param.enum
        if param.format:
            schema["format"] = param.format
        if param.nullable is not None:
            schema["nullable"] = param.nullable
        
        if param.type == ParameterType.ARRAY and param.items:
            schema["items"] = self._parameter_to_json_schema(param.items)
        
        if param.type == ParameterType.OBJECT and param.properties:
            schema["properties"] = {
                key: self._parameter_to_json_schema(value) 
                for key, value in param.properties.items()
            }
            if param.required:
                schema["required"] = param.required
        
        return schema

class ToolCall(BaseModel):
    """Represents a tool call request from an LLM."""
    id: Optional[str] = None
    name: str
    parameters: Dict[str, Any]

class ToolResult(BaseModel):
    """Represents the result of a tool execution."""
    tool_call_id: Optional[str] = None
    name: str
    result: Any
    error: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool result format."""
        return {
            "tool_call_id": self.tool_call_id,
            "role": "tool",
            "content": str(self.result) if self.error is None else f"Error: {self.error}"
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool result format."""
        return {
            "role": "tool",
            "content": str(self.result) if self.error is None else f"Error: {self.error}"
        }
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini tool result format."""
        return {
            "role": "function",
            "parts": [{
                "functionResponse": {
                    "name": self.name,
                    "response": self.result if self.error is None else {"error": self.error}
                }
            }]
        }

class ToolExecutor(ABC):
    """Abstract base class for tool executors."""
    
    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        pass

class Tool(BaseModel):
    """Represents a complete tool with schema and executor."""
    schema: ToolSchema
    executor: ToolExecutor
    
    class Config:
        arbitrary_types_allowed = True

class ToolChoice(BaseModel):
    """Represents tool choice configuration."""
    type: ToolChoiceType
    function_name: Optional[str] = None
    
    def to_openai_format(self) -> Union[str, Dict[str, Any]]:
        """Convert to OpenAI tool_choice format."""
        if self.type == ToolChoiceType.AUTO:
            return "auto"
        elif self.type == ToolChoiceType.NONE:
            return "none"
        elif self.type == ToolChoiceType.SPECIFIC and self.function_name:
            return {
                "type": "function",
                "function": {"name": self.function_name}
            }
        return "auto"
    
    def to_anthropic_format(self) -> Union[str, Dict[str, Any]]:
        """Convert to Anthropic tool_choice format."""
        if self.type == ToolChoiceType.AUTO:
            return "auto"
        elif self.type == ToolChoiceType.NONE:
            return "none"
        elif self.type == ToolChoiceType.ANY:
            return "any"
        elif self.type == ToolChoiceType.SPECIFIC and self.function_name:
            return {
                "type": "tool",
                "name": self.function_name
            }
        return "auto"
    
    def to_gemini_format(self) -> str:
        """Convert to Gemini function_calling_config format."""
        if self.type == ToolChoiceType.AUTO:
            return "AUTO"
        elif self.type == ToolChoiceType.NONE:
            return "NONE"
        elif self.type == ToolChoiceType.ANY:
            return "ANY"
        return "AUTO"

# Allow forward references for ToolParameter
ToolParameter.model_rebuild()