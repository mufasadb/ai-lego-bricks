"""
Provider-specific adapters for tool calling.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .tool_types import ToolSchema, ToolCall, ToolResult, ToolChoice

class ProviderAdapter(ABC):
    """Abstract base class for provider-specific tool calling adapters."""
    
    @abstractmethod
    def format_tools_for_request(self, tools: List[ToolSchema]) -> Dict[str, Any]:
        """Format tools for the provider's API request."""
        pass
    
    @abstractmethod
    def format_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Any:
        """Format tool choice for the provider's API request."""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from the provider's API response."""
        pass
    
    @abstractmethod
    def format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results for the provider's API."""
        pass

class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's tool calling format."""
    
    def format_tools_for_request(self, tools: List[ToolSchema]) -> Dict[str, Any]:
        """Format tools for OpenAI API request."""
        return {
            "tools": [tool.to_openai_format() for tool in tools]
        }
    
    def format_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Any:
        """Format tool choice for OpenAI API request."""
        if tool_choice:
            return tool_choice.to_openai_format()
        return "auto"
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from OpenAI API response."""
        tool_calls = []
        
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice and "tool_calls" in choice["message"]:
                    for tool_call in choice["message"]["tool_calls"]:
                        if tool_call["type"] == "function":
                            func = tool_call["function"]
                            tool_calls.append(ToolCall(
                                id=tool_call["id"],
                                name=func["name"],
                                parameters=func.get("arguments", {})
                            ))
        
        return tool_calls
    
    def format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results for OpenAI API."""
        return [result.to_openai_format() for result in results]

class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's tool calling format."""
    
    def format_tools_for_request(self, tools: List[ToolSchema]) -> Dict[str, Any]:
        """Format tools for Anthropic API request."""
        return {
            "tools": [tool.to_anthropic_format() for tool in tools]
        }
    
    def format_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Any:
        """Format tool choice for Anthropic API request."""
        if tool_choice:
            return tool_choice.to_anthropic_format()
        return "auto"
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Anthropic API response."""
        tool_calls = []
        
        if "content" in response:
            for content_block in response["content"]:
                if content_block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        id=content_block.get("id"),
                        name=content_block.get("name"),
                        parameters=content_block.get("input", {})
                    ))
        
        return tool_calls
    
    def format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results for Anthropic API."""
        return [result.to_anthropic_format() for result in results]

class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini's function calling format."""
    
    def format_tools_for_request(self, tools: List[ToolSchema]) -> Dict[str, Any]:
        """Format tools for Gemini API request."""
        return {
            "tools": [{
                "functionDeclarations": [tool.to_gemini_format() for tool in tools]
            }]
        }
    
    def format_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Any:
        """Format tool choice for Gemini API request."""
        if tool_choice:
            return {
                "function_calling_config": {
                    "mode": tool_choice.to_gemini_format()
                }
            }
        return {
            "function_calling_config": {
                "mode": "AUTO"
            }
        }
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Gemini API response."""
        tool_calls = []
        
        if "candidates" in response:
            for candidate in response["candidates"]:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "functionCall" in part:
                            func_call = part["functionCall"]
                            tool_calls.append(ToolCall(
                                name=func_call.get("name"),
                                parameters=func_call.get("args", {})
                            ))
        
        return tool_calls
    
    def format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results for Gemini API."""
        return [result.to_gemini_format() for result in results]

class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama's tool calling format (similar to OpenAI)."""
    
    def format_tools_for_request(self, tools: List[ToolSchema]) -> Dict[str, Any]:
        """Format tools for Ollama API request."""
        return {
            "tools": [tool.to_ollama_format() for tool in tools]
        }
    
    def format_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Any:
        """Format tool choice for Ollama API request."""
        if tool_choice:
            return tool_choice.to_openai_format()
        return "auto"
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Ollama API response."""
        tool_calls = []
        
        if "message" in response and "tool_calls" in response["message"]:
            for tool_call in response["message"]["tool_calls"]:
                if tool_call["type"] == "function":
                    func = tool_call["function"]
                    tool_calls.append(ToolCall(
                        id=tool_call.get("id"),
                        name=func["name"],
                        parameters=func.get("arguments", {})
                    ))
        
        return tool_calls
    
    def format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results for Ollama API."""
        return [result.to_openai_format() for result in results]

class AdapterFactory:
    """Factory for creating provider-specific adapters."""
    
    _adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter,
        "ollama": OllamaAdapter,
    }
    
    @classmethod
    def get_adapter(cls, provider: str) -> ProviderAdapter:
        """Get an adapter for the specified provider."""
        if provider not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider}")
        
        return cls._adapters[provider]()
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported providers."""
        return list(cls._adapters.keys())
    
    @classmethod
    def register_adapter(cls, provider: str, adapter_class: type) -> None:
        """Register a custom adapter for a provider."""
        cls._adapters[provider] = adapter_class