"""
Main tool service that provides unified interface for tool calling across providers.
"""
import asyncio
from typing import Dict, List, Optional, Any, Union
from .tool_registry import ToolRegistry, get_global_registry
from .provider_adapters import AdapterFactory
from .tool_types import ToolSchema, ToolCall, ToolResult, ToolChoice, Tool

try:
    from credentials import CredentialManager
except ImportError:
    CredentialManager = None

class ToolService:
    """Main service for managing tools and provider interactions."""
    
    def __init__(self, registry: Optional[ToolRegistry] = None, 
                 credential_manager: Optional[CredentialManager] = None):
        self.registry = registry
        self.credential_manager = credential_manager
        self._adapter_factory = AdapterFactory()
    
    async def _get_registry(self) -> ToolRegistry:
        """Get the tool registry (global if not provided)."""
        if self.registry is None:
            return await get_global_registry()
        return self.registry
    
    async def register_tool(self, tool: Tool, category: Optional[str] = None) -> None:
        """Register a tool."""
        registry = await self._get_registry()
        await registry.register_tool(tool, category)
    
    async def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        registry = await self._get_registry()
        return await registry.unregister_tool(name)
    
    async def get_available_tools(self, category: Optional[str] = None) -> List[ToolSchema]:
        """Get all available tool schemas."""
        registry = await self._get_registry()
        return await registry.get_all_schemas(category)
    
    async def prepare_tools_for_provider(
        self, 
        provider: str, 
        tool_names: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare tools in the format required by the specified provider."""
        registry = await self._get_registry()
        
        if tool_names:
            # Get specific tools
            tools = []
            for name in tool_names:
                schema = await registry.get_tool_schema(name)
                if schema:
                    tools.append(schema)
        else:
            # Get all tools or tools from category
            tools = await registry.get_all_schemas(category)
        
        if not tools:
            return {}
        
        adapter = self._adapter_factory.get_adapter(provider)
        return adapter.format_tools_for_request(tools)
    
    async def prepare_tool_choice_for_provider(
        self, 
        provider: str, 
        tool_choice: Optional[ToolChoice] = None
    ) -> Any:
        """Prepare tool choice in the format required by the specified provider."""
        adapter = self._adapter_factory.get_adapter(provider)
        return adapter.format_tool_choice(tool_choice)
    
    async def parse_tool_calls_from_response(
        self, 
        provider: str, 
        response: Dict[str, Any]
    ) -> List[ToolCall]:
        """Parse tool calls from a provider's API response."""
        adapter = self._adapter_factory.get_adapter(provider)
        return adapter.parse_tool_calls(response)
    
    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls."""
        registry = await self._get_registry()
        return await registry.execute_multiple_tools(tool_calls)
    
    async def execute_single_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        registry = await self._get_registry()
        return await registry.execute_tool(tool_call)
    
    async def format_tool_results_for_provider(
        self, 
        provider: str, 
        results: List[ToolResult]
    ) -> List[Dict[str, Any]]:
        """Format tool results for the specified provider."""
        adapter = self._adapter_factory.get_adapter(provider)
        return adapter.format_tool_results(results)
    
    async def get_tool_calling_config(
        self, 
        provider: str, 
        tool_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        """Get complete tool calling configuration for a provider."""
        config = {}
        
        # Add tools
        tools_config = await self.prepare_tools_for_provider(provider, tool_names, category)
        config.update(tools_config)
        
        # Add tool choice
        if tools_config:  # Only add tool choice if tools are available
            choice_config = await self.prepare_tool_choice_for_provider(provider, tool_choice)
            if provider == "gemini":
                config.update(choice_config)
            else:
                config["tool_choice"] = choice_config
        
        return config
    
    async def process_tool_calling_workflow(
        self, 
        provider: str, 
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a complete tool calling workflow: parse calls, execute, format results."""
        # Parse tool calls from response
        tool_calls = await self.parse_tool_calls_from_response(provider, response)
        
        if not tool_calls:
            return {"tool_calls": [], "tool_results": []}
        
        # Execute tool calls
        results = await self.execute_tool_calls(tool_calls)
        
        # Format results for provider
        formatted_results = await self.format_tool_results_for_provider(provider, results)
        
        return {
            "tool_calls": tool_calls,
            "tool_results": results,
            "formatted_results": formatted_results
        }
    
    async def search_tools(self, query: str, category: Optional[str] = None) -> List[ToolSchema]:
        """Search for tools by query."""
        registry = await self._get_registry()
        return await registry.search_tools(query, category)
    
    async def get_tool_categories(self) -> List[str]:
        """Get all available tool categories."""
        registry = await self._get_registry()
        return await registry.get_categories()
    
    async def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return self._adapter_factory.get_supported_providers()
    
    async def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about registered tools."""
        registry = await self._get_registry()
        categories = await registry.get_categories()
        
        stats = {
            "total_tools": registry.get_tool_count(),
            "categories": len(categories),
            "supported_providers": len(self._adapter_factory.get_supported_providers())
        }
        
        # Add per-category counts
        for category in categories:
            tools_in_category = await registry.list_tools(category)
            stats[f"category_{category}"] = len(tools_in_category)
        
        return stats
    
    def set_credential_manager(self, credential_manager: CredentialManager) -> None:
        """Set the credential manager for this tool service."""
        self.credential_manager = credential_manager
    
    def get_credential_manager(self) -> Optional[CredentialManager]:
        """Get the current credential manager."""
        return self.credential_manager
    
    async def register_secure_tool(self, tool: Tool, category: Optional[str] = None,
                                  required_credentials: Optional[List[str]] = None) -> None:
        """
        Register a tool with credential validation.
        
        Args:
            tool: Tool to register
            category: Tool category
            required_credentials: List of required credential keys for this tool
        """
        # Validate credentials if specified
        if required_credentials and self.credential_manager:
            missing = []
            for cred in required_credentials:
                if not self.credential_manager.get_credential(cred):
                    missing.append(cred)
            
            if missing:
                raise ValueError(
                    f"Cannot register tool '{tool.schema.name}': missing credentials {missing}"
                )
        
        await self.register_tool(tool, category)
    
    async def register_tools_with_credentials(self, tool_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Register multiple tools with credential validation.
        
        Args:
            tool_configs: List of tool configuration dictionaries with format:
                {
                    "tool": Tool instance,
                    "category": Optional[str],
                    "required_credentials": Optional[List[str]]
                }
        
        Returns:
            Dictionary with registration results
        """
        results = {
            "registered": [],
            "failed": [],
            "total": len(tool_configs)
        }
        
        for config in tool_configs:
            tool = config["tool"]
            category = config.get("category")
            required_credentials = config.get("required_credentials")
            
            try:
                await self.register_secure_tool(tool, category, required_credentials)
                results["registered"].append({
                    "name": tool.schema.name,
                    "category": category,
                    "credentials_required": required_credentials or []
                })
            except ValueError as e:
                results["failed"].append({
                    "name": tool.schema.name,
                    "error": str(e),
                    "category": category,
                    "credentials_required": required_credentials or []
                })
        
        return results
    
    async def validate_tool_credentials(self, tool_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate that tools have their required credentials available.
        
        Args:
            tool_names: Specific tools to validate, or None for all tools
        
        Returns:
            Validation results dictionary
        """
        if not self.credential_manager:
            return {
                "status": "no_credential_manager",
                "message": "No credential manager configured",
                "tools": []
            }
        
        registry = await self._get_registry()
        
        if tool_names:
            tools_to_check = tool_names
        else:
            tools_to_check = await registry.list_tools()
        
        results = {
            "status": "validated",
            "tools": [],
            "missing_credentials": [],
            "available_tools": 0,
            "unavailable_tools": 0
        }
        
        for tool_name in tools_to_check:
            tool = await registry.get_tool(tool_name)
            if not tool:
                continue
            
            tool_result = {
                "name": tool_name,
                "description": tool.schema.description,
                "available": True,
                "missing_credentials": []
            }
            
            # Check if tool executor has credential requirements
            if hasattr(tool.executor, 'required_credentials'):
                required_creds = tool.executor.required_credentials
                for cred in required_creds:
                    if not self.credential_manager.get_credential(cred):
                        tool_result["missing_credentials"].append(cred)
                        tool_result["available"] = False
            
            if tool_result["available"]:
                results["available_tools"] += 1
            else:
                results["unavailable_tools"] += 1
                results["missing_credentials"].extend(tool_result["missing_credentials"])
            
            results["tools"].append(tool_result)
        
        # Remove duplicates from missing credentials
        results["missing_credentials"] = list(set(results["missing_credentials"]))
        
        return results

# Global service instance
_global_service = ToolService()

async def get_global_tool_service() -> ToolService:
    """Get the global tool service instance."""
    return _global_service