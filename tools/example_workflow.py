"""
Example workflow demonstrating tool calling with different providers.
"""
import asyncio
import json
from typing import Dict, Any
from .example_tools import register_example_tools
from .tool_service import get_global_tool_service

async def create_example_workflow():
    """Create an example workflow configuration that uses tools."""
    
    # Register example tools first
    await register_example_tools()
    
    # Example workflow: Weather assistant with calculations
    workflow_config = {
        "name": "Weather Assistant with Tools",
        "description": "A workflow that uses tools to get weather and perform calculations",
        "global_config": {
            "thinking_tokens_mode": "hide",
            "parallelization": {
                "enabled": True,
                "max_concurrent_steps": 3
            }
        },
        "steps": [
            {
                "id": "get_user_input",
                "type": "input",
                "description": "Get user's location and calculation request",
                "config": {
                    "prompt": "Enter a location for weather and a calculation (e.g., 'Weather in New York and calculate 15 * 8')"
                },
                "outputs": ["user_request"]
            },
            {
                "id": "tool_assistant",
                "type": "tool_call",
                "description": "Use tools to handle the user's request",
                "config": {
                    "provider": "ollama",
                    "model": "llama3.1:8b",
                    "tool_category": "utilities",
                    "tool_choice": "auto",
                    "max_iterations": 3,
                    "auto_execute": True,
                    "prompt": "You are a helpful assistant with access to tools. Use the appropriate tools to fulfill the user's request. If they ask for weather, use the get_weather tool. If they ask for calculations, use the calculate tool.",
                    "temperature": 0.1
                },
                "inputs": {
                    "message": {"from_step": "get_user_input", "key": "user_request"}
                },
                "outputs": ["tool_response"]
            },
            {
                "id": "format_output",
                "type": "output",
                "description": "Format and display the final response",
                "config": {
                    "format": "json"
                },
                "inputs": {
                    "data": {"from_step": "tool_assistant", "key": "final_response"}
                }
            }
        ]
    }
    
    return workflow_config

async def create_multi_provider_workflow():
    """Create a workflow that demonstrates different providers."""
    
    await register_example_tools()
    
    workflow_config = {
        "name": "Multi-Provider Tool Calling Demo",
        "description": "Demonstrates tool calling across different LLM providers",
        "global_config": {
            "thinking_tokens_mode": "hide",
            "parallelization": {
                "enabled": True,
                "max_concurrent_steps": 2
            }
        },
        "steps": [
            {
                "id": "input_query",
                "type": "input",
                "description": "Get user query",
                "config": {
                    "prompt": "Enter your query (e.g., 'What's the weather in London and calculate 25 * 4')"
                },
                "outputs": ["query"]
            },
            {
                "id": "ollama_tools",
                "type": "tool_call",
                "description": "Use Ollama with tools",
                "config": {
                    "provider": "ollama",
                    "model": "llama3.1:8b",
                    "tools": ["get_weather", "calculate"],
                    "tool_choice": "auto",
                    "max_iterations": 2,
                    "auto_execute": True,
                    "prompt": "You are a helpful assistant. Use tools to answer the user's query.",
                    "temperature": 0.1
                },
                "inputs": {
                    "message": {"from_step": "input_query", "key": "query"}
                },
                "outputs": ["ollama_response"]
            },
            {
                "id": "gemini_tools",
                "type": "tool_call",
                "description": "Use Gemini with tools",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "tool_category": "utilities",
                    "tool_choice": "auto",
                    "max_iterations": 2,
                    "auto_execute": True,
                    "prompt": "You are a helpful assistant. Use the available tools to provide comprehensive answers.",
                    "temperature": 0.1
                },
                "inputs": {
                    "message": {"from_step": "input_query", "key": "query"}
                },
                "outputs": ["gemini_response"]
            },
            {
                "id": "compare_results",
                "type": "output",
                "description": "Compare results from different providers",
                "config": {
                    "format": "json"
                },
                "inputs": {
                    "ollama_result": {"from_step": "ollama_tools", "key": "final_response"},
                    "gemini_result": {"from_step": "gemini_tools", "key": "final_response"}
                }
            }
        ]
    }
    
    return workflow_config

async def test_tool_service_directly():
    """Test the tool service directly without workflows."""
    
    # Register tools
    await register_example_tools()
    
    # Get tool service
    tool_service = await get_global_tool_service()
    
    # Test 1: List available tools
    print("\n=== Available Tools ===")
    tools = await tool_service.get_available_tools()
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Test 2: Get tool configuration for different providers
    print("\n=== Tool Configuration for Different Providers ===")
    providers = ["openai", "anthropic", "gemini", "ollama"]
    
    for provider in providers:
        print(f"\n{provider.upper()} Configuration:")
        config = await tool_service.get_tool_calling_config(
            provider=provider,
            tool_names=["get_weather", "calculate"]
        )
        print(json.dumps(config, indent=2))
    
    # Test 3: Search tools
    print("\n=== Search Tools ===")
    search_results = await tool_service.search_tools("weather")
    print(f"Found {len(search_results)} tools matching 'weather':")
    for tool in search_results:
        print(f"- {tool.name}: {tool.description}")
    
    # Test 4: Get tool statistics
    print("\n=== Tool Statistics ===")
    stats = await tool_service.get_tool_stats()
    print(json.dumps(stats, indent=2))

async def main():
    """Main function to demonstrate tool service usage."""
    
    print("=== Tool Service Demo ===")
    
    # Test tool service directly
    await test_tool_service_directly()
    
    # Create example workflows
    print("\n=== Creating Example Workflows ===")
    
    basic_workflow = await create_example_workflow()
    multi_provider_workflow = await create_multi_provider_workflow()
    
    print("Basic workflow created:")
    print(f"- Name: {basic_workflow['name']}")
    print(f"- Steps: {len(basic_workflow['steps'])}")
    
    print("\nMulti-provider workflow created:")
    print(f"- Name: {multi_provider_workflow['name']}")
    print(f"- Steps: {len(multi_provider_workflow['steps'])}")
    
    # Save workflows to files
    with open("/tmp/basic_tool_workflow.json", "w") as f:
        json.dump(basic_workflow, f, indent=2)
    
    with open("/tmp/multi_provider_tool_workflow.json", "w") as f:
        json.dump(multi_provider_workflow, f, indent=2)
    
    print("\nWorkflows saved to /tmp/")
    print("- /tmp/basic_tool_workflow.json")
    print("- /tmp/multi_provider_tool_workflow.json")

if __name__ == "__main__":
    asyncio.run(main())