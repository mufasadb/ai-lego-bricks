"""
Example usage of MCP integration with AI Lego Bricks tool system.
"""
import asyncio
import logging
from typing import Dict, Any

from tools import (
    MCPServerConfig, MCPTransport, MCPConfigManager, 
    initialize_mcp_servers_from_config, register_mcp_tools_globally,
    get_global_tool_service, get_global_mcp_manager
)

try:
    from credentials import CredentialManager
except ImportError:
    CredentialManager = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_mcp_setup():
    """Example: Basic MCP server setup and tool registration."""
    logger.info("=== Basic MCP Setup Example ===")
    
    # 1. Create a simple MCP server configuration
    config = MCPServerConfig(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["/tmp"],  # Allow access to /tmp directory
        transport=MCPTransport.STDIO,
        timeout=30
    )
    
    # 2. Get the MCP server manager and add the server
    manager = await get_global_mcp_manager()
    
    try:
        await manager.add_server(config)
        logger.info("MCP server added successfully")
        
        # 3. Register the MCP tools with our tool system
        results = await register_mcp_tools_globally(["filesystem"])
        logger.info(f"Tool registration results: {results}")
        
        # 4. List available tools to verify integration
        tool_service = await get_global_tool_service()
        mcp_tools = await tool_service.get_available_tools("mcp_filesystem")
        logger.info(f"Available MCP tools: {[tool.name for tool in mcp_tools]}")
        
        # 5. Example: Use a tool (if available)
        if mcp_tools:
            # Get tool calling config for a provider (e.g., OpenAI)
            config = await tool_service.get_tool_calling_config(
                provider="openai",
                category="mcp_filesystem"
            )
            logger.info(f"Tool calling config: {list(config.get('tools', []))}")
        
    except Exception as e:
        logger.error(f"Error in basic MCP setup: {e}")
    finally:
        # Cleanup
        await manager.remove_server("filesystem")

async def example_config_file_setup():
    """Example: Load MCP servers from configuration files."""
    logger.info("=== Config File Setup Example ===")
    
    # 1. Create example configuration
    config_manager = MCPConfigManager()
    
    # You could save this to a file and load it
    example_config = {
        "servers": {
            "brave_search": {
                "command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
                "env": {
                    "BRAVE_API_KEY": "your-api-key-here"
                }
            },
            "git": {
                "command": ["npx", "-y", "@modelcontextprotocol/server-git"],
                "args": ["--repository", "."]
            }
        }
    }
    
    # Save example config (optional)
    # await config_manager.save_config("./example_mcp_config.json", example_config)
    
    # 2. Initialize servers from config
    try:
        results = await initialize_mcp_servers_from_config(config_manager)
        logger.info(f"Server initialization results: {results}")
        
        # 3. Register all discovered tools
        if results["servers"]:
            server_names = [server["name"] for server in results["servers"]]
            tool_results = await register_mcp_tools_globally(server_names)
            logger.info(f"Tool registration results: {tool_results}")
        
    except Exception as e:
        logger.error(f"Error in config file setup: {e}")

async def example_tool_usage_in_workflow():
    """Example: Using MCP tools in an agent workflow."""
    logger.info("=== Tool Usage in Workflow Example ===")
    
    # This would be used in agent orchestration
    from tools import ToolCall, ToolResult
    
    # Setup a simple MCP server for demonstration
    config = MCPServerConfig(
        name="demo",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["/tmp"]
    )
    
    manager = await get_global_mcp_manager()
    
    try:
        await manager.add_server(config)
        await register_mcp_tools_globally(["demo"])
        
        tool_service = await get_global_tool_service()
        
        # Simulate a tool call (like what would happen in agent orchestration)
        available_tools = await tool_service.get_available_tools("mcp_demo")
        
        if available_tools:
            # Example tool call
            tool = available_tools[0]
            tool_call = ToolCall(
                id="example_001",
                name=tool.name,
                parameters={"path": "/tmp"}  # Example parameters
            )
            
            # Execute the tool
            result = await tool_service.execute_single_tool_call(tool_call)
            logger.info(f"Tool execution result: {result}")
        
    except Exception as e:
        logger.error(f"Error in workflow example: {e}")
    finally:
        await manager.remove_server("demo")

async def example_multiple_providers():
    """Example: Using MCP tools with different LLM providers."""
    logger.info("=== Multiple Provider Example ===")
    
    config = MCPServerConfig(
        name="multi_provider_demo",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["/tmp"]
    )
    
    manager = await get_global_mcp_manager()
    
    try:
        await manager.add_server(config)
        await register_mcp_tools_globally(["multi_provider_demo"])
        
        tool_service = await get_global_tool_service()
        
        # Get the same tools formatted for different providers
        providers = ["openai", "anthropic", "gemini", "ollama"]
        
        for provider in providers:
            try:
                config = await tool_service.get_tool_calling_config(
                    provider=provider,
                    category="mcp_multi_provider_demo"
                )
                logger.info(f"Tools for {provider}: {len(config.get('tools', []))} available")
            except Exception as e:
                logger.error(f"Error getting tools for {provider}: {e}")
        
    except Exception as e:
        logger.error(f"Error in multiple provider example: {e}")
    finally:
        await manager.remove_server("multi_provider_demo")

async def example_credential_integration():
    """Example: Using MCP servers with credential management."""
    logger.info("=== Credential Integration Example ===")
    
    if not CredentialManager:
        logger.info("CredentialManager not available - skipping credential example")
        return
    
    # Setup credential manager with example credentials
    credential_manager = CredentialManager(
        credentials={
            "BRAVE_API_KEY": "your-brave-api-key-from-credential-manager",
            "GITHUB_TOKEN": "your-github-token-from-credential-manager",
            "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/db"
        }
    )
    
    # Create server config that uses credentials instead of hardcoded values
    config = MCPServerConfig(
        name="brave_search_secure",
        command=["npx", "-y", "@modelcontextprotocol/server-brave-search"],
        env_credentials={
            "BRAVE_API_KEY": "BRAVE_API_KEY"  # Maps env var to credential key
        },
        required_credentials=["BRAVE_API_KEY"],  # List required credentials
        transport=MCPTransport.STDIO
    )
    
    # Create manager with credential support
    from tools.mcp_server_manager import MCPServerManager
    manager = MCPServerManager(credential_manager)
    
    try:
        # This will validate credentials before starting the server
        await manager.add_server(config)
        logger.info("MCP server with credentials started successfully")
        
        # Register tools
        await register_mcp_tools_globally(["brave_search_secure"])
        
        logger.info("Credential-secured MCP tools registered")
        
    except ValueError as e:
        logger.error(f"Credential validation failed: {e}")
    except Exception as e:
        logger.error(f"Error in credential integration: {e}")
    finally:
        await manager.shutdown_all()

async def example_config_with_credentials():
    """Example: Configuration file with credential references."""
    logger.info("=== Config with Credentials Example ===")
    
    if not CredentialManager:
        logger.info("CredentialManager not available - skipping config credential example")
        return
    
    # Setup credential manager
    credential_manager = CredentialManager(
        credentials={
            "BRAVE_API_KEY": "your-brave-key",
            "GITHUB_TOKEN": "your-github-token"
        }
    )
    
    # Example of loading config with credential references
    try:
        config_manager = MCPConfigManager()
        
        # Initialize with credential manager
        results = await initialize_mcp_servers_from_config(
            config_manager=config_manager,
            credential_manager=credential_manager
        )
        
        logger.info(f"Server initialization with credentials: {results}")
        
        # This would work with the updated config format that uses env_credentials
        
    except Exception as e:
        logger.error(f"Error in config credential example: {e}")

async def example_custom_mcp_server():
    """Example: Integrating a custom MCP server."""
    logger.info("=== Custom MCP Server Example ===")
    
    # Example of how to integrate a custom MCP server with credentials
    config = MCPServerConfig(
        name="custom_server",
        command=["python", "path/to/your/custom_mcp_server.py"],
        env_credentials={
            "API_KEY": "CUSTOM_SERVER_API_KEY"  # Reference to credential
        },
        env={
            "LOG_LEVEL": "INFO"  # Static environment variable
        },
        working_directory="/path/to/server/directory",
        timeout=60,
        required_credentials=["CUSTOM_SERVER_API_KEY"]
    )
    
    try:
        # This would start your custom server with secure credential handling
        # manager = await get_global_mcp_manager()
        # await manager.add_server(config)
        # await register_mcp_tools_globally(["custom_server"])
        
        logger.info("Custom MCP server integration pattern demonstrated")
        logger.info("Uses env_credentials instead of hardcoded values")
        logger.info("Uncomment the lines above once you have a custom server")
        
    except Exception as e:
        logger.error(f"Error in custom server example: {e}")

async def main():
    """Run all examples."""
    logger.info("Starting MCP integration examples...")
    
    try:
        await example_basic_mcp_setup()
        await example_config_file_setup()
        await example_tool_usage_in_workflow()
        await example_multiple_providers()
        await example_credential_integration()
        await example_config_with_credentials()
        await example_custom_mcp_server()
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    finally:
        # Cleanup all servers
        manager = await get_global_mcp_manager()
        await manager.shutdown_all()
        logger.info("Examples completed and cleaned up")

if __name__ == "__main__":
    # Note: These examples require MCP servers to be available
    # Install with: npm install -g @modelcontextprotocol/server-filesystem
    asyncio.run(main())