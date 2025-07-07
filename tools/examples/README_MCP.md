# MCP Integration with AI Lego Bricks

This directory contains examples and documentation for integrating Model Context Protocol (MCP) servers with the AI Lego Bricks tool system.

## Overview

The MCP integration allows you to:
- Connect to external MCP servers 
- Automatically discover and register their tools
- Use MCP tools in agent workflows
- Support multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)
- Manage server lifecycles and configurations

## Quick Start

### 1. Install MCP Servers

```bash
# Install common MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-git
```

### 2. Basic Usage

```python
import asyncio
from tools import (
    MCPServerConfig, MCPTransport,
    get_global_mcp_manager, register_mcp_tools_globally,
    get_global_tool_service
)

async def main():
    # Configure MCP server
    config = MCPServerConfig(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["/tmp"],
        transport=MCPTransport.STDIO
    )
    
    # Start server and register tools
    manager = await get_global_mcp_manager()
    await manager.add_server(config)
    await register_mcp_tools_globally(["filesystem"])
    
    # Use tools in workflows
    tool_service = await get_global_tool_service()
    tools = await tool_service.get_available_tools("mcp_filesystem")
    print(f"Available tools: {[t.name for t in tools]}")

asyncio.run(main())
```

### 3. Configuration File with Secure Credentials

Create `mcp_servers.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
      "args": ["/path/to/directory"],
      "required_credentials": []
    },
    "brave_search": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
      "env_credentials": {
        "BRAVE_API_KEY": "BRAVE_API_KEY"
      },
      "required_credentials": ["BRAVE_API_KEY"]
    },
    "github": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env_credentials": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"
      },
      "required_credentials": ["GITHUB_TOKEN"]
    }
  }
}
```

Load with credentials:

```python
from tools import initialize_mcp_servers_from_config
from credentials import CredentialManager

# Setup credential manager with your API keys
credential_manager = CredentialManager()  # Loads from .env or environment

# This will auto-discover and load servers with secure credential handling
results = await initialize_mcp_servers_from_config(
    credential_manager=credential_manager
)
await register_mcp_tools_globally()
```

## Architecture

### Components

1. **MCP Types** (`mcp_types.py`)
   - Standard MCP protocol types and interfaces
   - JSON-RPC request/response handling
   - Server capabilities and configurations

2. **Server Manager** (`mcp_server_manager.py`)
   - Process lifecycle management
   - JSON-RPC communication over stdio
   - Server initialization and health monitoring

3. **Tool Executor** (`mcp_tool_executor.py`)
   - Integrates MCP tools with existing tool system
   - Tool discovery and conversion
   - Execution and result handling

4. **Configuration** (`mcp_config.py`)
   - Configuration file management
   - Auto-discovery from Claude Desktop config
   - Server registration and initialization

### Integration Flow

```
1. Configure MCP Server
   ↓
2. Start Server Process (stdio communication)
   ↓
3. Initialize MCP Protocol (JSON-RPC handshake)
   ↓
4. Discover Available Tools
   ↓
5. Convert to Universal Tool Format
   ↓
6. Register with Tool Registry
   ↓
7. Use in Agent Workflows
```

## Configuration Options

### Server Configuration

```python
MCPServerConfig(
    name="server_name",
    command=["path", "to", "executable"],
    args=["arg1", "arg2"],
    env={"ENV_VAR": "value"},  # Static environment variables
    env_credentials={"API_KEY": "MY_API_KEY_CREDENTIAL"},  # Credential references
    transport=MCPTransport.STDIO,  # or HTTP, WEBSOCKET
    timeout=30,
    auto_restart=True,
    working_directory="/path/to/workdir",
    required_credentials=["MY_API_KEY_CREDENTIAL"]  # Required credential keys
)
```

### Supported Transports

- **STDIO**: JSON-RPC over stdin/stdout (most common)
- **HTTP**: JSON-RPC over HTTP (future support)
- **WebSocket**: JSON-RPC over WebSocket (future support)

## Available MCP Servers

Common MCP servers you can use:

- **@modelcontextprotocol/server-filesystem**: File system operations
- **@modelcontextprotocol/server-git**: Git repository operations  
- **@modelcontextprotocol/server-brave-search**: Web search via Brave
- **@modelcontextprotocol/server-github**: GitHub API operations
- **@modelcontextprotocol/server-postgres**: PostgreSQL database operations
- **@modelcontextprotocol/server-sqlite**: SQLite database operations
- **@modelcontextprotocol/server-puppeteer**: Web browser automation
- **@modelcontextprotocol/server-memory**: Persistent memory/knowledge

## Usage in Agent Workflows

MCP tools automatically integrate with the existing agent orchestration system:

```python
# In agent configuration
{
    "type": "tool_call",
    "tool_choice": "any",
    "tool_categories": ["mcp_filesystem", "mcp_git"],
    "tools": ["mcp_filesystem_read_file", "mcp_git_commit"]
}
```

The tools work with any supported LLM provider (OpenAI, Anthropic, Gemini, Ollama).

## Credential Management

The MCP integration supports secure credential handling through the AI Lego Bricks credential system:

### Configuration Format

```json
{
  "servers": {
    "api_server": {
      "command": ["server-executable"],
      "env_credentials": {
        "API_KEY": "MY_SERVICE_API_KEY",
        "SECRET_TOKEN": "MY_SERVICE_SECRET"
      },
      "required_credentials": ["MY_SERVICE_API_KEY", "MY_SERVICE_SECRET"]
    }
  }
}
```

### Credential Sources

Credentials are pulled from the `CredentialManager`:
- Environment variables
- `.env` files  
- Secure credential stores
- Runtime configuration

### Validation

- Credentials are validated before server startup
- Missing credentials cause startup failure
- Credential references are resolved at runtime
- No hardcoded secrets in configuration files

## Security Considerations

- MCP servers run as separate processes with configured permissions
- Credentials are injected via environment variables at runtime
- No credentials stored in configuration files
- File system access is limited to configured directories
- Network access depends on individual server implementations

## Error Handling

The system handles various error scenarios:

- Server startup failures
- Communication timeouts
- Tool execution errors
- Server crashes (with auto-restart)
- Invalid configurations

## Examples

See the example files:

- `mcp_example.py`: Comprehensive usage examples
- `mcp_config_example.json`: Configuration file template
- `README_MCP.md`: This documentation

## Troubleshooting

### Common Issues

1. **Server not starting**: Check command path and permissions
2. **Tool discovery fails**: Verify server initialization completed
3. **Communication timeout**: Increase timeout in server config
4. **Missing dependencies**: Install required npm packages

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed MCP communication logs.

### Manual Testing

Test MCP servers directly:

```bash
# Start server manually
npx @modelcontextprotocol/server-filesystem /tmp

# Send JSON-RPC initialization
echo '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | npx @modelcontextprotocol/server-filesystem /tmp
```

## Contributing

When adding new MCP features:

1. Follow the existing architecture patterns
2. Add comprehensive error handling
3. Include example usage
4. Update documentation
5. Add tests for new functionality

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/docs)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Claude Desktop MCP Integration](https://claude.ai/docs)