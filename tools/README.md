# AI Lego Bricks Tool Service

Universal tool calling service for building AI agents with LLM providers (OpenAI, Anthropic, Gemini, Ollama). Define tools once, use everywhere.

## Key Features

- **Universal Tool Schema**: Write tools once, use with any provider
- **MCP Integration**: Connect to external Model Context Protocol servers
- **Secure Credential Management**: Built-in API key and secret handling
- **Agent Workflows**: Seamless integration with orchestration system
- **Async Execution**: Concurrent tool execution for performance

## Quick Start

### 1. Create and Register a Tool

```python
from tools import ToolSchema, ToolParameter, ParameterType, Tool, ToolExecutor
from tools import register_tool_globally

class WeatherExecutor(ToolExecutor):
    async def execute(self, tool_call):
        # Your tool logic here
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result={"weather": "sunny", "temp": "22°C"}
        )

# Define tool schema
weather_schema = ToolSchema(
    name="get_weather",
    description="Get current weather for a location",
    parameters=ToolParameter(
        type=ParameterType.OBJECT,
        properties={
            "location": ToolParameter(
                type=ParameterType.STRING,
                description="City name"
            )
        },
        required=["location"]
    )
)

# Create and register tool
weather_tool = Tool(schema=weather_schema, executor=WeatherExecutor())
await register_tool_globally(weather_tool, category="utilities")
```

### 2. Use in Agent Workflows

```python
from tools import get_global_tool_service

# Get tool service and execute tools
tool_service = await get_global_tool_service()
config = await tool_service.get_tool_calling_config(
    provider="openai",
    tool_names=["get_weather"]
)

# Use in workflow JSON
workflow_step = {
    "type": "tool_call",
    "config": {
        "provider": "openai",
        "model": "gpt-4",
        "tools": ["get_weather"],
        "tool_choice": "auto"
    }
}
```

### 3. MCP Integration (External Tools)

```python
from tools import initialize_mcp_servers_from_config, register_mcp_tools_globally
from credentials import CredentialManager

# Load MCP servers with credentials
creds = CredentialManager()  # Loads from .env
await initialize_mcp_servers_from_config(credential_manager=creds)
await register_mcp_tools_globally()

# Now use filesystem, GitHub, web search tools etc.
```

## Tool Schema Format

Define tools using a universal schema that works with all providers:

```python
ToolSchema(
    name="get_weather",
    description="Get current weather for a location",
    parameters=ToolParameter(
        type=ParameterType.OBJECT,
        properties={
            "location": ToolParameter(
                type=ParameterType.STRING,
                description="City name"
            )
        },
        required=["location"]
    )
)
```

**Parameter Types**: STRING, INTEGER, NUMBER, BOOLEAN, ARRAY, OBJECT

## MCP Integration

Connect to external Model Context Protocol servers for additional tools:

### Install MCP Servers

```bash
# Common MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-brave-search
```

### Configure with Credentials

Create `mcp_servers.json`:

```json
{
  "servers": {
    "github": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env_credentials": {"GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"},
      "required_credentials": ["GITHUB_TOKEN"]
    },
    "filesystem": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
      "args": ["/path/to/directory"]
    }
  }
}
```

### Use in Code

```python
# Auto-discovers and loads configured servers
await initialize_mcp_servers_from_config(credential_manager=creds)
await register_mcp_tools_globally()

# Now agents can use GitHub, filesystem, search tools etc.
```

## Secure API Tools

Build tools that securely handle API keys and secrets:

### Basic Pattern

```python
from tools import APIToolExecutor
from credentials import CredentialManager

class GitHubTool(APIToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            base_url="https://api.github.com",
            api_key_name="GITHUB_TOKEN",
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call):
        # API key automatically added to headers
        response = await self.make_api_request("repos/owner/repo", "GET")
        return ToolResult(...)
```

### Credential Management

```python
# Load from environment/.env
creds = CredentialManager()

# Or provide explicit credentials
creds = CredentialManager({
    "GITHUB_TOKEN": "your-token",
    "OPENAI_API_KEY": "your-key"
}, load_env=False)

# Use with tool service
tool_service = ToolService(credential_manager=creds)
```

### Common Credential Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | GPT API access |
| Anthropic | `ANTHROPIC_API_KEY` | Claude API access |
| GitHub | `GITHUB_TOKEN` | GitHub API access |
| Slack | `SLACK_WEBHOOK_URL` | Slack notifications |

## Common Patterns

### Tool Registration Pattern

```python
from tools import register_tool_globally

# Create tool once
tool = Tool(schema=your_schema, executor=YourExecutor())
await register_tool_globally(tool, category="utilities")

# Use everywhere
tool_service = await get_global_tool_service()
result = await tool_service.execute_tool_call(tool_call)
```

### Secure API Integration

```python
from tools import APIToolExecutor
from credentials import CredentialManager

class YourAPITool(APIToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            base_url="https://api.example.com",
            api_key_name="YOUR_API_KEY",
            credential_manager=credential_manager
        )
```

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| `tool_types.py` | Universal tool schema and interfaces |
| `tool_service.py` | Main orchestration service |
| `tool_registry.py` | Tool management and discovery |
| `provider_adapters.py` | Provider-specific formatting |
| `mcp_server_manager.py` | MCP server lifecycle management |
| `secure_tool_executor.py` | Secure credential handling |

### Provider Support

| Provider | Tool Choice | Streaming | Notes |
|----------|-------------|-----------|-------|
| OpenAI | auto, none, specific | ✓ | Full compatibility |
| Anthropic | auto, any, none, specific | ✓ | Claude integration |
| Gemini | AUTO, ANY, NONE | ✓ | Function declarations |
| Ollama | auto, none, specific | ✓ | Local models |

### Integration Flow

1. **Tool Definition**: Create universal tool schema
2. **Registration**: Add to global registry with categories
3. **Discovery**: Service finds tools by name/category
4. **Adaptation**: Convert to provider-specific format
5. **Execution**: Async execution with error handling
6. **Results**: Structured output with tool calls and results