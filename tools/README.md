# Tool Service for AI Lego Bricks

A comprehensive tool calling service that provides a unified interface for tool registration, execution, and provider abstraction across different LLM providers.

## Features

- **Universal Tool Schema**: Define tools once, use with any provider
- **Provider Abstraction**: Support for OpenAI, Anthropic, Google Gemini, and Ollama
- **Tool Registry**: Centralized tool management with categories and search
- **Async Execution**: Concurrent tool execution for better performance
- **Workflow Integration**: Seamless integration with agent orchestration

## Architecture

### Core Components

1. **Tool Types** (`tool_types.py`): Core data structures and interfaces
2. **Tool Registry** (`tool_registry.py`): Central tool management
3. **Provider Adapters** (`provider_adapters.py`): Provider-specific formatting
4. **Tool Service** (`tool_service.py`): Main service orchestrating everything

### Provider Support

| Provider | Tool Format | Choice Options | Streaming |
|----------|-------------|----------------|-----------|
| OpenAI | `tools` parameter | auto, none, specific | Yes |
| Anthropic | `tools` parameter | auto, any, none, specific | Yes |
| Gemini | `functionDeclarations` | AUTO, ANY, NONE | Yes |
| Ollama | `tools` parameter | auto, none, specific | Yes |

## Quick Start

### 1. Register a Tool

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

### 2. Use in Workflows

```json
{
  "id": "weather_assistant",
  "type": "tool_call",
  "config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "tools": ["get_weather"],
    "tool_choice": "auto",
    "max_iterations": 3,
    "auto_execute": true,
    "prompt": "You are a weather assistant with access to weather tools."
  },
  "inputs": {
    "message": "What's the weather in London?"
  }
}
```

### 3. Use Tool Service Directly

```python
from tools import get_global_tool_service

# Get tool service
tool_service = await get_global_tool_service()

# Get configuration for a provider
config = await tool_service.get_tool_calling_config(
    provider="openai",
    tool_names=["get_weather", "calculate"]
)

# Execute tools from LLM response
tool_calls = await tool_service.parse_tool_calls_from_response(
    provider="openai", 
    response=llm_response
)
results = await tool_service.execute_tool_calls(tool_calls)
```

## Tool Schema Format

Tools are defined using a universal schema that automatically converts to provider-specific formats:

```python
ToolSchema(
    name="function_name",
    description="What the function does",
    parameters=ToolParameter(
        type=ParameterType.OBJECT,
        properties={
            "param1": ToolParameter(
                type=ParameterType.STRING,
                description="Parameter description"
            ),
            "param2": ToolParameter(
                type=ParameterType.INTEGER,
                description="Number parameter"
            )
        },
        required=["param1"]
    )
)
```

### Supported Parameter Types

- `STRING`: Text values
- `INTEGER`: Whole numbers
- `NUMBER`: Floating point numbers
- `BOOLEAN`: True/false values
- `ARRAY`: Lists of values
- `OBJECT`: Nested objects

## Provider-Specific Behavior

### OpenAI
- Uses `tools` parameter with `function` type
- Supports `tool_choice`: "auto", "none", or specific function
- Returns tool calls in `choices[0].message.tool_calls`

### Anthropic
- Uses `tools` parameter in Messages API
- Supports `tool_choice`: "auto", "any", "none", or specific tool
- Returns tool calls in `content` blocks with `type: "tool_use"`

### Google Gemini
- Uses `functionDeclarations` in `tools` parameter
- Supports `function_calling_config.mode`: "AUTO", "ANY", "NONE"
- Returns tool calls in `candidates[0].content.parts[0].functionCall`

### Ollama
- Uses OpenAI-compatible `tools` parameter
- Supports `tool_choice`: "auto", "none", or specific function
- Returns tool calls in `message.tool_calls`

## Tool Choice Options

- **auto**: Model decides whether to use tools
- **any**: Model must use at least one tool
- **none**: Model cannot use tools
- **specific**: Model must use a specific tool

## Examples

### Basic Weather Tool

```python
# Run the example
python -m tools.example_tools

# Test the tool service
python -m tools.example_workflow
```

### Workflow Integration

See `example_workflow.py` for complete workflow examples demonstrating:
- Single provider tool calling
- Multi-provider comparison
- Tool categories and search
- Error handling

## Error Handling

The service provides comprehensive error handling:

```python
# Tool execution errors
result = await tool_service.execute_tool_calls(tool_calls)
if result.error:
    print(f"Tool failed: {result.error}")

# Provider format errors
try:
    config = await tool_service.get_tool_calling_config("unknown_provider")
except ValueError as e:
    print(f"Provider error: {e}")
```

## Performance Considerations

- **Concurrent Execution**: Tool calls are executed in parallel
- **Registry Caching**: Tool schemas are cached for fast lookup
- **Provider Optimization**: Each adapter is optimized for its provider
- **Async Operations**: All operations are async for better performance

## Integration with Agent Orchestration

The tool service integrates seamlessly with the agent orchestration system:

1. **New Step Type**: `TOOL_CALL` step type added to orchestrator
2. **Configuration**: Uses `ToolCallConfig` for step configuration
3. **Execution**: Automatic tool calling with conversation management
4. **Results**: Structured output with tool calls and results

## Best Practices

1. **Tool Design**: Make tools focused and single-purpose
2. **Error Handling**: Always handle tool execution errors gracefully
3. **Categories**: Organize tools into logical categories
4. **Descriptions**: Write clear, detailed tool descriptions
5. **Validation**: Validate tool parameters before execution
6. **Security**: Sanitize inputs and limit tool capabilities
7. **Testing**: Test tools with different providers and scenarios

## Secure Tool Development (API Keys & Secrets)

The tool service integrates with AI Lego Bricks' **CredentialManager** for secure handling of API keys and secrets.

### Basic Secure Tool Pattern

```python
from tools import SecureToolExecutor, APIToolExecutor
from credentials import CredentialManager

class MyAPITool(APIToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            base_url="https://api.example.com",
            api_key_name="MY_API_KEY",  # Environment variable name
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call):
        # API key is automatically included in headers
        response = await self.make_api_request("endpoint", "POST", data)
        return ToolResult(...)
```

### Credential Management Patterns

#### 1. Environment-Based (Production)
```python
# Loads from .env file automatically
creds = CredentialManager(load_env=True)
tool_service = ToolService(credential_manager=creds)
```

#### 2. Explicit Credentials (Library/Multi-tenant)
```python
# Explicit credentials, no .env interference
creds = CredentialManager({
    "OPENAI_API_KEY": user_provided_key,
    "GITHUB_TOKEN": user_github_token
}, load_env=False)

tool_service = ToolService(credential_manager=creds)
```

#### 3. Mixed Mode (Override + Environment)
```python
# Override specific keys, others from environment
creds = CredentialManager({
    "OPENAI_API_KEY": "override-key"  # Override this one
    # GITHUB_TOKEN comes from environment
}, load_env=True)
```

### Secure Tool Types

#### API Tools
```python
class GitHubTool(APIToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            base_url="https://api.github.com",
            api_key_name="GITHUB_TOKEN",
            credential_manager=credential_manager,
            additional_headers={"Accept": "application/vnd.github.v3+json"}
        )
```

#### Database Tools
```python
class SupabaseTool(DatabaseToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            connection_credentials=["SUPABASE_URL", "SUPABASE_ANON_KEY"],
            credential_manager=credential_manager
        )
    
    def get_connection_string(self):
        return self.get_connection_string(
            "postgresql://{SUPABASE_URL}?apikey={SUPABASE_ANON_KEY}"
        )
```

#### Webhook Tools
```python
class SlackTool(WebhookToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            webhook_url_key="SLACK_WEBHOOK_URL",
            secret_key="SLACK_SIGNING_SECRET",  # Optional
            credential_manager=credential_manager
        )
```

### Workflow Integration

#### Tool Call Step with Credentials
```json
{
  "id": "secure_api_call",
  "type": "tool_call",
  "config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "tools": ["github_api", "openai_chat"],
    "credential_validation": true,
    "required_credentials": ["GITHUB_TOKEN", "OPENAI_API_KEY"]
  }
}
```

#### Credential Validation
```python
# Validate credentials before workflow execution
tool_service = ToolService(credential_manager=creds)
validation = await tool_service.validate_tool_credentials()

if validation['unavailable_tools'] > 0:
    print(f"Missing credentials: {validation['missing_credentials']}")
    # Handle missing credentials
```

### Security Best Practices

#### 1. Credential Isolation
```python
# Each tenant gets isolated credentials
def create_tenant_tools(tenant_id, tenant_creds):
    creds = CredentialManager(tenant_creds, load_env=False)
    return ToolService(credential_manager=creds)

tenant_a = create_tenant_tools("tenant_a", {"API_KEY": "tenant-a-key"})
tenant_b = create_tenant_tools("tenant_b", {"API_KEY": "tenant-b-key"})
```

#### 2. Early Validation
```python
# Validate at startup, not at runtime
def validate_startup_credentials():
    creds = CredentialManager(load_env=True)
    required = ["OPENAI_API_KEY", "GITHUB_TOKEN", "SLACK_WEBHOOK_URL"]
    
    try:
        creds.validate_required_credentials(required, "Tool Service")
        return True
    except ValueError as e:
        print(f"Missing credentials: {e}")
        return False
```

#### 3. Safe Error Handling
```python
class SecureAPITool(APIToolExecutor):
    async def execute(self, tool_call):
        try:
            # Tool execution
            return ToolResult(...)
        except Exception as e:
            # Never expose API keys in error messages
            safe_error = str(e).replace(self.require_credential("API_KEY"), "[REDACTED]")
            return ToolResult(error=safe_error)
```

#### 4. Conditional Registration
```python
# Only register tools if credentials are available
async def register_available_tools():
    creds = CredentialManager(load_env=True)
    
    if creds.has_credential("OPENAI_API_KEY"):
        await register_tool_globally(openai_tool, "ai")
    
    if creds.has_credential("GITHUB_TOKEN"):
        await register_tool_globally(github_tool, "dev")
```

### Common Credential Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | GPT API access |
| Anthropic | `ANTHROPIC_API_KEY` | Claude API access |
| GitHub | `GITHUB_TOKEN` | GitHub API access |
| Slack | `SLACK_WEBHOOK_URL` | Slack notifications |
| Supabase | `SUPABASE_URL`, `SUPABASE_ANON_KEY` | Database access |
| Neo4j | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph database |

### Testing with Mock Credentials

```python
@pytest.fixture
def mock_credentials():
    return CredentialManager({
        "OPENAI_API_KEY": "test-key",
        "GITHUB_TOKEN": "test-token"
    }, load_env=False)

async def test_secure_tool(mock_credentials):
    tool_service = ToolService(credential_manager=mock_credentials)
    # Test with safe mock credentials
```

## Future Enhancements

- **Tool Composition**: Chain multiple tools together
- **Dynamic Tools**: Generate tools from API specifications
- **Tool Metrics**: Track tool usage and performance
- **Tool Versioning**: Support multiple versions of tools
- **Tool Permissions**: Role-based access control for tools
- **Credential Rotation**: Automatic API key rotation
- **Audit Logging**: Track credential usage and access