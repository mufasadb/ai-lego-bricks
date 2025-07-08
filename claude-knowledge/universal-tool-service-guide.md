# 🔧 Universal Tool Service - Complete Implementation Guide

## Overview

The Universal Tool Service is a comprehensive system for registering, managing, and executing tools across different LLM providers. It provides a unified interface that abstracts away provider-specific differences for OpenAI, Anthropic, Google Gemini, and Ollama.

## Architecture

### Core Components

1. **Tool Types** (`tools/tool_types.py`)
   - Universal schema that converts to any provider format
   - Type-safe parameter definitions
   - Execution interfaces and result handling

2. **Tool Registry** (`tools/tool_registry.py`)
   - Centralized tool management with categories
   - Async tool execution with concurrent support
   - Tool search and discovery capabilities

3. **Provider Adapters** (`tools/provider_adapters.py`)
   - OpenAI, Anthropic, Google Gemini, and Ollama adapters
   - Provider-specific formatting and parsing
   - Unified interface across all providers

4. **Tool Service** (`tools/tool_service.py`)
   - Main orchestration service
   - Complete workflow management
   - Provider abstraction layer
   - Credential management integration

5. **Secure Tool Executors** (`tools/secure_tool_executor.py`)
   - Base classes for tools requiring API keys
   - Integration with CredentialManager
   - Safe credential handling patterns

## Key Features

### ✅ Universal Schema
- Define tools once with a universal schema
- Automatically converts to provider-specific formats
- Type-safe parameter validation
- Support for all JSON Schema types

### ✅ Provider Abstraction
- Single codebase works with OpenAI, Anthropic, Gemini, Ollama
- Provider-specific optimizations handled automatically
- Consistent API across all providers

### ✅ Secure Credential Management
- Integrates with existing CredentialManager
- Early credential validation
- Safe error handling (never exposes API keys)
- Multi-tenant isolation support

### ✅ Workflow Integration
- New `tool_call` step type in agent orchestration
- Automatic tool calling and result processing
- Rich conversation tracking
- Error handling and retry logic

### ✅ Performance Optimizations
- Concurrent tool execution
- Registry caching for fast lookup
- Provider-specific optimizations
- Async/await throughout

## Usage Patterns

### 1. Basic Tool Registration

```python
from tools import ToolSchema, ToolParameter, ParameterType, Tool, ToolExecutor
from tools import register_tool_globally

class WeatherExecutor(ToolExecutor):
    async def execute(self, tool_call):
        location = tool_call.parameters.get("location")
        # Your weather API logic here
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result={"weather": "sunny", "temp": "22°C", "location": location}
        )

# Define universal schema
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

# Register globally
weather_tool = Tool(schema=weather_schema, executor=WeatherExecutor())
await register_tool_globally(weather_tool, category="utilities")
```

### 2. Secure API Tools

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
        repo = tool_call.parameters.get("repo")
        # API key automatically included in headers
        response = await self.make_api_request(f"repos/{repo}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result={
                "repo": repo,
                "stars": response["stargazers_count"],
                "description": response["description"]
            }
        )
```

### 3. Workflow Integration

```json
{
  "id": "tool_assistant",
  "type": "tool_call",
  "description": "Use tools to handle user requests",
  "config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "tools": ["get_weather", "calculate"],
    "tool_choice": "auto",
    "max_iterations": 3,
    "auto_execute": true,
    "prompt": "You are a helpful assistant with access to tools.",
    "temperature": 0.1
  },
  "inputs": {
    "message": "What's the weather in London and calculate 15 * 8?"
  }
}
```

## Provider Support Matrix

| Provider | Tool Format | Choice Options | Streaming | Status |
|----------|-------------|----------------|-----------|---------|
| **OpenAI** | `tools` parameter | auto, none, specific | ✅ | ✅ Full Support |
| **Anthropic** | `tools` parameter | auto, any, none, specific | ✅ | ✅ Full Support |
| **Google Gemini** | `functionDeclarations` | AUTO, ANY, NONE | ✅ | ✅ Full Support |
| **Ollama** | `tools` parameter | auto, none, specific | ✅ | ✅ Full Support |

## Credential Management

### Security Patterns

#### 1. Environment-Based (Production)
```python
creds = CredentialManager(load_env=True)
tool_service = ToolService(credential_manager=creds)
```

#### 2. Explicit Credentials (Library/Multi-tenant)
```python
creds = CredentialManager({
    "OPENAI_API_KEY": user_provided_key,
    "GITHUB_TOKEN": user_github_token
}, load_env=False)
tool_service = ToolService(credential_manager=creds)
```

#### 3. Credential Validation
```python
validation = await tool_service.validate_tool_credentials()
if validation['unavailable_tools'] > 0:
    print(f"Missing credentials: {validation['missing_credentials']}")
```

### Secure Tool Types

1. **APIToolExecutor**: REST APIs with Bearer token authentication
2. **DatabaseToolExecutor**: Database connections with connection strings  
3. **WebhookToolExecutor**: Webhooks with optional signature verification
4. **SecureToolExecutor**: Base class with credential validation

## Implementation Details

### Tool Schema Conversion

The universal schema automatically converts to provider formats:

```python
# Universal schema
schema = ToolSchema(name="get_weather", description="...", parameters=...)

# Automatically converts to:
# OpenAI: {"type": "function", "function": {...}}
# Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
# Gemini: {"name": "...", "description": "...", "parameters": {...}}
# Ollama: {"type": "function", "function": {...}}
```

### Error Handling

- **Credential errors**: Detected at registration time, not runtime
- **API errors**: Safe error messages that never expose credentials
- **Tool execution errors**: Graceful degradation with detailed error reporting
- **Provider errors**: Automatic retry logic and fallback mechanisms

### Performance Features

- **Concurrent execution**: Multiple tools execute in parallel
- **Registry caching**: Fast tool lookup and discovery
- **Provider optimization**: Each adapter optimized for its provider
- **Async/await**: Non-blocking operations throughout

## Agent Orchestration Integration

### New Step Type: `tool_call`

```python
# Added to StepType enum
class StepType(str, Enum):
    # ... existing types
    TOOL_CALL = "tool_call"

# Configuration model
class ToolCallConfig(BaseModel):
    provider: str
    model: str
    tools: Optional[List[str]] = None
    tool_category: Optional[str] = None
    tool_choice: str = "auto"
    max_iterations: int = 5
    auto_execute: bool = True
    prompt: Optional[str] = None
    temperature: float = 0.1
```

### Step Handler Implementation

- Sync wrapper around async tool execution
- Full conversation tracking
- Iterative tool calling support
- Rich result formatting

## Testing and Examples

### Example Tools Provided

1. **Calculator**: Mathematical operations with safe evaluation
2. **Weather**: Mock weather data for demonstrations
3. **File Manager**: Mock file system operations
4. **Web Search**: Mock search results
5. **OpenAI Chat**: Real OpenAI API integration
6. **GitHub API**: Repository and issue management
7. **Slack Webhook**: Message sending with signatures
8. **Supabase Query**: Database operations

### Testing Patterns

```python
# Mock credentials for testing
@pytest.fixture
def mock_credentials():
    return CredentialManager({
        "OPENAI_API_KEY": "test-key",
        "GITHUB_TOKEN": "test-token"
    }, load_env=False)

async def test_tool_execution(mock_credentials):
    tool_service = ToolService(credential_manager=mock_credentials)
    # Test with safe mock credentials
```

## Migration and Adoption

### For Existing Projects

1. **No breaking changes**: Existing LLM functionality continues to work
2. **Gradual adoption**: Add tools incrementally
3. **Backward compatibility**: All existing step types remain unchanged

### For New Projects

1. **Start with tool_call steps**: Leverage the full power from day one
2. **Use secure patterns**: Implement proper credential management
3. **Follow examples**: Use provided examples as templates

## Security Best Practices

1. **Early validation**: Check credentials at startup, not runtime
2. **Credential isolation**: Each tenant/user has separate credential scope
3. **Safe error handling**: Never expose API keys in logs or error messages
4. **Conditional registration**: Only register tools if credentials are available
5. **Audit logging**: Track tool usage and credential access

## Future Enhancements

- **Tool composition**: Chain multiple tools together
- **Dynamic tools**: Generate tools from OpenAPI specifications
- **Tool metrics**: Track usage, performance, and success rates
- **Tool versioning**: Support multiple versions of the same tool
- **Tool permissions**: Role-based access control
- **Credential rotation**: Automatic API key rotation
- **Enhanced validation**: More sophisticated parameter validation

## Files and Locations

- **Main README**: Updated with tool service section
- **Detailed README**: `TOOLS_README.md` with comprehensive documentation
- **Implementation**: `tools/` directory with complete implementation
- **Examples**: `tools/example_tools.py` and `tools/secure_example_tools.py`
- **Demo**: `tools/secure_tools_demo.py` for testing and learning
- **Integration**: Agent orchestration updated with `tool_call` step type

This universal tool service represents a major advancement in the AI Lego Bricks ecosystem, providing enterprise-grade tool management with security, performance, and ease of use.