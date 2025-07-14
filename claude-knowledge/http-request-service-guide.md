# HTTP Request Service Integration Guide

## Overview

The HTTP Request Service is a dedicated service within the AI Lego Bricks system that provides standardized HTTP request capabilities for the agent orchestrator. This service addresses the need for consistent, secure, and observable HTTP interactions across all workflows.

## Why Add a Dedicated HTTP Request Service?

### Current State Analysis
- **Scattered HTTP handling**: Different services using various HTTP libraries (httpx, aiohttp, requests)
- **Inconsistent patterns**: No standardized approach to authentication, error handling, or retries
- **Limited observability**: Difficult to monitor and debug HTTP requests across the system
- **Security concerns**: No centralized credential management for API interactions

### Benefits of Dedicated Service
- **Standardization**: Consistent HTTP handling across all services
- **Security**: Centralized credential management and secure authentication
- **Observability**: Unified logging, metrics, and debugging for all HTTP requests
- **Reliability**: Built-in retry logic with exponential backoff
- **Architecture fit**: Perfect alignment with the existing building block philosophy

## Architecture Integration

### Service Layer
The HTTP Request Service fits seamlessly into the existing service architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent Orchestrator                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ LLM Service │  │ Memory Svc  │  │ HTTP Request│         │
│  │             │  │             │  │ Service     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                   Step Handlers                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ LLM Steps   │  │ Memory Step │  │ HTTP Step   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                 Credential Manager                          │
└─────────────────────────────────────────────────────────────┘
```

### Step Handler Integration
The service integrates as a new step type (`http_request`) in the agent orchestrator:

```python
# In step_handlers.py
self.handlers[StepType.HTTP_REQUEST] = self._handle_http_request
```

### Service Registration
Registered in the orchestrator's service initialization:

```python
# In orchestrator.py
if create_http_request_service:
    self._services["http_request"] = create_http_request_service(
        credential_manager=credential_manager
    )
```

## Implementation Details

### Core Components

#### 1. HttpRequestService Class
```python
class HttpRequestService:
    def __init__(self, credential_manager: Optional[CredentialManager] = None,
                 default_timeout: float = 30.0,
                 max_retries: int = 3,
                 backoff_factor: float = 1.0)
```

**Key Features:**
- Async/await support with httpx backend
- Credential manager integration
- Configurable retry logic with exponential backoff
- Comprehensive error handling
- Context manager support for resource cleanup

#### 2. HttpRequestConfig Class
```python
@dataclass
class HttpRequestConfig:
    url: str
    method: HttpMethod = HttpMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    json_data: Optional[Dict[str, Any]] = None
    form_data: Optional[Dict[str, Any]] = None
    data: Optional[Union[str, bytes]] = None
    timeout: float = 30.0
    follow_redirects: bool = True
    verify_ssl: bool = True
    auth_type: Optional[str] = None
    auth_credentials: Optional[Dict[str, str]] = None
```

**Configuration Validation:**
- URL format validation
- Mutually exclusive data fields
- Type safety with enums and dataclasses

#### 3. HttpResponse Class
```python
@dataclass
class HttpResponse:
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    url: str
    method: str
    elapsed_time: float
```

**Response Properties:**
- `is_success`: 2xx status codes
- `is_client_error`: 4xx status codes
- `is_server_error`: 5xx status codes
- `json()`: Parse JSON response
- `raise_for_status()`: Raise exception for errors

### Authentication Integration

#### Credential Resolution Chain
1. **Explicit credentials** (highest priority)
2. **Environment variables** (via CredentialManager)
3. **Default values** (lowest priority)

#### Supported Authentication Types

**Bearer Token:**
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="bearer",
    auth_credentials={"token": "your-token"}
)
```

**Basic Authentication:**
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="basic",
    auth_credentials={"username": "user", "password": "pass"}
)
```

**API Key:**
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="api_key",
    auth_credentials={"api_key": "key", "header_name": "X-API-Key"}
)
```

### Error Handling Strategy

#### Retry Logic
- **Exponential backoff**: `delay = backoff_factor * (2 ** attempt)`
- **Configurable attempts**: Default 3 retries
- **Transient error handling**: Network timeouts, connection errors
- **Non-retriable errors**: 4xx client errors (except 429)

#### Error Response Format
```python
{
    "success": False,
    "error": "Connection timeout",
    "error_type": "ConnectTimeout",
    "status_code": None,
    "url": "https://api.example.com/data",
    "method": "GET",
    "elapsed_time": 0.0,
    "metadata": {
        "error_details": {
            "error": "Connection timeout",
            "error_type": "ConnectTimeout"
        }
    }
}
```

## Agent Orchestrator Integration

### Step Configuration
```json
{
  "id": "api_request",
  "type": "http_request",
  "config": {
    "method": "POST",
    "timeout": 60.0,
    "max_retries": 5,
    "backoff_factor": 2.0,
    "auth_type": "bearer",
    "include_content": true,
    "include_text": true
  },
  "inputs": {
    "url": "https://api.example.com/process",
    "json_data": {"key": "value"},
    "headers": {"Content-Type": "application/json"}
  },
  "outputs": ["json", "status_code", "success"]
}
```

### Dynamic Input Resolution
The step handler supports dynamic input resolution from previous steps:

```json
{
  "inputs": {
    "url": "https://api.example.com/users/{from_step: 'get_user_id', field: 'user_id'}",
    "json_data": {
      "name": "{from_step: 'user_input', field: 'name'}",
      "email": "{from_step: 'user_input', field: 'email'}"
    }
  }
}
```

### Conditional Routing
HTTP responses can drive workflow routing:

```json
{
  "routes": {
    "200": "success_handler",
    "404": "not_found_handler",
    "500": "error_handler",
    "default": "fallback_handler"
  }
}
```

## Common Use Cases

### 1. API Data Fetching
```json
{
  "id": "fetch_user_data",
  "type": "http_request",
  "config": {
    "method": "GET",
    "auth_type": "bearer",
    "timeout": 30.0
  },
  "inputs": {
    "url": "https://api.example.com/users/123",
    "headers": {"Accept": "application/json"}
  },
  "outputs": ["json", "status_code", "success"]
}
```

### 2. Form Submission
```json
{
  "id": "submit_form",
  "type": "http_request",
  "config": {
    "method": "POST",
    "timeout": 30.0
  },
  "inputs": {
    "url": "https://api.example.com/forms",
    "form_data": {
      "name": "John Doe",
      "email": "john@example.com"
    }
  },
  "outputs": ["success", "status_code", "json"]
}
```

### 3. JSON API Integration
```json
{
  "id": "create_resource",
  "type": "http_request",
  "config": {
    "method": "POST",
    "auth_type": "api_key",
    "max_retries": 3
  },
  "inputs": {
    "url": "https://api.example.com/resources",
    "json_data": {
      "name": "New Resource",
      "type": "document"
    },
    "headers": {"Content-Type": "application/json"}
  },
  "outputs": ["json", "status_code", "success"]
}
```

## Testing Strategy

### Unit Tests
- **Configuration validation**: URL format, data exclusivity
- **Authentication**: Bearer, Basic, API key methods
- **Response handling**: Success, error, JSON parsing
- **Retry logic**: Exponential backoff, max attempts

### Integration Tests
- **Step handler registration**: Verify handler availability
- **Workflow execution**: End-to-end request processing
- **Error scenarios**: Network failures, HTTP errors
- **Credential resolution**: Environment variable integration

### Test Coverage
- **40 comprehensive tests** covering all functionality
- **Mock-based testing** for reliable, fast execution
- **Edge case handling** for robust error scenarios
- **Async/await patterns** for proper async testing

## Performance Considerations

### Async Architecture
- **Non-blocking I/O**: httpx async client for high concurrency
- **Connection pooling**: Efficient resource usage
- **Proper cleanup**: Context manager support

### Resource Management
- **Timeout configuration**: Prevent hanging requests
- **Connection limits**: Configurable per service
- **Memory efficiency**: Streaming for large responses

### Monitoring & Observability
- **Request/response logging**: Structured logging with metadata
- **Performance metrics**: Elapsed time, retry counts
- **Error tracking**: Detailed error information and stack traces

## Security Best Practices

### Credential Management
- **Never hardcode credentials** in workflow configurations
- **Use CredentialManager** for environment variable resolution
- **Credential isolation** per service instance

### SSL/TLS
- **Verify SSL certificates** by default
- **Configurable verification** for development environments
- **Secure connection handling**

### Input Validation
- **URL validation** with proper parsing
- **Header sanitization** to prevent injection
- **Data type validation** with Pydantic models

## Migration and Adoption

### Existing Code Migration
1. **Identify HTTP calls** in existing services
2. **Replace with HttpRequestService** calls
3. **Update credential handling** to use CredentialManager
4. **Add proper error handling** with structured responses

### New Development
1. **Always use HttpRequestService** for HTTP requests
2. **Follow configuration patterns** from examples
3. **Implement proper error handling** in workflows
4. **Add comprehensive logging** for debugging

## Future Enhancements

### Planned Features
- **GraphQL support**: Dedicated GraphQL query handling
- **WebSocket support**: Real-time communication capabilities
- **Response caching**: Configurable caching for repeated requests
- **Rate limiting**: Built-in rate limiting for API compliance
- **Metrics collection**: Prometheus/OpenTelemetry integration

### Extension Points
- **Custom authenticators**: Plugin system for new auth methods
- **Request middleware**: Pre/post request processing
- **Response transformers**: Custom response parsing
- **Circuit breakers**: Automatic failure handling

## Conclusion

The HTTP Request Service provides a robust foundation for all HTTP interactions within the AI Lego Bricks system. By centralizing HTTP handling, we achieve:

- **Consistency** across all HTTP operations
- **Security** through centralized credential management
- **Reliability** via built-in retry mechanisms
- **Observability** through comprehensive logging
- **Maintainability** by following established patterns

This service exemplifies the building block philosophy of the AI Lego Bricks system, providing a reusable, configurable component that integrates seamlessly with the existing architecture while enabling powerful new workflow capabilities.