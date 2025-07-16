# HTTP Request Service

The HTTP Request Service provides a standardized, secure, and observable way to make HTTP requests across the AI Lego Bricks system. It integrates seamlessly with the agent orchestrator and credential management system.

## Features

- **Async Operations**: Built on httpx for high-performance async HTTP requests
- **Credential Integration**: Automatic resolution of authentication credentials
- **Retry Logic**: Configurable retry mechanism with exponential backoff
- **Comprehensive Error Handling**: Structured error responses with detailed information
- **Multiple Auth Types**: Support for Bearer tokens, Basic auth, and API keys
- **Agent Orchestrator Integration**: Works as a step handler in workflows
- **Observability**: Request/response logging and performance metrics

## Basic Usage

### Direct Service Usage

```python
from services.http_request_service import HttpRequestService, HttpRequestConfig, HttpMethod

# Create service instance
async with HttpRequestService() as service:
    # Simple GET request
    response = await service.get("https://api.example.com/data")
    print(f"Status: {response.status_code}")
    print(f"Data: {response.json()}")
    
    # POST request with JSON
    response = await service.post(
        "https://api.example.com/create",
        json_data={"name": "John", "email": "john@example.com"}
    )
```

### Agent Orchestrator Integration

```json
{
  "id": "fetch_user_data",
  "type": "http_request",
  "config": {
    "method": "GET",
    "auth_type": "bearer",
    "timeout": 30.0,
    "max_retries": 3
  },
  "inputs": {
    "url": "https://api.example.com/users/123",
    "headers": {
      "Accept": "application/json"
    }
  },
  "outputs": ["json", "status_code", "success"]
}
```

## Configuration

### HttpRequestConfig

```python
@dataclass
class HttpRequestConfig:
    url: str                           # Target URL (required)
    method: HttpMethod = GET           # HTTP method
    headers: Dict[str, str] = {}       # Request headers
    params: Dict[str, Any] = {}        # Query parameters
    json_data: Optional[Dict] = None   # JSON request body
    form_data: Optional[Dict] = None   # Form data
    data: Optional[Union[str, bytes]] = None  # Raw data
    timeout: float = 30.0              # Request timeout
    follow_redirects: bool = True      # Follow redirects
    verify_ssl: bool = True            # SSL verification
    auth_type: Optional[str] = None    # Authentication type
    auth_credentials: Optional[Dict] = None  # Auth credentials
```

### Authentication Types

#### Bearer Token
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="bearer",
    auth_credentials={"token": "your-token-here"}
)
```

#### Basic Authentication
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="basic",
    auth_credentials={"username": "user", "password": "pass"}
)
```

#### API Key
```python
config = HttpRequestConfig(
    url="https://api.example.com/data",
    auth_type="api_key",
    auth_credentials={
        "api_key": "your-key-here",
        "header_name": "X-API-Key"  # Optional, defaults to X-API-Key
    }
)
```

## Agent Orchestrator Step Handler

The HTTP request service integrates with the agent orchestrator as a step handler, allowing you to make HTTP requests as part of workflows.

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
    "auth_credentials": {
      "token": "static-token-here"
    },
    "include_content": true,
    "include_text": true,
    "follow_redirects": true,
    "verify_ssl": true
  },
  "inputs": {
    "url": "https://api.example.com/process",
    "json_data": {
      "data": "some value"
    },
    "headers": {
      "Content-Type": "application/json",
      "User-Agent": "AI-Lego-Bricks/1.0"
    },
    "params": {
      "version": "v1"
    }
  }
}
```

### Input Parameters

- `url` (required): The target URL for the request
- `headers` (optional): HTTP headers to include
- `params` (optional): Query parameters
- `json_data` (optional): JSON request body
- `form_data` (optional): Form data for POST requests
- `data` (optional): Raw request body
- `auth_credentials` (optional): Runtime authentication credentials

### Output Format

```json
{
  "success": true,
  "status_code": 200,
  "headers": {
    "Content-Type": "application/json",
    "Server": "nginx/1.18.0"
  },
  "url": "https://api.example.com/process",
  "method": "POST",
  "elapsed_time": 0.245,
  "content_length": 156,
  "is_client_error": false,
  "is_server_error": false,
  "content": "{\"result\": \"success\", \"id\": 123}",
  "text": "{\"result\": \"success\", \"id\": 123}",
  "json": {
    "result": "success",
    "id": 123
  },
  "metadata": {
    "request_config": {
      "url": "https://api.example.com/process",
      "method": "POST",
      "headers": {...},
      "params": {...},
      "auth_type": "bearer",
      "timeout": 60.0,
      "follow_redirects": true,
      "verify_ssl": true
    },
    "response_info": {
      "content_type": "application/json",
      "content_encoding": null,
      "server": "nginx/1.18.0"
    }
  }
}
```

## Error Handling

### HTTP Errors

```json
{
  "success": false,
  "status_code": 404,
  "error": "Not Found",
  "is_client_error": true,
  "is_server_error": false,
  "url": "https://api.example.com/nonexistent",
  "method": "GET",
  "elapsed_time": 0.123,
  "metadata": {...}
}
```

### Network Errors

```json
{
  "success": false,
  "error": "Connection timeout",
  "error_type": "ConnectTimeout",
  "status_code": null,
  "url": "https://api.example.com/slow",
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

## Credential Management

The HTTP request service integrates with the credential management system to automatically resolve authentication credentials from environment variables.

### Environment Variables

For Bearer tokens:
- `TOKEN`
- `ACCESS_TOKEN`
- `BEARER_TOKEN`
- `AUTH_TOKEN`

For Basic auth:
- `USERNAME`
- `PASSWORD`

For API keys:
- `API_KEY`

### Example with Credential Manager

```python
from credentials.credential_manager import CredentialManager
from services.http_request_service import HttpRequestService, HttpRequestConfig

# Create credential manager
cred_manager = CredentialManager()

# Create service with credential manager
async with HttpRequestService(credential_manager=cred_manager) as service:
    config = HttpRequestConfig(
        url="https://api.example.com/data",
        auth_type="bearer"  # Token will be resolved from environment
    )
    response = await service.request(config)
```

## Retry Logic

The service includes built-in retry logic with exponential backoff:

```python
service = HttpRequestService(
    max_retries=3,        # Maximum number of retry attempts
    backoff_factor=1.0    # Backoff factor (delay = backoff_factor * (2 ** attempt))
)
```

## Performance Considerations

- Uses httpx for high-performance async HTTP requests
- Connection pooling for efficient resource usage
- Configurable timeouts to prevent hanging requests
- Retry logic with exponential backoff to handle transient failures
- Structured logging for observability and debugging

## Integration Examples

### API Data Fetching Workflow

```json
{
  "name": "Fetch User Profile",
  "steps": [
    {
      "id": "get_user",
      "type": "http_request",
      "config": {
        "method": "GET",
        "auth_type": "bearer"
      },
      "inputs": {
        "url": "https://api.example.com/users/123"
      },
      "outputs": ["json", "success"]
    },
    {
      "id": "process_user",
      "type": "llm_chat",
      "inputs": {
        "message": "Analyze this user profile: {from_step: 'get_user', field: 'json'}"
      },
      "condition": {
        "field": "get_user.success",
        "operator": "equals",
        "value": true
      }
    }
  ]
}
```

### Data Submission Workflow

```json
{
  "name": "Submit Form Data",
  "steps": [
    {
      "id": "submit_form",
      "type": "http_request",
      "config": {
        "method": "POST",
        "auth_type": "api_key",
        "max_retries": 5
      },
      "inputs": {
        "url": "https://api.example.com/forms",
        "json_data": {
          "name": "$user_name",
          "email": "$user_email"
        }
      },
      "outputs": ["success", "status_code", "json"]
    },
    {
      "id": "handle_result",
      "type": "condition",
      "inputs": {
        "condition": "{from_step: 'submit_form', field: 'success'}"
      },
      "routes": {
        "true": "success_handler",
        "false": "error_handler"
      }
    }
  ]
}
```

## Best Practices

1. **Always use async context managers** for proper resource cleanup
2. **Configure appropriate timeouts** based on expected response times
3. **Use credential manager** for secure authentication handling
4. **Implement proper error handling** in workflows
5. **Log request/response details** for debugging and monitoring
6. **Use retry logic** for transient failures
7. **Structure JSON responses** for easy data extraction in workflows

## Testing

The HTTP request service includes comprehensive tests covering:

- Configuration validation
- Authentication methods
- Request/response handling
- Error scenarios
- Agent orchestrator integration
- Credential management
- Retry logic

Run tests with:
```bash
python -m pytest tests/test_http_request_service.py -v
python -m pytest tests/test_http_step_handler.py -v
```