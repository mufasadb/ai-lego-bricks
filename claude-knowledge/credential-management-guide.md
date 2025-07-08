# 🔐 Credential Management System

## Overview

AI Lego Bricks features a comprehensive credential management system designed to solve common problems when using AI libraries:

- **Library Safety**: Prevents unwanted .env file loading when used as a dependency
- **Credential Isolation**: Different services can have different credentials
- **Multi-Tenant Support**: Support for applications serving multiple clients
- **Backward Compatibility**: Existing environment variable patterns continue to work

## Architecture

### Core Components

**CredentialManager Class** (`credentials/credential_manager.py`)
- Central credential handling with environment fallback
- Support for explicit credential injection
- Optional environment loading (library-safe)
- Validation and error handling

**Integration Points**
- All LLM clients (Ollama, Gemini, Anthropic)
- Memory services (Supabase, Neo4j)
- TTS clients (OpenAI, Google, Coqui-XTTS)
- Factory classes and convenience functions

### Key Design Principles

1. **Backward Compatibility**: Existing code continues to work without changes
2. **Explicit Over Implicit**: Clear credential sources with optional environment fallback
3. **Isolation by Default**: Each service instance can have its own credentials
4. **Library-First**: Designed for both application and library usage patterns

## Usage Patterns

### 1. Traditional Application Pattern (Default)

```python
from llm import create_text_client
from memory import create_memory_service
from tts import create_tts_service

# Uses .env file automatically (backward compatible)
llm_client = create_text_client("gemini")
memory_service = create_memory_service("supabase")
tts_service = create_tts_service("openai")
```

### 2. Library Integration Pattern

```python
from credentials import CredentialManager
from llm import create_text_client
from memory import create_memory_service

def create_ai_services(api_keys: dict):
    """Create AI services for library consumers"""
    # Explicit credentials, no .env interference
    creds = CredentialManager(api_keys, load_env=False)
    
    return {
        'llm': create_text_client("gemini", credential_manager=creds),
        'memory': create_memory_service("supabase", credential_manager=creds)
    }

# Usage in consuming application
services = create_ai_services({
    "GOOGLE_AI_STUDIO_KEY": "user-provided-key",
    "SUPABASE_URL": "user-supabase-url",
    "SUPABASE_ANON_KEY": "user-supabase-key"
})
```

### 3. Multi-Tenant Application Pattern

```python
from credentials import CredentialManager

def create_tenant_services(tenant_id: str, tenant_creds: dict):
    """Create isolated services for each tenant"""
    cred_manager = CredentialManager(tenant_creds, load_env=False)
    
    return {
        'tenant_id': tenant_id,
        'llm': create_text_client("gemini", credential_manager=cred_manager),
        'memory': create_memory_service("supabase", credential_manager=cred_manager)
    }

# Different tenants with different credentials
tenant_a = create_tenant_services("tenant_a", {
    "GOOGLE_AI_STUDIO_KEY": "tenant-a-key",
    "SUPABASE_URL": "tenant-a-db-url"
})

tenant_b = create_tenant_services("tenant_b", {
    "GOOGLE_AI_STUDIO_KEY": "tenant-b-key", 
    "SUPABASE_URL": "tenant-b-db-url"
})
```

### 4. Mixed Environment Pattern

```python
from credentials import CredentialManager

# Some credentials explicit, some from environment
mixed_creds = CredentialManager({
    "GOOGLE_AI_STUDIO_KEY": "explicit-override-key"
    # ANTHROPIC_API_KEY will come from environment
}, load_env=True)

gemini_client = create_text_client("gemini", credential_manager=mixed_creds)
anthropic_client = create_text_client("anthropic", credential_manager=mixed_creds)
```

## CredentialManager API

### Constructor

```python
CredentialManager(credentials=None, load_env=True)
```

**Parameters:**
- `credentials` (Dict[str, str], optional): Explicit credentials dictionary
- `load_env` (bool, default=True): Whether to load .env file automatically

### Core Methods

#### get_credential(key: str, default=None) -> Optional[str]
Get credential with fallback chain: explicit → environment → default

```python
cred_manager = CredentialManager({"API_KEY": "explicit"}, load_env=True)

# Returns "explicit" (from constructor)
api_key = cred_manager.get_credential("API_KEY")

# Returns environment value or "default"
other_key = cred_manager.get_credential("OTHER_KEY", "default")
```

#### require_credential(key: str, service_name: str) -> str
Get required credential with validation (raises ValueError if missing)

```python
try:
    api_key = cred_manager.require_credential("GOOGLE_AI_STUDIO_KEY", "Gemini")
except ValueError as e:
    print(f"Missing credential: {e}")
```

#### has_credential(key: str) -> bool
Check if credential exists

```python
if cred_manager.has_credential("OPENAI_API_KEY"):
    tts_service = create_tts_service("openai", credential_manager=cred_manager)
```

#### validate_required_credentials(keys: list, service_name: str) -> Dict[str, str]
Validate multiple credentials at once

```python
try:
    creds = cred_manager.validate_required_credentials([
        "SUPABASE_URL", 
        "SUPABASE_ANON_KEY"
    ], "Supabase Memory Service")
    print("All Supabase credentials available!")
except ValueError as e:
    print(f"Missing credentials: {e}")
```

## Service Integration

All AI Lego Bricks services now accept an optional `credential_manager` parameter:

### LLM Services

```python
from llm.llm_factory import LLMClientFactory, create_text_client
from credentials import CredentialManager

# Factory method
creds = CredentialManager({"GOOGLE_AI_STUDIO_KEY": "key"})
client = LLMClientFactory.create_text_client(
    provider=LLMProvider.GEMINI,
    credential_manager=creds
)

# Convenience function
client = create_text_client("gemini", credential_manager=creds)
```

### Memory Services

```python
from memory import create_memory_service, MemoryServiceFactory

creds = CredentialManager({
    "SUPABASE_URL": "url",
    "SUPABASE_ANON_KEY": "key"
})

# Factory method
memory = MemoryServiceFactory.create_memory_service(
    "supabase", 
    credential_manager=creds
)

# Convenience function
memory = create_memory_service("supabase", credential_manager=creds)
```

### TTS Services

```python
from tts import create_tts_service, TTSServiceFactory

creds = CredentialManager({"OPENAI_API_KEY": "key"})

# Factory method
tts = TTSServiceFactory.create_tts_service(
    "openai",
    credential_manager=creds
)

# Convenience function
tts = create_tts_service("openai", credential_manager=creds)
```

## Common Patterns

### Startup Validation

```python
from credentials import CredentialManager

def validate_startup_credentials():
    """Validate all required credentials at application startup"""
    cred_manager = CredentialManager(load_env=True)
    
    required_services = {
        "Gemini LLM": ["GOOGLE_AI_STUDIO_KEY"],
        "Supabase Memory": ["SUPABASE_URL", "SUPABASE_ANON_KEY"],
        "OpenAI TTS": ["OPENAI_API_KEY"]
    }
    
    missing = []
    for service, keys in required_services.items():
        try:
            cred_manager.validate_required_credentials(keys, service)
            print(f"✅ {service} credentials available")
        except ValueError as e:
            missing.append(f"❌ {service}: {e}")
    
    if missing:
        print("Missing credentials:")
        for error in missing:
            print(f"  {error}")
        return False
    
    return True

# Run at startup
if not validate_startup_credentials():
    exit(1)
```

### Dynamic Service Creation

```python
def create_available_services(credentials: dict = None):
    """Create services based on available credentials"""
    cred_manager = CredentialManager(credentials, load_env=True)
    services = {}
    
    # Try to create LLM service
    if cred_manager.has_credential("GOOGLE_AI_STUDIO_KEY"):
        services['llm'] = create_text_client("gemini", credential_manager=cred_manager)
    elif cred_manager.has_credential("ANTHROPIC_API_KEY"):
        services['llm'] = create_text_client("anthropic", credential_manager=cred_manager)
    
    # Try to create memory service
    if cred_manager.has_credential("SUPABASE_URL"):
        services['memory'] = create_memory_service("supabase", credential_manager=cred_manager)
    elif cred_manager.has_credential("NEO4J_URI"):
        services['memory'] = create_memory_service("neo4j", credential_manager=cred_manager)
    
    return services
```

### Per-Request Credentials

```python
async def handle_user_request(user_id: str, request_data: dict):
    """Handle request with user-specific credentials"""
    
    # Get user's API keys from database
    user_creds = await get_user_credentials(user_id)
    
    # Create isolated credential manager
    cred_manager = CredentialManager(user_creds, load_env=False)
    
    # Create services with user's credentials
    llm_client = create_text_client("gemini", credential_manager=cred_manager)
    
    # Process request
    response = llm_client.chat(request_data['message'])
    return response
```

## Migration Guide

### From Environment-Only Pattern

**Before:**
```python
from llm import create_text_client

# Automatically loads .env
client = create_text_client("gemini")
```

**After (No Changes Required):**
```python
from llm import create_text_client

# Still works exactly the same - backward compatible!
client = create_text_client("gemini")
```

### For Library Integration

**Before (Problematic):**
```python
def create_ai_service():
    # This would load consumer's .env file
    return create_text_client("gemini")
```

**After (Library-Safe):**
```python
def create_ai_service(api_key: str):
    creds = CredentialManager({"GOOGLE_AI_STUDIO_KEY": api_key}, load_env=False)
    return create_text_client("gemini", credential_manager=creds)
```

### For Multi-Service Applications

**Before:**
```python
# All services shared same global environment
llm_client = create_text_client("gemini")
memory_service = create_memory_service("supabase")
```

**After (With Isolation):**
```python
# Each service can have different credentials
llm_creds = CredentialManager({"GOOGLE_AI_STUDIO_KEY": "llm-key"}, load_env=False)
memory_creds = CredentialManager({"SUPABASE_URL": "memory-url"}, load_env=False)

llm_client = create_text_client("gemini", credential_manager=llm_creds)
memory_service = create_memory_service("supabase", credential_manager=memory_creds)
```

## Security Considerations

### Best Practices

1. **Use `load_env=False` in library code** to prevent interfering with consumer applications
2. **Validate credentials at startup** rather than during first use
3. **Use credential isolation** for multi-tenant applications
4. **Never log or expose** credential values in error messages
5. **Rotate credentials regularly** and update credential managers accordingly

### Credential Sources (Priority Order)

1. **Explicit credentials** passed to CredentialManager constructor
2. **Environment variables** (if `load_env=True`)
3. **Default values** provided to `get_credential()`

### Error Handling

```python
from credentials import CredentialManager

def safe_service_creation():
    try:
        creds = CredentialManager(load_env=True)
        api_key = creds.require_credential("GOOGLE_AI_STUDIO_KEY", "Gemini")
        return create_text_client("gemini", credential_manager=creds)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    except Exception as e:
        print(f"Service creation failed: {e}")
        return None
```

## Testing Patterns

### Mock Credentials for Testing

```python
import pytest
from credentials import CredentialManager

@pytest.fixture
def mock_credentials():
    """Provide test credentials without loading .env"""
    return CredentialManager({
        "GOOGLE_AI_STUDIO_KEY": "test-key",
        "SUPABASE_URL": "test-url",
        "SUPABASE_ANON_KEY": "test-key"
    }, load_env=False)

def test_service_creation(mock_credentials):
    """Test service creation with mock credentials"""
    client = create_text_client("gemini", credential_manager=mock_credentials)
    assert client is not None
```

### Integration Testing

```python
def test_credential_fallback():
    """Test credential fallback behavior"""
    import os
    
    # Set environment variable
    os.environ["TEST_KEY"] = "env-value"
    
    # Test explicit override
    creds = CredentialManager({"TEST_KEY": "explicit-value"}, load_env=True)
    assert creds.get_credential("TEST_KEY") == "explicit-value"
    
    # Test environment fallback
    creds = CredentialManager({}, load_env=True)
    assert creds.get_credential("TEST_KEY") == "env-value"
    
    # Test default fallback
    creds = CredentialManager({}, load_env=False)
    assert creds.get_credential("TEST_KEY", "default") == "default"
```

## Performance Considerations

- **CredentialManager instances are lightweight** - create as needed
- **.env loading happens only once** per CredentialManager instance
- **No caching of environment variables** - values are fetched fresh each time
- **Validation happens immediately** - fail fast on missing credentials

## Future Enhancements

Potential future additions:
- **Credential encryption at rest**
- **Integration with secret management services** (AWS Secrets Manager, Azure Key Vault)
- **Credential rotation notifications**
- **Audit logging for credential access**
- **Integration with authentication providers**