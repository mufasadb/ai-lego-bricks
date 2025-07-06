# Credential Management Migration Guide

## Overview

AI Lego Bricks has been updated with a new credential management system that provides better library safety, credential isolation, and explicit configuration options. This guide will help you migrate your existing code to use the new system.

## What Changed

### Before (Old Pattern)
```python
# Automatic .env loading at module import
from llm.llm_factory import create_text_client

# Credentials automatically loaded from environment
client = create_text_client("gemini")
```

### After (New Pattern - Backward Compatible)
```python
from llm.llm_factory import create_text_client
from credentials import CredentialManager

# Option 1: Backward compatible (still loads .env automatically)
client = create_text_client("gemini")

# Option 2: Explicit credential injection for library usage
credentials = {"GOOGLE_AI_STUDIO_KEY": "your-api-key"}
cred_manager = CredentialManager(credentials, load_env=False)
client = create_text_client("gemini", credential_manager=cred_manager)
```

## Key Benefits

### 1. Library Safety
- **Old**: Importing any module automatically loads `.env` files
- **New**: Optional environment loading, preventing unwanted `.env` overwrites

### 2. Credential Isolation
- **Old**: All services share global environment variables
- **New**: Each service instance can have isolated credentials

### 3. Explicit Configuration
- **Old**: Credentials only come from environment variables
- **New**: Credentials can be passed explicitly or via environment

## Migration Steps

### Step 1: Understand Backward Compatibility

**Good News**: Your existing code will continue to work without changes. The new system is fully backward compatible.

```python
# This still works exactly as before
from llm.llm_factory import create_text_client
client = create_text_client("gemini")
```

### Step 2: Update for Library Usage (Optional)

If you're using AI Lego Bricks as a library in another application:

```python
from llm.llm_factory import create_text_client
from memory.memory_factory import create_memory_service
from tts.tts_factory import create_tts_service
from credentials import CredentialManager

# Create credential manager with explicit credentials
credentials = {
    "GOOGLE_AI_STUDIO_KEY": "your-gemini-key",
    "ANTHROPIC_API_KEY": "your-anthropic-key",
    "OPENAI_API_KEY": "your-openai-key",
    "SUPABASE_URL": "your-supabase-url",
    "SUPABASE_ANON_KEY": "your-supabase-key"
}

# Disable automatic .env loading for library safety
cred_manager = CredentialManager(credentials, load_env=False)

# Pass credential manager to services
llm_client = create_text_client("gemini", credential_manager=cred_manager)
memory_service = create_memory_service("supabase", credential_manager=cred_manager)
tts_service = create_tts_service("openai", credential_manager=cred_manager)
```

### Step 3: Update Factory Usage

All factory functions now accept an optional `credential_manager` parameter:

```python
from llm.llm_factory import LLMClientFactory
from credentials import CredentialManager

# Create with custom credentials
cred_manager = CredentialManager({"GOOGLE_AI_STUDIO_KEY": "key"})

# Factory method usage
client = LLMClientFactory.create_text_client(
    provider=LLMProvider.GEMINI,
    credential_manager=cred_manager
)

# Convenience function usage
client = create_text_client("gemini", credential_manager=cred_manager)
```

## Advanced Usage Patterns

### 1. Per-Service Credential Isolation

```python
from credentials import CredentialManager

# Different credentials for different services
gemini_creds = CredentialManager({"GOOGLE_AI_STUDIO_KEY": "gemini-key"})
anthropic_creds = CredentialManager({"ANTHROPIC_API_KEY": "anthropic-key"})

gemini_client = create_text_client("gemini", credential_manager=gemini_creds)
anthropic_client = create_text_client("anthropic", credential_manager=anthropic_creds)
```

### 2. Mixed Environment and Explicit Credentials

```python
from credentials import CredentialManager

# Some credentials from environment, some explicit
mixed_creds = CredentialManager({
    "GOOGLE_AI_STUDIO_KEY": "explicit-key"  # Override env var
    # ANTHROPIC_API_KEY will be loaded from environment
}, load_env=True)

client = create_text_client("gemini", credential_manager=mixed_creds)
```

### 3. Runtime Credential Validation

```python
from credentials import CredentialManager

cred_manager = CredentialManager(load_env=True)

# Check if credentials exist before creating services
if cred_manager.has_credential("GOOGLE_AI_STUDIO_KEY"):
    gemini_client = create_text_client("gemini", credential_manager=cred_manager)

# Validate multiple credentials at once
try:
    required_creds = cred_manager.validate_required_credentials(
        ["GOOGLE_AI_STUDIO_KEY", "ANTHROPIC_API_KEY"],
        "Multi-LLM Setup"
    )
    print("All credentials available!")
except ValueError as e:
    print(f"Missing credentials: {e}")
```

## CredentialManager API Reference

### Constructor
```python
CredentialManager(credentials=None, load_env=True)
```
- `credentials`: Dict of explicit credentials (optional)
- `load_env`: Whether to load .env file (default: True for backward compatibility)

### Methods
```python
# Get credential with fallback
get_credential(key: str, default=None) -> Optional[str]

# Get required credential (raises error if missing)
require_credential(key: str, service_name: str) -> str

# Check if credential exists
has_credential(key: str) -> bool

# Get multiple credentials
get_multiple_credentials(keys: list) -> Dict[str, Optional[str]]

# Validate required credentials
validate_required_credentials(keys: list, service_name: str) -> Dict[str, str]
```

## Common Migration Scenarios

### Scenario 1: Existing Application (No Changes Needed)
```python
# Your existing code continues to work
from llm.llm_factory import create_text_client
client = create_text_client("gemini")
```

### Scenario 2: Library Integration
```python
from credentials import CredentialManager
from llm.llm_factory import create_text_client

def create_ai_service(api_key: str):
    """Create AI service for library consumers"""
    creds = CredentialManager(
        {"GOOGLE_AI_STUDIO_KEY": api_key}, 
        load_env=False  # Don't interfere with consumer's .env
    )
    return create_text_client("gemini", credential_manager=creds)
```

### Scenario 3: Multi-Tenant Application
```python
from credentials import CredentialManager

def create_tenant_services(tenant_credentials: dict):
    """Create isolated services for each tenant"""
    cred_manager = CredentialManager(tenant_credentials, load_env=False)
    
    return {
        'llm': create_text_client("gemini", credential_manager=cred_manager),
        'memory': create_memory_service("supabase", credential_manager=cred_manager),
        'tts': create_tts_service("openai", credential_manager=cred_manager)
    }
```

## Troubleshooting

### Issue: "CredentialManager not found"
```python
# Solution: Import from credentials module
from credentials import CredentialManager
```

### Issue: Credentials not loading
```python
# Check if credential manager is properly initialized
cred_manager = CredentialManager(load_env=True)
print(cred_manager.get_credential("GOOGLE_AI_STUDIO_KEY"))

# For explicit credentials
creds = {"GOOGLE_AI_STUDIO_KEY": "your-key"}
cred_manager = CredentialManager(creds, load_env=False)
```

### Issue: Library users complaining about .env interference
```python
# Always set load_env=False when creating services for library consumers
def create_service_for_library(api_key: str):
    creds = CredentialManager({"API_KEY": api_key}, load_env=False)
    return create_text_client("gemini", credential_manager=creds)
```

## Best Practices

1. **For Applications**: Keep using the default behavior for simplicity
2. **For Libraries**: Always use explicit credentials with `load_env=False`
3. **For Multi-Tenant**: Create separate CredentialManager instances per tenant
4. **For Testing**: Use explicit credentials to avoid dependency on test environment
5. **For Production**: Validate required credentials at startup

## Summary

The new credential management system provides:
- ✅ **Backward Compatibility**: Existing code works without changes
- ✅ **Library Safety**: No unwanted .env loading when used as library
- ✅ **Credential Isolation**: Different services can have different credentials
- ✅ **Explicit Configuration**: Direct credential injection support
- ✅ **Better Error Messages**: Clear validation and error reporting

Choose the migration approach that best fits your use case!