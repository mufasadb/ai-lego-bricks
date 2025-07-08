# Credentials Module

Centralized credential management for AI Lego Bricks with environment variable fallback and secure credential injection.

## Features

- **Fallback Chain**: Explicit credentials → Environment variables → Default values
- **Validation**: Required credential validation with clear error messages
- **Security**: No hardcoded credentials, supports .env files
- **Flexibility**: Programmatic credential injection for library use

## Quick Start

```python
from credentials import CredentialManager, create_credential_manager

# Basic usage with environment variables
cred_manager = CredentialManager()
api_key = cred_manager.get_credential("OPENAI_API_KEY")

# With explicit credentials (overrides environment)
cred_manager = CredentialManager(credentials={"API_KEY": "your-key"})

# Factory function
cred_manager = create_credential_manager()
```

## Common Patterns

### Environment Variables + .env
```python
# .env file support (automatic)
cred_manager = CredentialManager()
token = cred_manager.get_credential("GITHUB_TOKEN")
```

### Required Credentials
```python
# Validate required credentials
try:
    api_key = cred_manager.require_credential("OPENAI_API_KEY", "OpenAI")
except ValueError as e:
    print(f"Missing credential: {e}")

# Validate multiple credentials
credentials = cred_manager.validate_required_credentials(
    ["API_KEY", "SECRET_KEY"], 
    "MyService"
)
```

### Programmatic Injection
```python
# For libraries - inject credentials explicitly
cred_manager = CredentialManager(
    credentials={"API_KEY": user_provided_key},
    load_env=False  # Don't load .env in library context
)
```

### Checking Credentials
```python
# Check if credential exists
if cred_manager.has_credential("OPTIONAL_KEY"):
    # Use optional feature

# Get multiple credentials
creds = cred_manager.get_multiple_credentials(["KEY1", "KEY2"])
```

## Security Best Practices

1. **Never hardcode credentials** - use environment variables or explicit injection
2. **Use .env files** for development (add to .gitignore)
3. **Validate required credentials** early with clear error messages
4. **Use explicit injection** for library code to avoid environment pollution
5. **Check credential existence** before using optional features

## Error Handling

```python
try:
    credentials = cred_manager.validate_required_credentials(
        ["API_KEY", "SECRET"], "MyService"
    )
except ValueError as e:
    # Handle missing credentials gracefully
    print(f"Setup required: {e}")
```

The credential manager provides clear error messages indicating which credentials are missing and how to provide them.