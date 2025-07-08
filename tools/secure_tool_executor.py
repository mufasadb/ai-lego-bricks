"""
Secure tool executor that handles API keys and secrets safely.
"""
from abc import abstractmethod
from typing import Dict, Any, Optional, List
from .tool_types import ToolExecutor, ToolCall, ToolResult

try:
    from credentials import CredentialManager
except ImportError:
    # Fallback for testing
    class CredentialManager:
        def __init__(self, credentials=None, load_env=True):
            self.credentials = credentials or {}
        
        def get_credential(self, key: str, default=None):
            return self.credentials.get(key, default)
        
        def require_credential(self, key: str, service_name: str):
            value = self.get_credential(key)
            if not value:
                raise ValueError(f"Missing required credential '{key}' for {service_name}")
            return value

class SecureToolExecutor(ToolExecutor):
    """Base class for tool executors that require API keys or secrets."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None, 
                 required_credentials: Optional[List[str]] = None):
        """
        Initialize secure tool executor.
        
        Args:
            credential_manager: CredentialManager instance for handling secrets
            required_credentials: List of required credential keys
        """
        self.credential_manager = credential_manager or CredentialManager()
        self.required_credentials = required_credentials or []
        self._validate_credentials()
    
    def _validate_credentials(self):
        """Validate that all required credentials are available."""
        missing = []
        for cred_key in self.required_credentials:
            if not self.credential_manager.get_credential(cred_key):
                missing.append(cred_key)
        
        if missing:
            raise ValueError(
                f"Missing required credentials for {self.__class__.__name__}: {', '.join(missing)}"
            )
    
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a credential value safely."""
        return self.credential_manager.get_credential(key, default)
    
    def require_credential(self, key: str) -> str:
        """Get a required credential, raising an error if not found."""
        return self.credential_manager.require_credential(key, self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute the tool with secure credential handling."""
        pass

class APIToolExecutor(SecureToolExecutor):
    """Base class for tools that make API calls with authentication."""
    
    def __init__(self, base_url: str, api_key_name: str, 
                 credential_manager: Optional[CredentialManager] = None,
                 additional_headers: Optional[Dict[str, str]] = None):
        """
        Initialize API tool executor.
        
        Args:
            base_url: Base URL for the API
            api_key_name: Name of the environment variable containing the API key
            credential_manager: CredentialManager instance
            additional_headers: Additional headers to include in requests
        """
        self.base_url = base_url
        self.api_key_name = api_key_name
        self.additional_headers = additional_headers or {}
        
        super().__init__(credential_manager, [api_key_name])
    
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        api_key = self.require_credential(self.api_key_name)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.additional_headers
        }
        return headers
    
    async def make_api_request(self, endpoint: str, method: str = "GET", 
                              data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an authenticated API request."""
        import aiohttp
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self.get_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if data else None
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                
                return await response.json()

class DatabaseToolExecutor(SecureToolExecutor):
    """Base class for tools that connect to databases with credentials."""
    
    def __init__(self, connection_credentials: List[str],
                 credential_manager: Optional[CredentialManager] = None):
        """
        Initialize database tool executor.
        
        Args:
            connection_credentials: List of required connection credential keys
            credential_manager: CredentialManager instance
        """
        super().__init__(credential_manager, connection_credentials)
    
    def get_connection_string(self, template: str) -> str:
        """Build connection string from template and credentials."""
        creds = {}
        for cred_key in self.required_credentials:
            creds[cred_key] = self.require_credential(cred_key)
        
        return template.format(**creds)

class WebhookToolExecutor(SecureToolExecutor):
    """Base class for tools that send webhooks with authentication."""
    
    def __init__(self, webhook_url_key: str, secret_key: Optional[str] = None,
                 credential_manager: Optional[CredentialManager] = None):
        """
        Initialize webhook tool executor.
        
        Args:
            webhook_url_key: Credential key for webhook URL
            secret_key: Credential key for webhook secret (optional)
            credential_manager: CredentialManager instance
        """
        required_creds = [webhook_url_key]
        if secret_key:
            required_creds.append(secret_key)
        
        self.webhook_url_key = webhook_url_key
        self.secret_key = secret_key
        
        super().__init__(credential_manager, required_creds)
    
    def get_webhook_headers(self, payload: str) -> Dict[str, str]:
        """Get headers for webhook including signature if secret is configured."""
        headers = {"Content-Type": "application/json"}
        
        if self.secret_key:
            import hmac
            import hashlib
            
            secret = self.require_credential(self.secret_key)
            signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            headers["X-Signature-SHA256"] = f"sha256={signature}"
        
        return headers
    
    async def send_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook with authentication."""
        import aiohttp
        import json
        
        webhook_url = self.require_credential(self.webhook_url_key)
        payload = json.dumps(data)
        headers = self.get_webhook_headers(payload)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                data=payload,
                headers=headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Webhook failed: {response.status} - {error_text}")
                
                return {"status": response.status, "response": await response.text()}

def create_tool_with_credentials(tool_class, credential_config: Dict[str, Any], 
                                tool_config: Optional[Dict[str, Any]] = None) -> 'Tool':
    """
    Factory function to create a tool with credential management.
    
    Args:
        tool_class: Tool executor class (must inherit from SecureToolExecutor)
        credential_config: Configuration for credential management
        tool_config: Additional tool-specific configuration
    
    Returns:
        Tool instance with configured credentials
    
    Example:
        credential_config = {
            "credentials": {"OPENAI_API_KEY": "sk-..."},
            "load_env": True
        }
        tool = create_tool_with_credentials(
            OpenAIToolExecutor, 
            credential_config,
            {"model": "gpt-4"}
        )
    """
    from .tool_types import Tool
    
    # Create credential manager
    cred_manager = CredentialManager(
        credentials=credential_config.get("credentials"),
        load_env=credential_config.get("load_env", True)
    )
    
    # Create tool executor with credentials
    executor = tool_class(credential_manager=cred_manager, **(tool_config or {}))
    
    # For this factory, we assume the tool class has a schema attribute or method
    if hasattr(tool_class, 'get_schema'):
        schema = tool_class.get_schema()
    elif hasattr(tool_class, 'schema'):
        schema = tool_class.schema
    else:
        raise ValueError(f"Tool class {tool_class.__name__} must provide schema")
    
    return Tool(schema=schema, executor=executor)