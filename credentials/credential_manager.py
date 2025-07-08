"""
Credential management for AI Lego Bricks

This module provides centralized credential handling with support for:
- Environment variable fallback
- Explicit credential injection
- Library-safe credential isolation
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv


class CredentialManager:
    """Manages credentials with support for explicit injection and environment fallback"""
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None, load_env: bool = True):
        """
        Initialize credential manager
        
        Args:
            credentials: Dictionary of explicit credentials (takes precedence over env vars)
            load_env: Whether to load .env file automatically (default: True for backward compatibility)
        """
        self.credentials = credentials or {}
        self.load_env = load_env
        
        if load_env:
            load_dotenv()
    
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get credential value with fallback chain:
        1. Explicit credentials passed to constructor
        2. Environment variables
        3. Default value
        
        Args:
            key: Credential key (case-sensitive)
            default: Default value if credential not found
            
        Returns:
            Credential value or default
        """
        # First check explicit credentials
        if key in self.credentials:
            return self.credentials[key]
        
        # Then check environment variables
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        
        # Finally return default
        return default
    
    def require_credential(self, key: str, service_name: str = "service") -> str:
        """
        Get required credential with validation
        
        Args:
            key: Credential key
            service_name: Service name for error messages
            
        Returns:
            Credential value
            
        Raises:
            ValueError: If credential is missing
        """
        value = self.get_credential(key)
        if not value:
            raise ValueError(
                f"{key} not found for {service_name}. "
                f"Please set it in environment variables or pass explicitly to CredentialManager."
            )
        return value
    
    def has_credential(self, key: str) -> bool:
        """Check if credential exists"""
        return self.get_credential(key) is not None
    
    def get_multiple_credentials(self, keys: list[str]) -> Dict[str, Optional[str]]:
        """Get multiple credentials at once"""
        return {key: self.get_credential(key) for key in keys}
    
    def validate_required_credentials(self, required_keys: list[str], service_name: str = "service") -> Dict[str, str]:
        """
        Validate that all required credentials are present
        
        Args:
            required_keys: List of required credential keys
            service_name: Service name for error messages
            
        Returns:
            Dictionary of validated credentials
            
        Raises:
            ValueError: If any required credential is missing
        """
        missing_keys = []
        credentials = {}
        
        for key in required_keys:
            value = self.get_credential(key)
            if not value:
                missing_keys.append(key)
            else:
                credentials[key] = value
        
        if missing_keys:
            raise ValueError(
                f"Missing required credentials for {service_name}: {', '.join(missing_keys)}. "
                f"Please set them in environment variables or pass explicitly to CredentialManager."
            )
        
        return credentials


# Factory function for common use cases
def create_credential_manager(
    credentials: Optional[Dict[str, str]] = None,
    load_env: bool = True
) -> CredentialManager:
    """
    Factory function to create CredentialManager with common defaults
    
    Args:
        credentials: Optional explicit credentials
        load_env: Whether to load .env file (default: True)
        
    Returns:
        CredentialManager instance
    """
    return CredentialManager(credentials=credentials, load_env=load_env)


# Default instance for backward compatibility
default_credential_manager = CredentialManager()