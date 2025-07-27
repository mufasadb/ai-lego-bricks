"""
Utility functions to handle imports that work both as a package and standalone.
This resolves the "attempted relative import beyond top-level package" error.
"""

import sys
import importlib
from typing import Any, Optional


def safe_import(relative_module: str, absolute_module: str, fromlist: Optional[list] = None) -> Any:
    """
    Safely import a module using relative imports when in package, absolute when standalone.
    
    Args:
        relative_module: Relative import path (e.g., "..credentials")
        absolute_module: Absolute import path (e.g., "credentials") 
        fromlist: List of names to import from module (for "from X import Y")
        
    Returns:
        Imported module or None if import fails
    """
    try:
        # Try relative import first (works when used as package)
        return importlib.import_module(relative_module, package=__name__)
    except (ImportError, ValueError):
        try:
            # Fall back to absolute import (works when run directly)
            return importlib.import_module(absolute_module, package=None)
        except ImportError:
            return None


def safe_import_from(relative_module: str, absolute_module: str, names: list, package: str = None) -> dict:
    """
    Safely import specific names from a module.
    
    Args:
        relative_module: Relative import path
        absolute_module: Absolute import path  
        names: List of names to import
        package: Package name for relative imports
        
    Returns:
        Dictionary mapping names to imported objects
    """
    result = {}
    
    try:
        # Try relative import
        if package:
            module = importlib.import_module(relative_module, package=package)
        else:
            module = importlib.import_module(relative_module)
            
        for name in names:
            if hasattr(module, name):
                result[name] = getattr(module, name)
                
    except (ImportError, ValueError):
        try:
            # Fall back to absolute import
            module = importlib.import_module(absolute_module)
            for name in names:
                if hasattr(module, name):
                    result[name] = getattr(module, name)
        except ImportError:
            pass
    
    return result


def conditional_import_credentials():
    """Import credentials with fallback for both package and standalone usage."""
    try:
        from .credentials import CredentialManager, default_credential_manager
        return CredentialManager, default_credential_manager
    except ImportError:
        try:
            from credentials import CredentialManager, default_credential_manager
            return CredentialManager, default_credential_manager
        except ImportError:
            return None, None


def conditional_import_llm():
    """Import LLM components with fallback."""
    try:
        from .llm.llm_types import LLMProvider, VisionProvider
        from .llm.llm_factory import LLMClientFactory
        return LLMProvider, VisionProvider, LLMClientFactory
    except ImportError:
        try:
            from llm.llm_types import LLMProvider, VisionProvider  
            from llm.llm_factory import LLMClientFactory
            return LLMProvider, VisionProvider, LLMClientFactory
        except ImportError:
            return None, None, None