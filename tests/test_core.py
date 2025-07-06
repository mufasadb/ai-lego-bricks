"""
Tests for the AI Lego Bricks core functionality.
"""

import pytest
from ailego.core import get_version, get_available_providers


def test_get_version():
    """Test that version can be retrieved."""
    version = get_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_get_available_providers():
    """Test that available providers can be retrieved."""
    providers = get_available_providers()
    assert isinstance(providers, dict)
    assert "memory" in providers
    assert "llm" in providers
    assert "tts" in providers
    assert "prompt" in providers
    
    # Each service should have a list of providers
    for service, provider_list in providers.items():
        assert isinstance(provider_list, list)