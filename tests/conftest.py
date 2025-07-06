"""
Pytest configuration and shared fixtures for AI Lego Bricks tests.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session") 
def sample_workflow():
    """Sample workflow for testing."""
    return {
        "name": "TestWorkflow",
        "description": "A simple test workflow",
        "steps": [
            {
                "id": "test_step_1",
                "type": "llm_prompt",
                "prompt": "Hello, world!",
                "provider": "auto"
            }
        ]
    }


@pytest.fixture(scope="session")
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return {
        "simple_response": "This is a test response.",
        "json_response": '{"result": "success", "data": {"key": "value"}}',
        "error_response": "Error: Something went wrong",
        "streaming_response": ["Hello", " there", "! How", " are", " you", "?"]
    }


@pytest.fixture(scope="function")
def mock_memory_service():
    """Mock memory service for testing."""
    memory_service = Mock()
    memory_service.store_memory.return_value = "test-memory-id-123"
    memory_service.get_memory_by_id.return_value = Mock(
        memory_id="test-memory-id-123",
        content="Test memory content",
        metadata={"test": True},
        similarity=0.95
    )
    memory_service.retrieve_memories.return_value = [
        Mock(
            memory_id="test-memory-id-123",
            content="Test memory content",
            metadata={"test": True},
            similarity=0.95
        )
    ]
    memory_service.delete_memory.return_value = True
    memory_service.delete_memories.return_value = {"test-memory-id-123": True}
    return memory_service


@pytest.fixture(scope="function")
def mock_llm_service():
    """Mock LLM service for testing."""
    llm_service = Mock()
    llm_service.generate.return_value = "This is a test response."
    llm_service.generate_stream.return_value = iter(["Hello", " world", "!"])
    return llm_service


@pytest.fixture(scope="function")
def mock_tts_service():
    """Mock TTS service for testing."""
    tts_service = Mock()
    tts_service.generate_speech.return_value = b"fake_audio_data"
    tts_service.generate_speech_stream.return_value = iter([b"chunk1", b"chunk2"])
    return tts_service


@pytest.fixture(scope="function")
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test_supabase_key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "test_password",
        "GOOGLE_API_KEY": "test_google_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "OPENAI_API_KEY": "test_openai_key",
        "OLLAMA_URL": "http://localhost:11434"
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture(scope="function")
def temporary_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create basic structure
        (workspace / "workflows").mkdir()
        (workspace / "output").mkdir()
        (workspace / "templates").mkdir()
        
        yield workspace


@pytest.fixture(scope="function")
def sample_workflow_file(temporary_workspace, sample_workflow):
    """Create a sample workflow file for testing."""
    workflow_file = temporary_workspace / "workflows" / "test_workflow.json"
    
    with open(workflow_file, 'w') as f:
        json.dump(sample_workflow, f, indent=2)
    
    return workflow_file


@pytest.fixture(scope="function")
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('builtins.open', mock_open_factory()) as mock_file:
        with patch('os.path.exists', return_value=True) as mock_exists:
            with patch('os.makedirs') as mock_makedirs:
                yield {
                    'open': mock_file,
                    'exists': mock_exists,
                    'makedirs': mock_makedirs
                }


def mock_open_factory():
    """Factory for creating mock file objects."""
    def mock_open_func(*args, **kwargs):
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        mock_file.read.return_value = '{"test": "data"}'
        mock_file.write = Mock()
        return mock_file
    return mock_open_func


@pytest.fixture(scope="function")
def capture_logs(caplog):
    """Capture logs for testing."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


# Skip integration tests if no services are available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle integration tests."""
    skip_integration = pytest.mark.skip(reason="Integration tests require external services")
    
    for item in items:
        if "integration" in item.keywords:
            # Check if required services are available
            if not _check_services_available():
                item.add_marker(skip_integration)


def _check_services_available():
    """Check if external services are available for integration tests."""
    # This is a simple check - in practice you might want more sophisticated detection
    required_env_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    return all(os.getenv(var) for var in required_env_vars)