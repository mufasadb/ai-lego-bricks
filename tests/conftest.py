"""
Pytest configuration for AI Lego Bricks tests.

This module configures pytest fixtures, VCR settings, and test environment
for both integration and unit tests.
"""

import os
import pytest
from typing import Dict, Any
from pathlib import Path

# Import our VCR configuration
from .vcr_config import get_pytest_vcr_config, get_unit_test_vcr_config


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    """
    Session-scoped VCR configuration fixture for pytest-recording.

    This fixture is automatically used by @pytest.mark.vcr() decorator.
    For unit tests, it uses record_mode=none to force cassette replay.
    """
    # Always use unit test configuration with host exclusion for unit tests
    # The record mode will be controlled by pytest command line args
    config = get_unit_test_vcr_config()
    
    # Ensure host is excluded from matching for unit tests
    if "match_on" in config:
        # Remove 'host' from match_on if it exists
        match_on = [m for m in config["match_on"] if m != "host"]
        config["match_on"] = match_on
    
    return config


@pytest.fixture(scope="session")
def cassette_dir() -> Path:
    """
    Return the directory where VCR cassettes are stored.
    """
    return Path(__file__).parent / "cassettes"


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Set up test environment variables and configurations.

    This fixture runs automatically for all tests and ensures
    proper isolation and consistent environment setup.
    """
    # Set test-specific environment variables
    monkeypatch.setenv("TESTING", "true")

    # Ensure we don't accidentally use production settings
    monkeypatch.setenv("ENVIRONMENT", "test")

    # For integration tests, check that required env vars are present
    # but don't fail here - let individual tests handle missing vars


@pytest.fixture
def mock_credentials():
    """
    Provide mock credentials for unit tests.

    These are safe placeholder values that will never be recorded
    to cassettes since unit tests run with record_mode=none.
    """
    return {
        "OPENAI_API_KEY": "sk-test-mock-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-mock-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-supabase-key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "test",
        "NEO4J_PASSWORD": "test",
    }


@pytest.fixture
def integration_env_check():
    """
    Fixture that checks for required environment variables in integration tests.

    Use this fixture in integration tests to ensure proper setup.
    """
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        # Add other required vars as needed
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        pytest.skip(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


# Configure pytest markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    # Add marker descriptions
    config.addinivalue_line(
        "markers",
        "real_api: marks tests as requiring real API calls (integration tests)",
    )
    config.addinivalue_line(
        "markers", "cassette: marks tests as using VCR cassettes (unit tests)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test location.

    Automatically mark tests in integration/ as integration tests
    and tests in unit/ as unit tests.
    """
    for item in items:
        # Get the test file path relative to tests directory
        test_path = Path(item.fspath).relative_to(Path(__file__).parent)

        # Auto-mark based on directory structure
        if "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.real_api)
        elif "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.cassette)


# Custom pytest command line options
def pytest_addoption(parser):
    """Add custom command line options for test execution."""
    # Don't add --record-mode as pytest-recording already provides it

    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests",
    )

    parser.addoption(
        "--unit-only", action="store_true", default=False, help="Run only unit tests"
    )


def pytest_runtest_setup(item):
    """
    Set up individual test runs with proper VCR configuration.
    """
    # Get record mode from command line
    record_mode = item.config.getoption("--record-mode")

    # Skip integration tests if we're in none mode and missing env vars
    if record_mode == "none" and "integration" in item.keywords:
        # Check for required env vars for integration tests
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Integration test skipped in none mode without API keys")

    # Apply command line filters
    if (
        item.config.getoption("--integration-only")
        and "integration" not in item.keywords
    ):
        pytest.skip("Skipping non-integration test")

    if item.config.getoption("--unit-only") and "unit" not in item.keywords:
        pytest.skip("Skipping non-unit test")
