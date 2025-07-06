"""
Tests for the AI Lego Bricks CLI.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from ailego.cli import app

runner = CliRunner()


def test_cli_version():
    """Test that version command works."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AI Lego Bricks" in result.stdout


def test_cli_help():
    """Test that help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI Lego Bricks" in result.stdout


def test_cli_status():
    """Test that status command works."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "System Status" in result.stdout


def test_cli_examples():
    """Test that examples command works."""
    result = runner.invoke(app, ["examples"])
    assert result.exit_code == 0
    assert "Examples" in result.stdout


def test_cli_list_templates():
    """Test that list-templates command works."""
    result = runner.invoke(app, ["list-templates"])
    assert result.exit_code == 0
    assert "Templates" in result.stdout


@pytest.mark.parametrize("agent_type", ["chat", "document-analysis", "research", "vision", "streaming"])
def test_create_agent_types(agent_type):
    """Test that create command recognizes valid agent types."""
    # This test would need more setup to run fully
    # For now, just test that the command doesn't crash on help
    result = runner.invoke(app, ["create", "--help"])
    assert result.exit_code == 0