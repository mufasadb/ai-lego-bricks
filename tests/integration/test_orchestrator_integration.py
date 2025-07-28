"""
Integration tests for Agent Orchestrator using real agent examples.

These tests run actual agent workflows from the examples directory
and record VCR cassettes for later unit testing.
"""

import os
import json
import pytest
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from agent_orchestration.orchestrator import AgentOrchestrator
from agent_orchestration.models import WorkflowConfig


class TestAgentOrchestratorIntegration:
    """Integration tests for Agent Orchestrator with real examples."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance for testing."""
        return AgentOrchestrator()

    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory path."""
        return Path(__file__).parent.parent.parent / "agent_orchestration" / "examples"

    @pytest.fixture
    def example_files(self, examples_dir):
        """Get all JSON example files."""
        return list(examples_dir.glob("*.json"))

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_basic_chat_agent(self, orchestrator, examples_dir, integration_env_check):
        """Test the basic chat agent example."""
        example_file = examples_dir / "basic_chat_agent.json"

        if not example_file.exists():
            pytest.skip("basic_chat_agent.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        # Create workflow
        workflow = WorkflowConfig.model_validate(workflow_data)

        # Execute workflow
        result = orchestrator.execute_workflow(workflow)

        # Verify execution
        assert result is not None
        assert hasattr(result, "step_outputs")
        assert len(result.step_outputs) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_simple_ollama_chat(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test the simple ollama chat example."""
        example_file = examples_dir / "simple_ollama_chat.json"

        if not example_file.exists():
            pytest.skip("simple_ollama_chat.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_multi_turn_chat_demo(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test the multi-turn chat demo."""
        example_file = examples_dir / "multi_turn_chat_demo.json"

        if not example_file.exists():
            pytest.skip("multi_turn_chat_demo.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_streaming_agent(self, orchestrator, examples_dir, integration_env_check):
        """Test streaming agent functionality."""
        example_file = examples_dir / "streaming_agent.json"

        if not example_file.exists():
            pytest.skip("streaming_agent.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_http_request_examples(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test HTTP request examples."""
        example_file = examples_dir / "http_request_examples.json"

        if not example_file.exists():
            pytest.skip("http_request_examples.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complex_workflow_agent(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test complex workflow agent."""
        example_file = examples_dir / "complex_workflow_agent.json"

        if not example_file.exists():
            pytest.skip("complex_workflow_agent.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_parallel_workflow_example(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test parallel workflow execution."""
        example_file = examples_dir / "parallel_workflow_example.json"

        if not example_file.exists():
            pytest.skip("parallel_workflow_example.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_text_analyzer_specialist(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test text analyzer specialist agent."""
        example_file = examples_dir / "text_analyzer_specialist.json"

        if not example_file.exists():
            pytest.skip("text_analyzer_specialist.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_dollar_amount_extraction_agent(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test dollar amount extraction agent."""
        example_file = examples_dir / "dollar_amount_extraction_agent.json"

        if not example_file.exists():
            pytest.skip("dollar_amount_extraction_agent.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_agent_coordinator(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test multi-agent coordinator."""
        example_file = examples_dir / "multi_agent_coordinator.json"

        if not example_file.exists():
            pytest.skip("multi_agent_coordinator.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "example_filename",
        [
            "json_props_demo_agent.json",
            "voice_assistant_agent.json",
            "ai_coordinator_agent.json",
            "graph_memory_agent.json",
            "simple_enhanced_streaming.json",
            "enhanced_streaming_multi_agent.json",
            "multi_agent_streaming_demo.json",
            "simple_multi_agent_streaming.json",
            "gemini_ollama_parallel_agent.json",
        ],
    )
    def test_additional_agents(
        self, orchestrator, examples_dir, example_filename, integration_env_check
    ):
        """Test additional agent examples parametrically."""
        example_file = examples_dir / example_filename

        if not example_file.exists():
            pytest.skip(f"{example_filename} not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_tool_functionality_across_providers(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test tool functionality across different providers."""
        example_file = examples_dir / "test_tool_functionality_across_providers.json"

        if not example_file.exists():
            pytest.skip("test_tool_functionality_across_providers.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_debug_mcp_tool_calling_reliability(
        self, orchestrator, examples_dir, integration_env_check
    ):
        """Test MCP tool calling reliability debugging."""
        example_file = examples_dir / "debug_mcp_tool_calling_reliability.json"

        if not example_file.exists():
            pytest.skip("debug_mcp_tool_calling_reliability.json not found")

        with open(example_file, "r") as f:
            workflow_data = json.load(f)

        workflow = WorkflowConfig.model_validate(workflow_data)
        result = orchestrator.execute_workflow(workflow)

        assert result is not None
        assert hasattr(result, "step_outputs")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_all_examples_can_load(self, orchestrator, example_files):
        """Test that all example files can be loaded as valid workflows."""
        failed_files = []

        for example_file in example_files:
            try:
                with open(example_file, "r") as f:
                    workflow_data = json.load(f)

                # Just validate the schema, don't execute
                workflow = WorkflowConfig.model_validate(workflow_data)
                assert workflow is not None

            except Exception as e:
                failed_files.append((example_file.name, str(e)))

        if failed_files:
            pytest.fail(
                f"Failed to load {len(failed_files)} example files: {failed_files}"
            )
