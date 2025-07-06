"""
Unit tests for agent orchestration functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from agent_orchestration.orchestrator import AgentOrchestrator, WorkflowExecutor
from agent_orchestration.models import WorkflowConfig, ExecutionContext, WorkflowResult, StepConfig
from agent_orchestration.step_handlers import StepHandlerRegistry


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = AgentOrchestrator()
        
        self.simple_workflow = {
            "name": "TestWorkflow",
            "description": "A test workflow",
            "steps": [
                {
                    "id": "step1",
                    "type": "llm_prompt",
                    "prompt": "Hello world",
                    "provider": "auto"
                }
            ]
        }
        
        self.complex_workflow = {
            "name": "ComplexWorkflow", 
            "description": "Multi-step workflow",
            "steps": [
                {
                    "id": "step1",
                    "type": "llm_prompt",
                    "prompt": "Analyze: {input}",
                    "provider": "auto"
                },
                {
                    "id": "step2", 
                    "type": "memory_store",
                    "content": "{step1.response}",
                    "metadata": {"source": "workflow"}
                },
                {
                    "id": "step3",
                    "type": "tts_generate",
                    "text": "{step1.response}",
                    "provider": "auto"
                }
            ]
        }
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert hasattr(self.orchestrator, 'step_handlers')
        assert hasattr(self.orchestrator, 'execute_workflow')
    
    def test_load_workflow_from_dict(self):
        """Test loading workflow from dictionary."""
        workflow = self.orchestrator.load_workflow(self.simple_workflow)
        
        assert workflow is not None
        assert workflow.name == "TestWorkflow"
        assert workflow.description == "A test workflow"
        assert len(workflow.steps) == 1
        assert workflow.steps[0].id == "step1"
        assert workflow.steps[0].type == "llm_prompt"
    
    def test_load_workflow_from_file(self, sample_workflow_file):
        """Test loading workflow from file."""
        workflow = self.orchestrator.load_workflow_from_file(str(sample_workflow_file))
        
        assert workflow is not None
        assert workflow.name is not None
        assert len(workflow.steps) > 0
    
    def test_load_invalid_workflow(self):
        """Test loading invalid workflow raises appropriate error."""
        invalid_workflow = {"invalid": "structure"}
        
        with pytest.raises(ValueError):
            self.orchestrator.load_workflow(invalid_workflow)
    
    @patch('agent_orchestration.orchestrator.LLMPromptHandler')
    def test_execute_simple_workflow(self, mock_handler_class):
        """Test executing a simple workflow."""
        # Setup mock handler
        mock_handler = Mock()
        mock_handler.execute.return_value = StepResult(
            step_id="step1",
            status="completed", 
            response="Test response",
            execution_time=1.0
        )
        mock_handler_class.return_value = mock_handler
        
        # Execute workflow
        result = self.orchestrator.execute_workflow(self.simple_workflow)
        
        # Verify results
        assert result is not None
        assert result.status == "completed"
        assert "step1" in result.step_results
        assert result.step_results["step1"].response == "Test response"
        
        # Verify handler was called
        mock_handler.execute.assert_called_once()
    
    @patch('agent_orchestration.orchestrator.LLMPromptHandler')
    @patch('agent_orchestration.orchestrator.MemoryStoreHandler')
    @patch('agent_orchestration.orchestrator.TTSGenerateHandler')
    def test_execute_complex_workflow(self, mock_tts, mock_memory, mock_llm):
        """Test executing a complex multi-step workflow."""
        # Setup mock handlers
        mock_llm_handler = Mock()
        mock_llm_handler.execute.return_value = StepResult(
            step_id="step1",
            status="completed",
            response="Analysis complete",
            execution_time=1.0
        )
        mock_llm.return_value = mock_llm_handler
        
        mock_memory_handler = Mock()
        mock_memory_handler.execute.return_value = StepResult(
            step_id="step2",
            status="completed",
            response="memory_id_123",
            execution_time=0.5
        )
        mock_memory.return_value = mock_memory_handler
        
        mock_tts_handler = Mock()
        mock_tts_handler.execute.return_value = StepResult(
            step_id="step3",
            status="completed", 
            response="/tmp/audio.wav",
            execution_time=2.0
        )
        mock_tts.return_value = mock_tts_handler
        
        # Execute workflow
        result = self.orchestrator.execute_workflow(
            self.complex_workflow,
            input_data={"input": "test data"}
        )
        
        # Verify results
        assert result.status == "completed"
        assert len(result.step_results) == 3
        assert all(step.status == "completed" for step in result.step_results.values())
        
        # Verify all handlers were called
        mock_llm_handler.execute.assert_called_once()
        mock_memory_handler.execute.assert_called_once()
        mock_tts_handler.execute.assert_called_once()
    
    def test_variable_substitution(self):
        """Test variable substitution in workflow steps."""
        workflow_data = {
            "name": "VariableTest",
            "steps": [
                {
                    "id": "step1",
                    "type": "llm_prompt",
                    "prompt": "Process this: {input}",
                    "provider": "auto"
                }
            ]
        }
        
        with patch('agent_orchestration.orchestrator.LLMPromptHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.execute.return_value = StepResult(
                step_id="step1",
                status="completed",
                response="Processed: test data",
                execution_time=1.0
            )
            mock_handler_class.return_value = mock_handler
            
            result = self.orchestrator.execute_workflow(
                workflow_data,
                input_data={"input": "test data"}
            )
            
            # Verify the prompt was substituted correctly
            call_args = mock_handler.execute.call_args[0]
            step = call_args[0]
            assert "test data" in step.prompt
    
    def test_step_dependency_resolution(self):
        """Test that step dependencies are resolved correctly."""
        workflow_data = {
            "name": "DependencyTest",
            "steps": [
                {
                    "id": "step1",
                    "type": "llm_prompt",
                    "prompt": "First step",
                    "provider": "auto"
                },
                {
                    "id": "step2",
                    "type": "llm_prompt", 
                    "prompt": "Second step using: {step1.response}",
                    "provider": "auto"
                }
            ]
        }
        
        with patch('agent_orchestration.orchestrator.LLMPromptHandler') as mock_handler_class:
            mock_handler = Mock()
            
            # First call returns response for step1
            # Second call should receive step1's response in the prompt
            responses = [
                StepResult(step_id="step1", status="completed", response="Step 1 result", execution_time=1.0),
                StepResult(step_id="step2", status="completed", response="Step 2 result", execution_time=1.0)
            ]
            mock_handler.execute.side_effect = responses
            mock_handler_class.return_value = mock_handler
            
            result = self.orchestrator.execute_workflow(workflow_data)
            
            # Verify both steps completed
            assert result.status == "completed"
            assert len(result.step_results) == 2
            
            # Verify second call received the substituted prompt
            assert mock_handler.execute.call_count == 2
            second_call_args = mock_handler.execute.call_args_list[1][0]
            step2 = second_call_args[0]
            assert "Step 1 result" in step2.prompt
    
    def test_error_handling_in_workflow(self):
        """Test error handling during workflow execution."""
        with patch('agent_orchestration.orchestrator.LLMPromptHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.execute.side_effect = Exception("Test error")
            mock_handler_class.return_value = mock_handler
            
            result = self.orchestrator.execute_workflow(self.simple_workflow)
            
            # Verify error was handled
            assert result.status == "failed"
            assert "step1" in result.step_results
            assert result.step_results["step1"].status == "failed"
    
    def test_workflow_validation(self):
        """Test workflow validation."""
        # Test missing required fields
        invalid_workflows = [
            {},  # Empty workflow
            {"name": "Test"},  # Missing steps
            {"steps": []},  # Missing name
            {"name": "Test", "steps": [{}]},  # Empty step
            {"name": "Test", "steps": [{"id": "step1"}]},  # Missing step type
        ]
        
        for invalid_workflow in invalid_workflows:
            with pytest.raises(ValueError):
                self.orchestrator.load_workflow(invalid_workflow)
    
    def test_workflow_context_management(self):
        """Test workflow context management."""
        context = WorkflowContext()
        
        # Test setting and getting variables
        context.set_variable("test_var", "test_value")
        assert context.get_variable("test_var") == "test_value"
        
        # Test step results
        step_result = StepResult(
            step_id="test_step",
            status="completed",
            response="test_response",
            execution_time=1.0
        )
        context.add_step_result(step_result)
        
        assert "test_step" in context.step_results
        assert context.step_results["test_step"].response == "test_response"
    
    def test_concurrent_step_execution(self):
        """Test concurrent execution of independent steps."""
        workflow_data = {
            "name": "ConcurrentTest",
            "steps": [
                {
                    "id": "step1",
                    "type": "llm_prompt",
                    "prompt": "Independent step 1",
                    "provider": "auto"
                },
                {
                    "id": "step2",
                    "type": "llm_prompt",
                    "prompt": "Independent step 2", 
                    "provider": "auto"
                }
            ]
        }
        
        with patch('agent_orchestration.orchestrator.LLMPromptHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.execute.side_effect = [
                StepResult(step_id="step1", status="completed", response="Response 1", execution_time=1.0),
                StepResult(step_id="step2", status="completed", response="Response 2", execution_time=1.0)
            ]
            mock_handler_class.return_value = mock_handler
            
            result = self.orchestrator.execute_workflow(workflow_data)
            
            # Both steps should complete successfully
            assert result.status == "completed"
            assert len(result.step_results) == 2
            assert mock_handler.execute.call_count == 2


class TestStepHandlers:
    """Test suite for individual step handlers."""
    
    def test_llm_prompt_handler(self):
        """Test LLM prompt handler."""
        handler = LLMPromptHandler()
        
        step = WorkflowStep(
            id="test_step",
            type="llm_prompt",
            prompt="Hello world",
            provider="auto"
        )
        
        context = WorkflowContext()
        
        with patch('llm.generation_service.create_generation_service') as mock_service:
            mock_llm = Mock()
            mock_llm.generate.return_value = "Hello! How can I help you?"
            mock_service.return_value = mock_llm
            
            result = handler.execute(step, context)
            
            assert result.status == "completed"
            assert result.response == "Hello! How can I help you?"
            assert result.step_id == "test_step"
            mock_llm.generate.assert_called_once_with("Hello world")
    
    def test_memory_store_handler(self):
        """Test memory store handler."""
        handler = MemoryStoreHandler()
        
        step = WorkflowStep(
            id="memory_step",
            type="memory_store",
            content="Test content to store",
            metadata={"source": "test"}
        )
        
        context = WorkflowContext()
        
        with patch('memory.memory_factory.create_memory_service') as mock_service:
            mock_memory = Mock()
            mock_memory.store_memory.return_value = "memory_id_123"
            mock_service.return_value = mock_memory
            
            result = handler.execute(step, context)
            
            assert result.status == "completed"
            assert result.response == "memory_id_123"
            mock_memory.store_memory.assert_called_once_with(
                "Test content to store",
                {"source": "test"}
            )
    
    def test_memory_retrieve_handler(self):
        """Test memory retrieve handler."""
        handler = MemoryRetrieveHandler()
        
        step = WorkflowStep(
            id="retrieve_step",
            type="memory_retrieve",
            query="test query",
            limit=5
        )
        
        context = WorkflowContext()
        
        with patch('memory.memory_factory.create_memory_service') as mock_service:
            mock_memory = Mock()
            mock_results = [
                Mock(content="Result 1", similarity=0.9),
                Mock(content="Result 2", similarity=0.8)
            ]
            mock_memory.retrieve_memories.return_value = mock_results
            mock_service.return_value = mock_memory
            
            result = handler.execute(step, context)
            
            assert result.status == "completed"
            assert len(result.response) == 2
            mock_memory.retrieve_memories.assert_called_once_with("test query", limit=5)
    
    def test_tts_generate_handler(self):
        """Test TTS generate handler."""
        handler = TTSGenerateHandler()
        
        step = WorkflowStep(
            id="tts_step",
            type="tts_generate",
            text="Hello world",
            provider="auto"
        )
        
        context = WorkflowContext()
        
        with patch('tts.tts_factory.create_tts_service') as mock_service:
            mock_tts = Mock()
            mock_tts.generate_speech.return_value = "/tmp/audio.wav"
            mock_service.return_value = mock_tts
            
            result = handler.execute(step, context)
            
            assert result.status == "completed"
            assert result.response == "/tmp/audio.wav"
            mock_tts.generate_speech.assert_called_once_with("Hello world")


class TestWorkflowModels:
    """Test suite for workflow data models."""
    
    def test_workflow_step_creation(self):
        """Test WorkflowStep model creation."""
        step = WorkflowStep(
            id="test_step",
            type="llm_prompt",
            prompt="Test prompt",
            provider="auto"
        )
        
        assert step.id == "test_step"
        assert step.type == "llm_prompt"
        assert step.prompt == "Test prompt"
        assert step.provider == "auto"
    
    def test_step_result_creation(self):
        """Test StepResult model creation."""
        result = StepResult(
            step_id="test_step",
            status="completed",
            response="Test response",
            execution_time=1.5
        )
        
        assert result.step_id == "test_step"
        assert result.status == "completed"
        assert result.response == "Test response"
        assert result.execution_time == 1.5
    
    def test_workflow_context_operations(self):
        """Test WorkflowContext operations."""
        context = WorkflowContext()
        
        # Test input data
        context.input_data = {"key": "value"}
        assert context.input_data["key"] == "value"
        
        # Test variables
        context.set_variable("var1", "value1")
        assert context.get_variable("var1") == "value1"
        assert context.get_variable("nonexistent") is None
        
        # Test step results
        result = StepResult("step1", "completed", "response", 1.0)
        context.add_step_result(result)
        assert "step1" in context.step_results
        assert context.step_results["step1"].response == "response"