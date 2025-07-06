"""
Integration tests for end-to-end workflow execution.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from agent_orchestration.orchestrator import AgentOrchestrator
from tests.fixtures.mock_responses import MockLLMResponses, MockWorkflowResponses


class TestWorkflowExecution:
    """Integration tests for complete workflow execution."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = WorkflowOrchestrator()
    
    @pytest.mark.integration
    def test_simple_chat_workflow(self, mock_environment_variables):
        """Test executing a simple chat workflow."""
        workflow = MockWorkflowResponses.SIMPLE_WORKFLOW
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            mock_service = mock_llm.return_value
            mock_service.generate.return_value = MockLLMResponses.SIMPLE_CHAT
            
            result = self.orchestrator.execute_workflow(workflow)
            
            assert result.status == "completed"
            assert len(result.step_results) == 1
            assert "step_1" in result.step_results
            assert MockLLMResponses.SIMPLE_CHAT in result.step_results["step_1"].response
    
    @pytest.mark.integration
    def test_document_analysis_workflow(self, mock_environment_variables):
        """Test executing a document analysis workflow."""
        workflow = {
            "name": "DocumentAnalysisWorkflow",
            "description": "Analyze a document and store results",
            "steps": [
                {
                    "id": "analyze_step",
                    "type": "llm_prompt",
                    "prompt": "Analyze this document: {input}",
                    "provider": "auto"
                },
                {
                    "id": "store_step",
                    "type": "memory_store",
                    "content": "{analyze_step.response}",
                    "metadata": {"type": "analysis", "source": "integration_test"}
                }
            ]
        }
        
        input_data = {"input": "This is a test document for analysis."}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            with patch('memory.memory_factory.create_memory_service') as mock_memory:
                # Setup mocks
                mock_llm_service = mock_llm.return_value
                mock_llm_service.generate.return_value = MockLLMResponses.DOCUMENT_ANALYSIS
                
                mock_memory_service = mock_memory.return_value
                mock_memory_service.store_memory.return_value = "analysis_memory_123"
                
                result = self.orchestrator.execute_workflow(workflow, input_data=input_data)
                
                assert result.status == "completed"
                assert len(result.step_results) == 2
                
                # Verify analysis step
                analysis_result = result.step_results["analyze_step"]
                assert analysis_result.status == "completed"
                assert "technical specification" in analysis_result.response.lower()
                
                # Verify storage step
                storage_result = result.step_results["store_step"]
                assert storage_result.status == "completed"
                assert storage_result.response == "analysis_memory_123"
    
    @pytest.mark.integration
    def test_conversation_with_memory_workflow(self, mock_environment_variables):
        """Test conversation workflow with memory retrieval."""
        workflow = {
            "name": "ConversationWithMemory",
            "description": "Chat with memory context",
            "steps": [
                {
                    "id": "retrieve_context",
                    "type": "memory_retrieve",
                    "query": "{input}",
                    "limit": 3
                },
                {
                    "id": "generate_response",
                    "type": "llm_prompt",
                    "prompt": "Context: {retrieve_context.results}\\n\\nUser: {input}\\n\\nAssistant:",
                    "provider": "auto"
                },
                {
                    "id": "store_interaction",
                    "type": "memory_store",
                    "content": "User: {input}\\nAssistant: {generate_response.response}",
                    "metadata": {"type": "conversation", "timestamp": "2025-01-01T10:00:00Z"}
                }
            ]
        }
        
        input_data = {"input": "What is machine learning?"}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            with patch('memory.memory_factory.create_memory_service') as mock_memory:
                # Setup memory service mock
                mock_memory_service = mock_memory.return_value
                mock_memory_service.retrieve_memories.return_value = [
                    type('MockMemory', (), {
                        'content': 'Machine learning is a subset of AI',
                        'similarity': 0.9
                    })(),
                    type('MockMemory', (), {
                        'content': 'AI involves algorithms that learn from data',
                        'similarity': 0.8
                    })()
                ]
                mock_memory_service.store_memory.return_value = "conversation_memory_456"
                
                # Setup LLM service mock
                mock_llm_service = mock_llm.return_value
                mock_llm_service.generate.return_value = "Machine learning is indeed a subset of AI that enables computers to learn from data without being explicitly programmed."
                
                result = self.orchestrator.execute_workflow(workflow, input_data=input_data)
                
                assert result.status == "completed"
                assert len(result.step_results) == 3
                
                # Verify context retrieval
                retrieve_result = result.step_results["retrieve_context"]
                assert retrieve_result.status == "completed"
                assert len(retrieve_result.response) == 2
                
                # Verify response generation
                response_result = result.step_results["generate_response"]
                assert response_result.status == "completed"
                assert "machine learning" in response_result.response.lower()
                
                # Verify conversation storage
                storage_result = result.step_results["store_interaction"]
                assert storage_result.status == "completed"
                assert storage_result.response == "conversation_memory_456"
    
    @pytest.mark.integration
    def test_streaming_workflow(self, mock_environment_variables):
        """Test workflow with streaming response."""
        workflow = {
            "name": "StreamingWorkflow",
            "description": "Generate streaming response",
            "steps": [
                {
                    "id": "stream_step",
                    "type": "llm_stream",
                    "prompt": "Tell a short story about {topic}",
                    "provider": "auto"
                },
                {
                    "id": "tts_step",
                    "type": "tts_generate",
                    "text": "{stream_step.response}",
                    "provider": "auto"
                }
            ]
        }
        
        input_data = {"topic": "a robot learning to paint"}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            with patch('tts.tts_factory.create_tts_service') as mock_tts:
                # Setup LLM streaming mock
                mock_llm_service = mock_llm.return_value
                mock_llm_service.generate_stream.return_value = iter([
                    "Once upon a time, ",
                    "there was a robot ",
                    "who wanted to paint. ",
                    "It learned through practice ",
                    "and became quite skilled."
                ])
                
                # Setup TTS mock
                mock_tts_service = mock_tts.return_value
                mock_tts_service.generate_speech.return_value = "/tmp/story_audio.wav"
                
                result = self.orchestrator.execute_workflow(workflow, input_data=input_data)
                
                assert result.status == "completed"
                assert len(result.step_results) == 2
                
                # Verify streaming step
                stream_result = result.step_results["stream_step"]
                assert stream_result.status == "completed"
                full_story = stream_result.response
                assert "robot" in full_story
                assert "paint" in full_story
                
                # Verify TTS step
                tts_result = result.step_results["tts_step"]
                assert tts_result.status == "completed"
                assert tts_result.response == "/tmp/story_audio.wav"
    
    @pytest.mark.integration
    def test_multi_provider_workflow(self, mock_environment_variables):
        """Test workflow using multiple AI providers."""
        workflow = {
            "name": "MultiProviderWorkflow",
            "description": "Compare responses from different providers",
            "steps": [
                {
                    "id": "anthropic_step",
                    "type": "llm_prompt",
                    "prompt": "Explain quantum computing in simple terms",
                    "provider": "anthropic"
                },
                {
                    "id": "openai_step",
                    "type": "llm_prompt",
                    "prompt": "Explain quantum computing in simple terms",
                    "provider": "openai"
                },
                {
                    "id": "compare_step",
                    "type": "llm_prompt",
                    "prompt": "Compare these explanations:\\n1: {anthropic_step.response}\\n2: {openai_step.response}",
                    "provider": "auto"
                }
            ]
        }
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            def mock_create_service(provider):
                mock_service = type('MockLLMService', (), {})()
                if provider == "anthropic":
                    mock_service.generate = lambda prompt, **kwargs: "Anthropic's explanation of quantum computing..."
                elif provider == "openai":
                    mock_service.generate = lambda prompt, **kwargs: "OpenAI's explanation of quantum computing..."
                else:  # auto
                    mock_service.generate = lambda prompt, **kwargs: "Comparison: Both explanations are accurate but different in approach..."
                return mock_service
            
            mock_llm.side_effect = mock_create_service
            
            result = self.orchestrator.execute_workflow(workflow)
            
            assert result.status == "completed"
            assert len(result.step_results) == 3
            
            # Verify each provider's response
            anthropic_result = result.step_results["anthropic_step"]
            assert "Anthropic's explanation" in anthropic_result.response
            
            openai_result = result.step_results["openai_step"]
            assert "OpenAI's explanation" in openai_result.response
            
            compare_result = result.step_results["compare_step"]
            assert "Comparison" in compare_result.response
    
    @pytest.mark.integration
    def test_error_handling_workflow(self, mock_environment_variables):
        """Test workflow error handling and recovery."""
        workflow = {
            "name": "ErrorHandlingWorkflow",
            "description": "Test error handling",
            "steps": [
                {
                    "id": "normal_step",
                    "type": "llm_prompt",
                    "prompt": "This should work: {input}",
                    "provider": "auto"
                },
                {
                    "id": "error_step",
                    "type": "llm_prompt",
                    "prompt": "This will fail: {nonexistent_variable}",
                    "provider": "auto",
                    "on_error": "continue"
                },
                {
                    "id": "recovery_step",
                    "type": "llm_prompt",
                    "prompt": "Recovering from any errors above",
                    "provider": "auto"
                }
            ]
        }
        
        input_data = {"input": "test data"}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            mock_service = mock_llm.return_value
            
            def mock_generate(prompt, **kwargs):
                if "nonexistent_variable" in prompt:
                    raise ValueError("Variable not found")
                elif "This should work" in prompt:
                    return "Normal step completed successfully"
                elif "Recovering" in prompt:
                    return "Recovery step completed"
                return "Default response"
            
            mock_service.generate.side_effect = mock_generate
            
            result = self.orchestrator.execute_workflow(workflow, input_data=input_data)
            
            # Workflow should complete despite the error in step 2
            assert result.status == "completed"
            assert len(result.step_results) == 3
            
            # Normal step should succeed
            normal_result = result.step_results["normal_step"]
            assert normal_result.status == "completed"
            assert "successfully" in normal_result.response
            
            # Error step should fail
            error_result = result.step_results["error_step"]
            assert error_result.status == "failed"
            
            # Recovery step should succeed
            recovery_result = result.step_results["recovery_step"]
            assert recovery_result.status == "completed"
            assert "Recovery" in recovery_result.response
    
    @pytest.mark.integration
    def test_conditional_workflow_execution(self, mock_environment_variables):
        """Test workflow with conditional step execution."""
        workflow = {
            "name": "ConditionalWorkflow",
            "description": "Execute steps based on conditions",
            "steps": [
                {
                    "id": "sentiment_check",
                    "type": "llm_prompt",
                    "prompt": "Is this text positive or negative: '{input}'. Answer only 'positive' or 'negative'",
                    "provider": "auto"
                },
                {
                    "id": "positive_response",
                    "type": "llm_prompt",
                    "prompt": "Generate an encouraging response to: {input}",
                    "provider": "auto",
                    "condition": "{sentiment_check.response} == 'positive'"
                },
                {
                    "id": "negative_response",
                    "type": "llm_prompt",
                    "prompt": "Generate a supportive response to: {input}",
                    "provider": "auto",
                    "condition": "{sentiment_check.response} == 'negative'"
                }
            ]
        }
        
        # Test with positive input
        positive_input = {"input": "I'm so happy today!"}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            mock_service = mock_llm.return_value
            
            def mock_generate_positive(prompt, **kwargs):
                if "positive or negative" in prompt:
                    return "positive"
                elif "encouraging response" in prompt:
                    return "That's wonderful! Keep up the positive energy!"
                elif "supportive response" in prompt:
                    return "I understand, things will get better."
                return "Default response"
            
            mock_service.generate.side_effect = mock_generate_positive
            
            result = self.orchestrator.execute_workflow(workflow, input_data=positive_input)
            
            assert result.status == "completed"
            
            # Sentiment check should run
            sentiment_result = result.step_results["sentiment_check"]
            assert sentiment_result.status == "completed"
            assert sentiment_result.response == "positive"
            
            # Positive response should run
            assert "positive_response" in result.step_results
            positive_result = result.step_results["positive_response"]
            assert positive_result.status == "completed"
            assert "wonderful" in positive_result.response
            
            # Negative response should NOT run
            assert "negative_response" not in result.step_results
    
    @pytest.mark.integration
    def test_workflow_from_file(self, sample_workflow_file, mock_environment_variables):
        """Test executing workflow loaded from file."""
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            mock_service = mock_llm.return_value
            mock_service.generate.return_value = "File workflow executed successfully"
            
            result = self.orchestrator.execute_workflow_from_file(str(sample_workflow_file))
            
            assert result.status == "completed"
            assert len(result.step_results) > 0
    
    @pytest.mark.integration
    def test_large_workflow_performance(self, mock_environment_variables):
        """Test performance with a large workflow."""
        # Create a workflow with many steps
        steps = []
        for i in range(10):
            steps.append({
                "id": f"step_{i}",
                "type": "llm_prompt",
                "prompt": f"Process step {i}: {{input}}",
                "provider": "auto"
            })
        
        large_workflow = {
            "name": "LargeWorkflow",
            "description": "Workflow with many steps",
            "steps": steps
        }
        
        input_data = {"input": "test data"}
        
        with patch('llm.generation_service.create_generation_service') as mock_llm:
            mock_service = mock_llm.return_value
            mock_service.generate.return_value = "Step completed"
            
            result = self.orchestrator.execute_workflow(large_workflow, input_data=input_data)
            
            assert result.status == "completed"
            assert len(result.step_results) == 10
            
            # Verify all steps completed
            for i in range(10):
                step_id = f"step_{i}"
                assert step_id in result.step_results
                assert result.step_results[step_id].status == "completed"
            
            # Check total execution time is reasonable
            assert result.total_execution_time < 10.0  # Should complete quickly with mocks


class TestCrossServiceIntegration:
    """Test integration between different services."""
    
    @pytest.mark.integration
    def test_llm_memory_integration(self, mock_environment_variables):
        """Test integration between LLM and memory services."""
        from llm.generation_service import create_generation_service
        from memory.memory_factory import create_memory_service
        
        with patch('llm.generation_service.LLMFactory') as mock_llm_factory:
            with patch('memory.memory_factory.create_memory_service') as mock_memory_factory:
                # Setup mocks
                mock_llm_client = type('MockLLMClient', (), {})()
                mock_llm_client.generate = lambda prompt, **kwargs: f"Generated response for: {prompt}"
                
                mock_llm_factory.return_value.create_client.return_value = mock_llm_client
                
                mock_memory_service = type('MockMemoryService', (), {})()
                mock_memory_service.store_memory = lambda content, metadata: "stored_memory_123"
                mock_memory_service.retrieve_memories = lambda query, limit: [
                    type('MockMemory', (), {'content': 'Previous conversation', 'similarity': 0.9})()
                ]
                
                mock_memory_factory.return_value = mock_memory_service
                
                # Test the integration
                llm_service = create_generation_service("auto")
                memory_service = create_memory_service("auto")
                
                # Generate content
                response = llm_service.generate("What is AI?")
                assert "Generated response" in response
                
                # Store in memory
                memory_id = memory_service.store_memory(response, {"type": "qa"})
                assert memory_id == "stored_memory_123"
                
                # Retrieve from memory
                memories = memory_service.retrieve_memories("AI", limit=5)
                assert len(memories) == 1
                assert memories[0].content == "Previous conversation"
    
    @pytest.mark.integration
    def test_chat_memory_integration(self, mock_environment_variables):
        """Test integration between chat and memory services."""
        from chat.conversation_service import create_conversation_service
        
        with patch('chat.conversation_service.create_generation_service') as mock_llm:
            with patch('chat.conversation_service.create_memory_service') as mock_memory:
                # Setup mocks
                mock_llm_service = mock_llm.return_value
                mock_llm_service.generate.return_value = "I can help with that!"
                
                mock_memory_service = mock_memory.return_value
                mock_memory_service.retrieve_memories.return_value = [
                    type('MockMemory', (), {
                        'content': 'User previously asked about Python',
                        'similarity': 0.8
                    })()
                ]
                mock_memory_service.store_memory.return_value = "conversation_123"
                
                # Create conversation service
                conversation_service = create_conversation_service("auto")
                
                # Test conversation with memory
                response = conversation_service.generate_response(
                    "Can you help me with programming?",
                    use_memory=True
                )
                
                assert response == "I can help with that!"
                
                # Verify memory was queried
                mock_memory_service.retrieve_memories.assert_called_once()
                
                # Test storing conversation
                memory_id = conversation_service.store_conversation()
                assert memory_id == "conversation_123"
    
    @pytest.mark.integration 
    def test_prompt_llm_integration(self, mock_environment_variables):
        """Test integration between prompt and LLM services."""
        from prompt.prompt_service import create_prompt_service
        from llm.generation_service import create_generation_service
        
        with patch('prompt.prompt_service.FilePromptStorage') as mock_storage:
            with patch('llm.generation_service.LLMFactory') as mock_llm_factory:
                # Setup prompt storage mock
                mock_template = type('MockTemplate', (), {
                    'id': 'greeting_template',
                    'template': 'Hello {name}, welcome to {platform}!',
                    'variables': ['name', 'platform']
                })()
                
                mock_storage_instance = mock_storage.return_value
                mock_storage_instance.get_template.return_value = mock_template
                
                # Setup LLM factory mock
                mock_llm_client = type('MockLLMClient', (), {})()
                mock_llm_client.generate = lambda prompt, **kwargs: f"LLM processed: {prompt}"
                
                mock_llm_factory.return_value.create_client.return_value = mock_llm_client
                
                # Create services
                prompt_service = create_prompt_service("file")
                llm_service = create_generation_service("auto")
                
                # Test integration
                rendered_prompt = prompt_service.render_prompt(
                    'greeting_template',
                    variables={'name': 'Alice', 'platform': 'AI Assistant'}
                )
                
                assert rendered_prompt == 'Hello Alice, welcome to AI Assistant!'
                
                # Use rendered prompt with LLM
                response = llm_service.generate(rendered_prompt)
                assert "LLM processed: Hello Alice" in response