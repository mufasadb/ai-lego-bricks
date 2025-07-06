"""
Unit tests for chat and conversation services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from chat.conversation_service import ConversationService, create_conversation
from chat.chat_service import ChatService
from llm.llm_types import GenerationConfig


class TestConversationService:
    """Test suite for ConversationService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_memory_service = Mock()
        
        self.service = ConversationService(
            llm_service=self.mock_llm_service,
            memory_service=self.mock_memory_service
        )
        
        self.sample_conversation = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            {"role": "user", "content": "Can you explain quantum computing?"}
        ]
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.llm_service == self.mock_llm_service
        assert self.service.memory_service == self.mock_memory_service
        assert hasattr(self.service, 'conversation_history')
        assert len(self.service.conversation_history) == 0
    
    def test_add_message_to_conversation(self):
        """Test adding messages to conversation."""
        # Add user message
        self.service.add_message("user", "Hello")
        assert len(self.service.conversation_history) == 1
        assert self.service.conversation_history[0]["role"] == "user"
        assert self.service.conversation_history[0]["content"] == "Hello"
        
        # Add assistant message
        self.service.add_message("assistant", "Hi there!")
        assert len(self.service.conversation_history) == 2
        assert self.service.conversation_history[1]["role"] == "assistant"
    
    def test_generate_response(self):
        """Test generating a response."""
        self.mock_llm_service.generate.return_value = "This is a test response."
        
        response = self.service.generate_response("Hello, how are you?")
        
        assert response == "This is a test response."
        assert len(self.service.conversation_history) == 2  # User message + assistant response
        
        # Verify LLM service was called
        self.mock_llm_service.generate.assert_called_once()
        call_args = self.mock_llm_service.generate.call_args[0]
        assert "Hello, how are you?" in str(call_args)
    
    def test_generate_response_with_system_prompt(self):
        """Test generating response with system prompt."""
        system_prompt = "You are a helpful assistant."
        self.mock_llm_service.generate.return_value = "I'm here to help!"
        
        response = self.service.generate_response(
            "Hello", 
            system_prompt=system_prompt
        )
        
        assert response == "I'm here to help!"
        self.mock_llm_service.generate.assert_called_once()
        
        # Verify system prompt was included
        call_args = self.mock_llm_service.generate.call_args
        assert system_prompt in str(call_args)
    
    def test_generate_streaming_response(self):
        """Test generating streaming response."""
        self.mock_llm_service.generate_stream.return_value = iter(["Hello", " there", "!"])
        
        stream = self.service.generate_streaming_response("Hi")
        chunks = list(stream)
        
        assert chunks == ["Hello", " there", "!"]
        assert len(self.service.conversation_history) == 2  # User + assistant messages
        
        # Check that the complete response was stored
        assert self.service.conversation_history[1]["content"] == "Hello there!"
    
    def test_conversation_with_memory_retrieval(self):
        """Test conversation with memory context."""
        # Setup memory service mock
        mock_memories = [
            Mock(content="Previous context about AI", similarity=0.9),
            Mock(content="User prefers detailed explanations", similarity=0.8)
        ]
        self.mock_memory_service.retrieve_memories.return_value = mock_memories
        self.mock_llm_service.generate.return_value = "Based on context, here's my response."
        
        response = self.service.generate_response(
            "Tell me about machine learning",
            use_memory=True,
            memory_limit=5
        )
        
        assert response == "Based on context, here's my response."
        
        # Verify memory was retrieved
        self.mock_memory_service.retrieve_memories.assert_called_once_with(
            "Tell me about machine learning", 
            limit=5
        )
        
        # Verify context was included in LLM call
        call_args = self.mock_llm_service.generate.call_args[0]
        prompt_text = str(call_args)
        assert "Previous context about AI" in prompt_text
        assert "User prefers detailed explanations" in prompt_text
    
    def test_store_conversation_in_memory(self):
        """Test storing conversation in memory."""
        self.mock_memory_service.store_memory.return_value = "memory_id_123"
        
        # Add some conversation history
        self.service.conversation_history = self.sample_conversation.copy()
        
        memory_id = self.service.store_conversation()
        
        assert memory_id == "memory_id_123"
        self.mock_memory_service.store_memory.assert_called_once()
        
        # Verify the conversation was formatted correctly for storage
        call_args = self.mock_memory_service.store_memory.call_args
        stored_content = call_args[0][0]
        assert "user:" in stored_content.lower()
        assert "assistant:" in stored_content.lower()
    
    def test_clear_conversation(self):
        """Test clearing conversation history."""
        # Add some messages
        self.service.conversation_history = self.sample_conversation.copy()
        assert len(self.service.conversation_history) > 0
        
        self.service.clear_conversation()
        assert len(self.service.conversation_history) == 0
    
    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        self.mock_llm_service.generate.return_value = "This conversation was about greetings and quantum computing."
        
        # Add conversation history
        self.service.conversation_history = self.sample_conversation.copy()
        
        summary = self.service.get_conversation_summary()
        
        assert summary == "This conversation was about greetings and quantum computing."
        self.mock_llm_service.generate.assert_called_once()
        
        # Verify summarization prompt was used
        call_args = self.mock_llm_service.generate.call_args[0]
        prompt_text = str(call_args)
        assert "summarize" in prompt_text.lower() or "summary" in prompt_text.lower()
    
    def test_conversation_length_management(self):
        """Test automatic conversation length management."""
        # Create a service with conversation limit
        service = ConversationService(
            llm_service=self.mock_llm_service,
            memory_service=self.mock_memory_service,
            max_conversation_length=4
        )
        
        # Add more messages than the limit
        for i in range(6):
            service.add_message("user", f"Message {i}")
        
        # Should only keep the most recent messages
        assert len(service.conversation_history) <= 4
        assert service.conversation_history[-1]["content"] == "Message 5"
    
    def test_error_handling_in_generation(self):
        """Test error handling during response generation."""
        self.mock_llm_service.generate.side_effect = Exception("LLM API Error")
        
        with pytest.raises(Exception, match="LLM API Error"):
            self.service.generate_response("Hello")
        
        # Verify conversation state is preserved
        assert len(self.service.conversation_history) == 1  # Only user message added
        assert self.service.conversation_history[0]["role"] == "user"
    
    def test_conversation_with_config(self):
        """Test conversation with generation config."""
        config = GenerationConfig(temperature=0.3, max_tokens=500)
        self.mock_llm_service.generate.return_value = "Configured response"
        
        response = self.service.generate_response("Hello", config=config)
        
        assert response == "Configured response"
        self.mock_llm_service.generate.assert_called_once()
        
        # Verify config was passed
        call_args = self.mock_llm_service.generate.call_args
        assert call_args.kwargs.get('config') == config


class TestChatService:
    """Test suite for ChatService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_service = Mock()
        self.service = ChatService(self.mock_llm_service)
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.llm_service == self.mock_llm_service
    
    def test_simple_chat(self):
        """Test simple chat without conversation history."""
        self.mock_llm_service.generate.return_value = "Hello! How can I help you?"
        
        response = self.service.chat("Hi there")
        
        assert response == "Hello! How can I help you?"
        self.mock_llm_service.generate.assert_called_once_with("Hi there", config=None)
    
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        system_prompt = "You are a coding assistant."
        self.mock_llm_service.generate.return_value = "I can help with coding!"
        
        response = self.service.chat("Help me code", system_prompt=system_prompt)
        
        assert response == "I can help with coding!"
        self.mock_llm_service.generate.assert_called_once()
        
        # Verify system prompt was included
        call_args = self.mock_llm_service.generate.call_args
        assert system_prompt in str(call_args)
    
    def test_streaming_chat(self):
        """Test streaming chat response."""
        self.mock_llm_service.generate_stream.return_value = iter(["Streaming", " response", "!"])
        
        stream = self.service.chat_stream("Tell me a story")
        chunks = list(stream)
        
        assert chunks == ["Streaming", " response", "!"]
        self.mock_llm_service.generate_stream.assert_called_once()
    
    def test_chat_with_config(self):
        """Test chat with generation configuration."""
        config = GenerationConfig(temperature=0.9, max_tokens=200)
        self.mock_llm_service.generate.return_value = "Creative response"
        
        response = self.service.chat("Be creative", config=config)
        
        assert response == "Creative response"
        self.mock_llm_service.generate.assert_called_once_with("Be creative", config=config)


class TestCreateConversationService:
    """Test suite for create_conversation factory function."""
    
    @patch('chat.conversation_service.create_generation_service')
    @patch('chat.conversation_service.create_memory_service')
    def test_create_service_auto(self, mock_memory_factory, mock_llm_factory):
        """Test creating service with auto provider."""
        mock_llm = Mock()
        mock_memory = Mock()
        mock_llm_factory.return_value = mock_llm
        mock_memory_factory.return_value = mock_memory
        
        service = create_conversation("auto")
        
        assert isinstance(service, ConversationService)
        assert service.llm_service == mock_llm
        assert service.memory_service == mock_memory
        
        mock_llm_factory.assert_called_once_with("auto")
        mock_memory_factory.assert_called_once_with("auto")
    
    @patch('chat.conversation_service.create_generation_service')
    @patch('chat.conversation_service.create_memory_service')
    def test_create_service_specific_providers(self, mock_memory_factory, mock_llm_factory):
        """Test creating service with specific providers."""
        mock_llm = Mock()
        mock_memory = Mock()
        mock_llm_factory.return_value = mock_llm
        mock_memory_factory.return_value = mock_memory
        
        service = create_conversation(
            llm_provider="anthropic",
            memory_provider="supabase"
        )
        
        assert isinstance(service, ConversationService)
        mock_llm_factory.assert_called_once_with("anthropic")
        mock_memory_factory.assert_called_once_with("supabase")
    
    @patch('chat.conversation_service.create_generation_service')
    def test_create_service_without_memory(self, mock_llm_factory):
        """Test creating service without memory."""
        mock_llm = Mock()
        mock_llm_factory.return_value = mock_llm
        
        service = create_conversation(llm_provider="anthropic", use_memory=False)
        
        assert isinstance(service, ConversationService)
        assert service.llm_service == mock_llm
        assert service.memory_service is None
        
        mock_llm_factory.assert_called_once_with("anthropic")
    
    def test_create_service_invalid_provider(self):
        """Test creating service with invalid provider."""
        with pytest.raises(ValueError, match="Invalid provider"):
            create_conversation("invalid_provider")


class TestConversationFeatures:
    """Test suite for advanced conversation features."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_memory_service = Mock()
        
        self.service = ConversationService(
            llm_service=self.mock_llm_service,
            memory_service=self.mock_memory_service
        )
    
    def test_conversation_personas(self):
        """Test conversation with different personas."""
        personas = {
            "helpful": "You are a helpful assistant.",
            "technical": "You are a technical expert.",
            "creative": "You are a creative writing assistant."
        }
        
        self.mock_llm_service.generate.return_value = "Persona-specific response"
        
        for persona_name, persona_prompt in personas.items():
            response = self.service.generate_response(
                "Hello",
                system_prompt=persona_prompt
            )
            
            assert response == "Persona-specific response"
            
            # Verify persona was applied
            call_args = self.mock_llm_service.generate.call_args
            assert persona_prompt in str(call_args)
    
    def test_conversation_context_window(self):
        """Test conversation context window management."""
        # Create service with small context window
        service = ConversationService(
            llm_service=self.mock_llm_service,
            memory_service=self.mock_memory_service,
            context_window=1000  # Small window for testing
        )
        
        # Add a very long conversation
        long_message = "This is a very long message. " * 100  # ~3000 characters
        
        service.add_message("user", long_message)
        service.add_message("assistant", long_message)
        service.add_message("user", "Short message")
        
        self.mock_llm_service.generate.return_value = "Response"
        
        service.generate_response("Another message")
        
        # Verify context was truncated
        call_args = self.mock_llm_service.generate.call_args[0]
        prompt_text = str(call_args)
        assert len(prompt_text) <= 1200  # Some buffer for formatting
    
    def test_conversation_with_metadata(self):
        """Test storing conversation with metadata."""
        self.mock_memory_service.store_memory.return_value = "memory_id_123"
        
        # Add conversation with metadata
        self.service.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        metadata = {
            "user_id": "user123",
            "session_id": "session456",
            "topic": "greeting"
        }
        
        memory_id = self.service.store_conversation(metadata=metadata)
        
        assert memory_id == "memory_id_123"
        
        # Verify metadata was included
        call_args = self.mock_memory_service.store_memory.call_args
        stored_metadata = call_args[0][1]
        assert stored_metadata["user_id"] == "user123"
        assert stored_metadata["session_id"] == "session456"
        assert stored_metadata["topic"] == "greeting"
    
    def test_conversation_emotions_detection(self):
        """Test detecting emotions in conversation."""
        self.mock_llm_service.generate.side_effect = [
            "positive",  # Emotion detection response
            "I'm glad you're feeling positive!"  # Main response
        ]
        
        response = self.service.generate_response_with_emotion_detection(
            "I'm so happy today!"
        )
        
        assert response == "I'm glad you're feeling positive!"
        assert self.mock_llm_service.generate.call_count == 2
        
        # Verify emotion detection was called first
        first_call = self.mock_llm_service.generate.call_args_list[0][0]
        assert "emotion" in str(first_call).lower()
    
    def test_conversation_export(self):
        """Test exporting conversation history."""
        self.service.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
        
        # Test JSON export
        json_export = self.service.export_conversation("json")
        assert isinstance(json_export, str)
        assert "Hello" in json_export
        assert "Hi there!" in json_export
        
        # Test markdown export
        md_export = self.service.export_conversation("markdown")
        assert isinstance(md_export, str)
        assert "**User:**" in md_export or "**Assistant:**" in md_export
        
        # Test plain text export
        txt_export = self.service.export_conversation("text")
        assert isinstance(txt_export, str)
        assert "Hello" in txt_export
    
    def test_conversation_search(self):
        """Test searching conversation history."""
        self.service.conversation_history = [
            {"role": "user", "content": "Tell me about Python programming"},
            {"role": "assistant", "content": "Python is a versatile programming language"},
            {"role": "user", "content": "What about JavaScript?"},
            {"role": "assistant", "content": "JavaScript is great for web development"}
        ]
        
        # Search for Python-related messages
        python_messages = self.service.search_conversation("Python")
        assert len(python_messages) == 2  # User question + assistant response
        assert "Python" in python_messages[0]["content"]
        assert "Python" in python_messages[1]["content"]
        
        # Search for JavaScript-related messages
        js_messages = self.service.search_conversation("JavaScript")
        assert len(js_messages) == 2
        assert "JavaScript" in js_messages[0]["content"]