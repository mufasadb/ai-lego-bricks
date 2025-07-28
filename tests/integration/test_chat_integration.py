"""
Integration tests for Chat Services.

These tests make real API calls to record VCR cassettes.
Run with: pytest tests/integration/ --record-mode=once
"""

import os
import pytest

# Import chat services
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from chat.chat_service import (
    ChatService,
    ChatMessage,
    create_chat_service,
    quick_chat_ollama,
    quick_chat_gemini,
)
from chat.conversation_service import ConversationService


class TestChatServiceIntegration:
    """Integration tests for ChatService with real API calls."""

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_ollama_basic_chat(self, integration_env_check):
        """Test basic Ollama chat functionality."""
        # Use a simple model that's likely to be available
        service = ChatService("ollama", model="llama3.2:3b")

        response = service.chat("Hello! Can you respond with just 'Hi there!'?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "hi" in response.lower() or "hello" in response.lower()

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_ollama_chat_with_history(self, integration_env_check):
        """Test Ollama chat with conversation history."""
        service = ChatService("ollama", model="llama3.2:3b")

        # First message
        history = []
        response1 = service.chat("My name is Alice. Remember this.")
        history.append(
            ChatMessage(role="user", content="My name is Alice. Remember this.")
        )
        history.append(ChatMessage(role="assistant", content=response1))

        # Second message referencing first
        response2 = service.chat("What is my name?", chat_history=history)

        assert isinstance(response2, str)
        assert "alice" in response2.lower()

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_ollama_streaming_chat(self, integration_env_check):
        """Test Ollama streaming chat functionality."""
        service = ChatService("ollama", model="llama3.2:3b")

        response_chunks = []
        full_response = None

        for chunk in service.chat_stream(
            "Tell me a very short joke in exactly 10 words."
        ):
            response_chunks.append(chunk)
            if isinstance(chunk, str):
                full_response = (
                    chunk if full_response is None else full_response + chunk
                )

        assert len(response_chunks) > 0
        assert full_response is not None
        assert len(full_response) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_gemini_basic_chat(self, integration_env_check):
        """Test basic Gemini chat functionality."""
        if not os.getenv("GOOGLE_AI_STUDIO_KEY"):
            pytest.skip("GOOGLE_AI_STUDIO_KEY not available")

        service = ChatService("gemini", model="gemini-1.5-flash")

        response = service.chat(
            "Hello! Please respond with exactly: 'Hello, I am Gemini!'"
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "gemini" in response.lower() or "hello" in response.lower()

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_gemini_chat_with_history(self, integration_env_check):
        """Test Gemini chat with conversation history."""
        if not os.getenv("GOOGLE_AI_STUDIO_KEY"):
            pytest.skip("GOOGLE_AI_STUDIO_KEY not available")

        service = ChatService("gemini", model="gemini-1.5-flash")

        # First message
        history = []
        response1 = service.chat("I am a teacher. Remember my profession.")
        history.append(
            ChatMessage(role="user", content="I am a teacher. Remember my profession.")
        )
        history.append(ChatMessage(role="assistant", content=response1))

        # Second message referencing first
        response2 = service.chat("What is my profession?", chat_history=history)

        assert isinstance(response2, str)
        assert "teach" in response2.lower()

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_gemini_streaming_chat(self, integration_env_check):
        """Test Gemini streaming chat functionality."""
        if not os.getenv("GOOGLE_AI_STUDIO_KEY"):
            pytest.skip("GOOGLE_AI_STUDIO_KEY not available")

        service = ChatService("gemini", model="gemini-1.5-flash")

        response_chunks = []

        for chunk in service.chat_stream("Count from 1 to 5, one number per response."):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0
        # Gemini streaming currently simulates streaming, so we expect chunks
        full_response = "".join(response_chunks)
        assert len(full_response) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_chat_service_factory(self, integration_env_check):
        """Test chat service factory functions."""
        # Test Ollama factory
        ollama_service = create_chat_service("ollama", "llama3.2:3b")
        assert isinstance(ollama_service, ChatService)
        assert ollama_service.service == "ollama"

        response1 = ollama_service.chat("Say 'Factory test successful' exactly.")
        assert isinstance(response1, str)

        # Test Gemini factory if available
        if os.getenv("GOOGLE_AI_STUDIO_KEY"):
            gemini_service = create_chat_service("gemini", "gemini-1.5-flash")
            assert isinstance(gemini_service, ChatService)
            assert gemini_service.service == "gemini"

            response2 = gemini_service.chat(
                "Say 'Gemini factory test successful' exactly."
            )
            assert isinstance(response2, str)

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_quick_chat_functions(self, integration_env_check):
        """Test quick chat convenience functions."""
        # Test quick Ollama chat
        response1 = quick_chat_ollama("Respond with 'Quick Ollama works!'")
        assert isinstance(response1, str)
        assert len(response1) > 0

        # Test quick Gemini chat if available
        if os.getenv("GOOGLE_AI_STUDIO_KEY"):
            response2 = quick_chat_gemini("Respond with 'Quick Gemini works!'")
            assert isinstance(response2, str)
            assert len(response2) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_error_handling(self, integration_env_check):
        """Test error handling with invalid configurations."""
        # Test invalid service
        with pytest.raises(ValueError, match="Unsupported service"):
            ChatService("invalid_service")

        # Test invalid model (should not crash, might use default)
        service = ChatService("ollama", model="definitely_nonexistent_model")

        # This might fail or use fallback - test that it raises a reasonable error
        with pytest.raises(RuntimeError):
            service.chat("Hello")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_different_parameters(self, integration_env_check):
        """Test chat with different temperature and token settings."""
        # High temperature (more creative)
        service_creative = ChatService(
            "ollama", model="llama3.2:3b", temperature=0.9, max_tokens=50
        )
        response1 = service_creative.chat("Write a creative greeting.")

        # Low temperature (more deterministic)
        service_factual = ChatService(
            "ollama", model="llama3.2:3b", temperature=0.1, max_tokens=50
        )
        response2 = service_factual.chat("Write a creative greeting.")

        assert isinstance(response1, str)
        assert isinstance(response2, str)
        # Both should be valid responses, but potentially different due to temperature


class TestConversationServiceIntegration:
    """Integration tests for ConversationService with real API calls."""

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_service_basic(self, integration_env_check):
        """Test basic conversation service functionality."""
        conversation = ConversationService("ollama", model="llama3.2:3b")

        # Add system message
        conversation.add_system_message(
            "You are a helpful assistant that gives very brief answers."
        )

        # Send first message
        response1 = conversation.send_message("What is 2+2?")
        assert isinstance(response1, str)
        assert "4" in response1

        # Check conversation state
        assert len(conversation.messages) >= 3  # system + user + assistant

        # Send follow-up message
        response2 = conversation.send_message("What about 3+3?")
        assert isinstance(response2, str)
        assert "6" in response2

        # Check conversation history grew
        assert (
            len(conversation.messages) >= 5
        )  # system + user1 + assistant1 + user2 + assistant2

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_memory_context(self, integration_env_check):
        """Test that conversation maintains context."""
        conversation = ConversationService("ollama", model="llama3.2:3b")

        # Establish context
        response1 = conversation.send_message(
            "My favorite color is blue. Remember this."
        )
        assert isinstance(response1, str)

        # Test if context is remembered
        response2 = conversation.send_message("What is my favorite color?")
        assert isinstance(response2, str)
        assert "blue" in response2.lower()

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_with_history_limit(self, integration_env_check):
        """Test conversation service with limited history."""
        # Create conversation with small history limit
        conversation = ConversationService("ollama", model="llama3.2:3b", max_history=5)

        # Add multiple messages to exceed limit
        for i in range(10):
            response = conversation.send_message(f"Message number {i}")
            assert isinstance(response, str)

        # Check that history was trimmed
        assert len(conversation.messages) <= 5

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_statistics(self, integration_env_check):
        """Test conversation analytics and statistics."""
        conversation = ConversationService("ollama", model="llama3.2:3b")

        # Add some messages
        conversation.send_message("Hello")
        conversation.send_message("How are you?")
        conversation.send_message("Tell me a joke")

        # Get statistics
        stats = conversation.get_conversation_stats()

        assert isinstance(stats, dict)
        assert "total_messages" in stats
        assert "user_messages" in stats
        assert "ai_messages" in stats
        assert stats["total_messages"] >= 6  # 3 user + 3 assistant
        assert stats["user_messages"] == 3
        assert stats["ai_messages"] == 3

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_search(self, integration_env_check):
        """Test conversation message search functionality."""
        conversation = ConversationService("ollama", model="llama3.2:3b")

        # Add messages with searchable content
        conversation.send_message("I love pizza and Italian food")
        conversation.send_message("My hobby is reading books")
        conversation.send_message("I work as a software engineer")

        # Search for specific content
        pizza_messages = conversation.search_messages("pizza")
        assert len(pizza_messages) >= 1
        assert any("pizza" in msg.content.lower() for msg in pizza_messages)

        # Search by role
        user_messages = conversation.search_messages("", role="user")
        assert len(user_messages) == 3
        assert all(msg.role == "user" for msg in user_messages)

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_conversation_export(self, integration_env_check):
        """Test conversation export functionality."""
        conversation = ConversationService("ollama", model="llama3.2:3b")

        # Add some conversation content
        conversation.send_message("Hello assistant")
        conversation.send_message("What's the weather like?")

        # Test JSON export
        json_export = conversation.export_conversation("json")
        assert isinstance(json_export, str)
        assert "hello" in json_export.lower()

        # Test Markdown export
        md_export = conversation.export_conversation("markdown")
        assert isinstance(md_export, str)
        assert "**User:**" in md_export or "**Assistant:**" in md_export

        # Test CSV export
        csv_export = conversation.export_conversation("csv")
        assert isinstance(csv_export, str)
        assert "Role,Content" in csv_export


class TestChatServiceEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_empty_message_handling(self, integration_env_check):
        """Test handling of empty or whitespace messages."""
        service = ChatService("ollama", model="llama3.2:3b")

        # Test empty string
        with pytest.raises((ValueError, RuntimeError)):
            service.chat("")

        # Test whitespace only
        with pytest.raises((ValueError, RuntimeError)):
            service.chat("   ")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_very_long_message(self, integration_env_check):
        """Test handling of very long messages."""
        service = ChatService("ollama", model="llama3.2:3b", max_tokens=100)

        # Create a long message
        long_message = (
            "Please respond with 'Got it'. " + "This is a very long message. " * 100
        )

        # Should handle gracefully (might truncate or summarize)
        response = service.chat(long_message)
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_special_characters(self, integration_env_check):
        """Test handling of special characters and unicode."""
        service = ChatService("ollama", model="llama3.2:3b")

        # Test with emojis and special characters
        message = "Hello! 🤖 Can you respond to this message with émojis and spëcial chars? 中文"
        response = service.chat(message)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_concurrent_requests(self, integration_env_check):
        """Test multiple concurrent chat requests."""
        import concurrent.futures

        service = ChatService("ollama", model="llama3.2:3b")

        def make_chat_request(message_id):
            return service.chat(f"This is message {message_id}. Please acknowledge it.")

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_chat_request, i) for i in range(3)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        assert len(results) == 3
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
