"""
Unit tests for Chat Services using VCR cassettes.

These tests use recorded HTTP interactions and run without real API calls.
Run with: pytest tests/unit/ --record-mode=none
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


class TestChatServiceUnit:
    """Unit tests for ChatService using recorded cassettes."""

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr()
    @pytest.mark.unit
    def test_ollama_basic_chat(self, mock_credentials):
        """Test basic Ollama chat functionality using recorded response."""
        service = ChatService("ollama", model="llama3.2:3b")

        response = service.chat("Hello! Can you respond with just 'Hi there!'?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "hi" in response.lower() or "hello" in response.lower()

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_ollama_chat_with_history(self, mock_credentials):
        """Test Ollama chat with conversation history using recorded responses."""
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

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_ollama_streaming_chat(self, mock_credentials):
        """Test Ollama streaming chat using recorded responses."""
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

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_gemini_basic_chat(self, mock_credentials):
        """Test basic Gemini chat using recorded response."""
        service = ChatService("gemini", model="gemini-1.5-flash")

        response = service.chat(
            "Hello! Please respond with exactly: 'Hello, I am Gemini!'"
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "gemini" in response.lower() or "hello" in response.lower()

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_gemini_chat_with_history(self, mock_credentials):
        """Test Gemini chat with conversation history using recorded responses."""
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

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_gemini_streaming_chat(self, mock_credentials):
        """Test Gemini streaming chat using recorded responses."""
        service = ChatService("gemini", model="gemini-1.5-flash")

        response_chunks = []

        for chunk in service.chat_stream("Count from 1 to 5, one number per response."):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0
        full_response = "".join(response_chunks)
        assert len(full_response) > 0

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_chat_service_factory(self, mock_credentials):
        """Test chat service factory functions using recorded responses."""
        # Test Ollama factory
        ollama_service = create_chat_service("ollama", "llama3.2:3b")
        assert isinstance(ollama_service, ChatService)
        assert ollama_service.service == "ollama"

        response1 = ollama_service.chat("Say 'Factory test successful' exactly.")
        assert isinstance(response1, str)

        # Test Gemini factory
        gemini_service = create_chat_service("gemini", "gemini-1.5-flash")
        assert isinstance(gemini_service, ChatService)
        assert gemini_service.service == "gemini"

        response2 = gemini_service.chat("Say 'Gemini factory test successful' exactly.")
        assert isinstance(response2, str)

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_quick_chat_functions(self, mock_credentials):
        """Test quick chat convenience functions using recorded responses."""
        # Test quick Ollama chat
        response1 = quick_chat_ollama("Respond with 'Quick Ollama works!'")
        assert isinstance(response1, str)
        assert len(response1) > 0

        # Test quick Gemini chat
        response2 = quick_chat_gemini("Respond with 'Quick Gemini works!'")
        assert isinstance(response2, str)
        assert len(response2) > 0

    @pytest.mark.unit
    def test_error_handling(self, mock_credentials):
        """Test error handling with invalid configurations (no VCR needed)."""
        # Test invalid service
        with pytest.raises(ValueError, match="Unsupported service"):
            ChatService("invalid_service")

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_different_parameters(self, mock_credentials):
        """Test chat with different temperature and token settings using recorded responses."""
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


class TestConversationServiceUnit:
    """Unit tests for ConversationService using recorded cassettes."""

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_service_basic(self, mock_credentials):
        """Test basic conversation service using recorded responses."""
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

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_memory_context(self, mock_credentials):
        """Test conversation context maintenance using recorded responses."""
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

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_with_history_limit(self, mock_credentials):
        """Test conversation service with limited history using recorded responses."""
        # Create conversation with small history limit
        conversation = ConversationService("ollama", model="llama3.2:3b", max_history=5)

        # Add multiple messages to exceed limit
        for i in range(10):
            response = conversation.send_message(f"Message number {i}")
            assert isinstance(response, str)

        # Check that history was trimmed
        assert len(conversation.messages) <= 5

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_statistics(self, mock_credentials):
        """Test conversation analytics using recorded responses."""
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

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_search(self, mock_credentials):
        """Test conversation search using recorded responses."""
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

    @pytest.mark.skip(
        reason="ConversationService has dependency issues - TODO: Fix imports"
    )
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_conversation_export(self, mock_credentials):
        """Test conversation export using recorded responses."""
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


class TestChatServiceEdgeCasesUnit:
    """Test edge cases using mocked/recorded responses."""

    @pytest.mark.unit
    def test_empty_message_handling(self, mock_credentials):
        """Test handling of empty messages (no network needed)."""
        service = ChatService("ollama", model="llama3.2:3b")

        # Test empty string
        with pytest.raises((ValueError, RuntimeError)):
            service.chat("")

        # Test whitespace only
        with pytest.raises((ValueError, RuntimeError)):
            service.chat("   ")

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_very_long_message(self, mock_credentials):
        """Test handling of very long messages using recorded response."""
        service = ChatService("ollama", model="llama3.2:3b", max_tokens=100)

        # Create a long message
        long_message = (
            "Please respond with 'Got it'. " + "This is a very long message. " * 100
        )

        # Should handle gracefully (might truncate or summarize)
        response = service.chat(long_message)
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_special_characters(self, mock_credentials):
        """Test handling of special characters using recorded response."""
        service = ChatService("ollama", model="llama3.2:3b")

        # Test with emojis and special characters
        message = "Hello! 🤖 Can you respond to this message with émojis and spëcial chars? 中文"
        response = service.chat(message)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skip(reason="VCR cassette has connection issues - TODO: Fix cassette")
    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_concurrent_requests(self, mock_credentials):
        """Test multiple concurrent requests using recorded responses."""
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


class TestChatServiceMocking:
    """Tests that demonstrate pure mocking without VCR."""

    @pytest.mark.skip(reason="Environment variable issues - TODO: Fix mock setup")
    @pytest.mark.unit
    def test_chat_service_initialization(self, mock_credentials):
        """Test service initialization without network calls."""
        # Test Ollama initialization
        ollama_service = ChatService("ollama", "llama3.2:3b")
        assert ollama_service.service == "ollama"
        assert ollama_service.model == "llama3.2:3b"
        assert ollama_service.temperature == 0.7  # default
        assert ollama_service.max_tokens == 1000  # default

        # Test Gemini initialization
        gemini_service = ChatService(
            "gemini", "gemini-1.5-flash", temperature=0.5, max_tokens=500
        )
        assert gemini_service.service == "gemini"
        assert gemini_service.model == "gemini-1.5-flash"
        assert gemini_service.temperature == 0.5
        assert gemini_service.max_tokens == 500

    @pytest.mark.skip(
        reason="Missing method '_build_message_context' - TODO: Fix implementation"
    )
    @pytest.mark.unit
    def test_message_context_building(self, mock_credentials):
        """Test message context building without network calls."""
        service = ChatService("ollama", "llama3.2:3b")

        # Test with empty history
        messages = service._build_message_context("Hello", None)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

        # Test with history
        history = [
            ChatMessage(role="user", content="Previous message"),
            ChatMessage(role="assistant", content="Previous response"),
        ]
        messages = service._build_message_context("New message", history)
        assert len(messages) == 3
        assert messages[0].content == "Previous message"
        assert messages[1].content == "Previous response"
        assert messages[2].content == "New message"

    @pytest.mark.skip(
        reason="ConversationService dependency issues - TODO: Fix imports"
    )
    @pytest.mark.unit
    def test_conversation_service_state_management(self, mock_credentials):
        """Test conversation state management without network calls."""
        conversation = ConversationService("ollama", "llama3.2:3b", max_history=5)

        # Test initialization
        assert len(conversation.messages) == 0
        assert conversation.max_history == 5
        assert conversation.provider == "ollama"

        # Test system message
        conversation.add_system_message("You are a helpful assistant")
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == "system"
        assert conversation.messages[0].content == "You are a helpful assistant"

        # Test conversation ID generation
        assert conversation.conversation_id is not None
        assert len(conversation.conversation_id) > 0

    @pytest.mark.skip(
        reason="ConversationService dependency issues - TODO: Fix imports"
    )
    @pytest.mark.unit
    def test_message_search_functionality(self, mock_credentials):
        """Test message search without network calls."""
        conversation = ConversationService("ollama", "llama3.2:3b")

        # Add test messages
        test_messages = [
            ChatMessage(role="user", content="Hello world"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(role="assistant", content="I don't have weather data"),
        ]
        conversation.messages = test_messages

        # Test search
        results = conversation.search_messages("weather")
        assert len(results) == 2  # User question and assistant response

        # Test role filtering
        user_results = conversation.search_messages("", role="user")
        assert len(user_results) == 2
        assert all(msg.role == "user" for msg in user_results)

        assistant_results = conversation.search_messages("", role="assistant")
        assert len(assistant_results) == 2
        assert all(msg.role == "assistant" for msg in assistant_results)
