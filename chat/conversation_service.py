"""
Enhanced conversation service for managing multi-turn conversations with rich state management.
Provides access to conversation history, context, and state for agent orchestration.
"""

from typing import List, Optional, Dict, Any, Generator
from datetime import datetime
from pydantic import BaseModel


# Import LLM abstraction layer
try:
    from llm.llm_factory import LLMClientFactory
    from llm.llm_types import LLMProvider
except ImportError:
    # Fallback for when running as part of the package
    try:
        from llm.llm_factory import LLMClientFactory
        from llm.llm_types import LLMProvider
    except ImportError:
        # If LLM factory not available, we'll handle it gracefully
        LLMClientFactory = None
        LLMProvider = None


class ConversationMessage(BaseModel):
    """Individual message in a conversation"""

    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationState(BaseModel):
    """Complete conversation state with rich access methods"""

    id: str
    messages: List[ConversationMessage] = []
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}


class ConversationService:
    """
    Enhanced conversation service with rich state management and history access.
    Designed for multi-turn conversations with full context tracking.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        conversation_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the conversation service.

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            conversation_id: Optional existing conversation ID to continue
            **kwargs: Additional configuration parameters
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        # Create the underlying LLM client
        self.client = LLMClientFactory.create_text_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Initialize conversation state
        self.conversation = ConversationState(
            id=conversation_id or self._generate_conversation_id(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID"""
        return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """
        Add a message to the conversation.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata for the message

        Returns:
            The created ConversationMessage
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self.conversation.messages.append(message)
        self.conversation.updated_at = datetime.now()

        return message

    def send_message(
        self, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a user message and get an assistant response.

        Args:
            message: User message
            metadata: Optional metadata for the user message

        Returns:
            Assistant response
        """
        # Add user message
        self.add_message("user", message, metadata)

        # Prepare conversation context for LLM
        conversation_context = self._build_conversation_context()

        # Get response from LLM
        response = self.client.chat(conversation_context)

        # Add assistant response
        self.add_message("assistant", response)

        return response

    def send_message_stream(
        self, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, str]:
        """
        Send a user message and get a streaming assistant response.

        Args:
            message: User message
            metadata: Optional metadata for the user message

        Yields:
            str: Partial response chunks as they arrive

        Returns:
            str: Complete assistant response
        """
        # Add user message
        self.add_message("user", message, metadata)

        # Prepare conversation context for LLM
        conversation_context = self._build_conversation_context()

        # Get streaming response from LLM
        full_response = ""
        for chunk in self.client.chat_stream(conversation_context):
            full_response += chunk
            yield chunk

        # Add assistant response
        self.add_message("assistant", full_response)

        return full_response

    def _build_conversation_context(self) -> str:
        """Build the conversation context for the LLM"""
        context_parts = []

        for message in self.conversation.messages:
            if message.role == "system":
                context_parts.append(f"System: {message.content}")
            elif message.role == "user":
                context_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                context_parts.append(f"Assistant: {message.content}")

        return "\n\n".join(context_parts) + "\n\nAssistant:"

    # Rich conversation access methods
    def get_first_prompt(self) -> Optional[str]:
        """Get the first user message in the conversation"""
        for message in self.conversation.messages:
            if message.role == "user":
                return message.content
        return None

    def get_last_response(self) -> Optional[str]:
        """Get the most recent assistant response"""
        for message in reversed(self.conversation.messages):
            if message.role == "assistant":
                return message.content
        return None

    def get_conversation_summary(self) -> str:
        """Get the entire conversation as a single string"""
        summary_parts = []

        for message in self.conversation.messages:
            timestamp_str = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            summary_parts.append(
                f"[{timestamp_str}] {message.role.title()}: {message.content}"
            )

        return "\n".join(summary_parts)

    def get_message_by_index(self, index: int) -> Optional[ConversationMessage]:
        """Get a specific message by index (0-based)"""
        if 0 <= index < len(self.conversation.messages):
            return self.conversation.messages[index]
        return None

    def get_user_messages(self) -> List[ConversationMessage]:
        """Get all user messages"""
        return [msg for msg in self.conversation.messages if msg.role == "user"]

    def get_assistant_messages(self) -> List[ConversationMessage]:
        """Get all assistant messages"""
        return [msg for msg in self.conversation.messages if msg.role == "assistant"]

    def get_system_messages(self) -> List[ConversationMessage]:
        """Get all system messages"""
        return [msg for msg in self.conversation.messages if msg.role == "system"]

    def get_messages_since(self, timestamp: datetime) -> List[ConversationMessage]:
        """Get all messages since a specific timestamp"""
        return [msg for msg in self.conversation.messages if msg.timestamp >= timestamp]

    def get_recent_messages(self, count: int) -> List[ConversationMessage]:
        """Get the most recent N messages"""
        return self.conversation.messages[-count:] if count > 0 else []

    def get_conversation_length(self) -> int:
        """Get the total number of messages in the conversation"""
        return len(self.conversation.messages)

    def get_message_count_by_role(self, role: str) -> int:
        """Get the count of messages by role"""
        return len([msg for msg in self.conversation.messages if msg.role == role])

    def search_messages(
        self, query: str, role: Optional[str] = None
    ) -> List[ConversationMessage]:
        """Search for messages containing specific text"""
        results = []
        for message in self.conversation.messages:
            if role and message.role != role:
                continue
            if query.lower() in message.content.lower():
                results.append(message)
        return results

    # Conversation management methods
    def add_system_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """Add a system message to the conversation"""
        return self.add_message("system", content, metadata)

    def clear_conversation(self):
        """Clear all messages from the conversation"""
        self.conversation.messages.clear()
        self.conversation.updated_at = datetime.now()

    def export_conversation(self, format: str = "json") -> str:
        """
        Export the conversation in various formats.

        Args:
            format: Export format ('json', 'text', 'markdown')

        Returns:
            Exported conversation string
        """
        if format == "json":
            return self.conversation.model_dump_json(indent=2)
        elif format == "text":
            return self.get_conversation_summary()
        elif format == "markdown":
            return self._export_as_markdown()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_as_markdown(self) -> str:
        """Export conversation as markdown"""
        markdown_parts = [
            f"# Conversation {self.conversation.id}",
            f"**Created:** {self.conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Updated:** {self.conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Messages:** {len(self.conversation.messages)}",
            "",
            "## Messages",
            "",
        ]

        for i, message in enumerate(self.conversation.messages):
            role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(
                message.role, "❓"
            )

            timestamp_str = message.timestamp.strftime("%H:%M:%S")
            markdown_parts.append(
                f"### {role_emoji} {message.role.title()} ({timestamp_str})"
            )
            markdown_parts.append(f"{message.content}")
            markdown_parts.append("")

        return "\n".join(markdown_parts)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        messages = self.conversation.messages
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m.role == "user"]),
            "assistant_messages": len([m for m in messages if m.role == "assistant"]),
            "system_messages": len([m for m in messages if m.role == "system"]),
            "conversation_id": self.conversation.id,
            "created_at": self.conversation.created_at.isoformat(),
            "updated_at": self.conversation.updated_at.isoformat(),
            "duration_minutes": (
                self.conversation.updated_at - self.conversation.created_at
            ).total_seconds()
            / 60,
        }


# Convenience functions for quick conversation creation
def create_conversation(
    provider: LLMProvider, model: Optional[str] = None, **kwargs
) -> ConversationService:
    """
    Create a new conversation service.

    Args:
        provider: LLM provider to use
        model: Model name (optional)
        **kwargs: Additional configuration

    Returns:
        ConversationService instance
    """
    return ConversationService(provider, model, **kwargs)


def create_ollama_conversation(
    model: Optional[str] = None, **kwargs
) -> ConversationService:
    """Create a conversation service using Ollama"""
    return create_conversation(LLMProvider.OLLAMA, model, **kwargs)


def create_gemini_conversation(
    model: Optional[str] = None, **kwargs
) -> ConversationService:
    """Create a conversation service using Gemini"""
    return create_conversation(LLMProvider.GEMINI, model, **kwargs)


def create_anthropic_conversation(
    model: Optional[str] = None, **kwargs
) -> ConversationService:
    """Create a conversation service using Anthropic"""
    return create_conversation(LLMProvider.ANTHROPIC, model, **kwargs)


# Example usage
if __name__ == "__main__":
    print("=== Conversation Service Example ===")

    # Create a conversation
    conv = create_gemini_conversation(temperature=0.7)

    # Add a system message
    conv.add_system_message("You are a helpful assistant specializing in programming.")

    # Have a conversation
    response1 = conv.send_message("What is Python?")
    print(f"Response 1: {response1}")

    response2 = conv.send_message("How do I create a list in Python?")
    print(f"Response 2: {response2}")

    # Demonstrate rich access methods
    print(f"\nFirst prompt: {conv.get_first_prompt()}")
    print(f"Last response: {conv.get_last_response()}")
    print(f"Total messages: {conv.get_conversation_length()}")

    # Get conversation statistics
    stats = conv.get_conversation_stats()
    print(f"\nConversation stats: {stats}")

    # Export conversation
    print("\n=== Conversation Export (Markdown) ===")
    markdown_export = conv.export_conversation("markdown")
    print(markdown_export)
