# Chat Services

Unified chat and conversation services for LLM integrations. Choose between stateless generation or stateful multi-turn conversations.

## Services

### ChatService
Quick, stateless chat interactions with minimal setup.

```python
from chat import create_chat_service

# Create service
chat = create_chat_service("gemini")  # or "ollama"
response = chat.chat("What is Python?")

# With conversation history
history = []
response, history = chat.chat_with_history("What is Python?", history)
response, history = chat.chat_with_history("How do I create lists?", history)
```

### ConversationService
Rich multi-turn conversations with full state management and history access.

```python
from chat import create_gemini_conversation

# Create conversation
conv = create_gemini_conversation(temperature=0.7)

# Add system context
conv.add_system_message("You are a helpful Python tutor.")

# Send messages
response1 = conv.send_message("What is Python?")
response2 = conv.send_message("How do I create lists?")

# Access conversation history
print(f"First prompt: {conv.get_first_prompt()}")
print(f"Last response: {conv.get_last_response()}")
print(f"Total messages: {conv.get_conversation_length()}")
```

## Quick Start

```python
from chat import quick_chat_gemini, create_conversation
from llm.llm_types import LLMProvider

# One-shot generation
answer = quick_chat_gemini("What's 2+2?")

# Multi-turn conversation
conv = create_conversation(LLMProvider.GEMINI)
conv.add_system_message("You are a coding assistant.")
response = conv.send_message("Help me debug this code")
```

## Streaming Support

```python
# Streaming responses
for chunk in chat.chat_stream("Tell me a story"):
    print(chunk, end='', flush=True)

# Streaming conversations
for chunk in conv.send_message_stream("Explain async programming"):
    print(chunk, end='', flush=True)
```

## Rich Conversation Access

```python
# Search messages
results = conv.search_messages("Python", role="user")

# Get recent messages
recent = conv.get_recent_messages(5)

# Export conversation
markdown = conv.export_conversation('markdown')
json_data = conv.export_conversation('json')

# Get statistics
stats = conv.get_conversation_stats()
```

## Configuration

Set environment variables:
```bash
# For Gemini
export GOOGLE_AI_STUDIO_KEY=your_key_here

# For Ollama
export OLLAMA_URL=http://localhost:11434
export OLLAMA_DEFAULT_MODEL=llama2
```

## Use Cases

- **ChatService**: Quick Q&A, single interactions, lightweight bots
- **ConversationService**: Multi-turn dialogs, context-aware assistants, conversation analysis

## Supported Providers

- **Gemini**: Google's AI models
- **Ollama**: Local LLM inference
- **Anthropic**: Claude models (via conversation service)