# 🌊 Streaming Support Guide

This guide covers the new streaming capabilities added to the AI Lego Bricks project, enabling real-time response generation for better user experiences.

## Overview

Streaming support has been added throughout the LLM stack:
- **Text Clients**: Native streaming for Ollama, simulated for Gemini/Anthropic
- **Generation Service**: One-shot streaming responses
- **Conversation Service**: Multi-turn streaming conversations
- **Chat Service**: Legacy compatibility with streaming
- **Agent Orchestration**: JSON-driven workflows with streaming
- **TTS Integration**: Real-time audio generation from streaming text

## Key Features

✅ **Backward Compatible**: All existing code continues to work unchanged  
✅ **Provider Support**: Native Ollama streaming, simulated for others  
✅ **Fallback Safety**: Automatic fallback to regular APIs if streaming fails  
✅ **Rich Integration**: Works with agent orchestration workflows  
✅ **Type Safe**: Full TypeScript-style typing with generators  
✅ **TTS Integration**: Real-time audio generation from streaming responses

## Quick Start

### Basic Streaming Generation
```python
from llm.generation_service import quick_generate_ollama_stream

# Stream a response
for chunk in quick_generate_ollama_stream("Tell me about quantum computing"):
    print(chunk, end='', flush=True)
```

### Streaming Conversation
```python
from chat.conversation_service import create_ollama_conversation

conv = create_ollama_conversation()
conv.add_system_message("You are a helpful assistant.")

# Stream a conversation
for chunk in conv.send_message_stream("What are the benefits of renewable energy?"):
    print(chunk, end='', flush=True)
```

### Agent Orchestration with Streaming
```json
{
  "id": "stream_response",
  "type": "llm_chat",
  "config": {
    "provider": "ollama",
    "stream": true,
    "use_conversation": false
  },
  "inputs": {
    "message": "Your prompt here"
  }
}
```

## API Reference

### Text Clients

All text clients now support streaming methods:

```python
# OllamaTextClient, GeminiTextClient, AnthropicTextClient
client.chat_stream(message, chat_history) -> Generator[str, None, str]
client.chat_with_messages_stream(messages) -> Generator[str, None, str]
```

**Ollama**: True streaming via HTTP stream  
**Gemini**: Simulated streaming (chunked response)  
**Anthropic**: Native streaming via official client  

### Generation Service

```python
from llm.generation_service import GenerationService

service = GenerationService(LLMProvider.OLLAMA)

# Streaming methods
service.generate_stream(prompt) -> Generator[str, None, str]
service.generate_with_system_prompt_stream(prompt, system) -> Generator[str, None, str]

# Convenience functions
quick_generate_ollama_stream(prompt) -> Generator[str, None, str]
quick_generate_gemini_stream(prompt) -> Generator[str, None, str]
quick_generate_anthropic_stream(prompt) -> Generator[str, None, str]
```

### Conversation Service

```python
from chat.conversation_service import ConversationService

conv = ConversationService(LLMProvider.OLLAMA)

# Streaming conversation
conv.send_message_stream(message, metadata) -> Generator[str, None, str]

# Regular methods still work
conv.send_message(message) -> str
```

### Chat Service (Legacy)

```python
from chat.chat_service import ChatService

chat = ChatService("ollama")

# Streaming chat
chat.chat_stream(message, history) -> Generator[str, None, str]

# Convenience functions
quick_chat_ollama_stream(message) -> Generator[str, None, str]
quick_chat_gemini_stream(message) -> Generator[str, None, str]
```

## Agent Orchestration

### Configuration

Add streaming to any `llm_chat` step with the `stream` parameter:

```json
{
  "id": "my_streaming_step",
  "type": "llm_chat",
  "config": {
    "provider": "ollama",
    "model": "llama2",
    "stream": true,
    "use_conversation": false,
    "temperature": 0.7
  },
  "inputs": {
    "message": "Your prompt"
  }
}
```

### Output Format

Streaming steps return additional fields:

```json
{
  "response": "Complete response text",
  "streamed": true,
  "chunks": ["chunk1", "chunk2", "chunk3"],
  "message": "Original prompt",
  "provider": "ollama",
  "model": "llama2"
}
```

### Examples

- **[streaming_chat_agent.json](agent_orchestration/examples/streaming_chat_agent.json)**: Basic streaming chat
- **[streaming_conversation_agent.json](agent_orchestration/examples/streaming_conversation_agent.json)**: Multi-turn streaming conversation

## Implementation Details

### How Streaming Works

1. **Ollama**: Uses HTTP streaming via `httpx.stream()` to receive chunks as they're generated
2. **Gemini**: Simulates streaming by chunking the complete response (real streaming API coming)
3. **Anthropic**: Uses native streaming via the official client library
4. **Fallback**: All implementations fall back to regular APIs if streaming fails

### Generator Pattern

All streaming methods return Python generators:

```python
def chat_stream(self, message: str) -> Generator[str, None, str]:
    """
    Yields: str - Partial response chunks
    Returns: str - Complete response when done
    """
```

### Error Handling

Streaming methods automatically fall back to regular APIs on error:

```python
try:
    # Attempt streaming
    for chunk in client.chat_stream(message):
        yield chunk
except Exception:
    # Fall back to regular API
    full_response = client.chat(message)
    yield full_response
```

## Best Practices

### 1. UI Integration
```python
# Real-time UI updates
response_area = ""
for chunk in service.generate_stream(prompt):
    response_area += chunk
    update_ui(response_area)  # Update UI in real-time
```

### 2. Error Handling
```python
try:
    chunks = []
    for chunk in service.generate_stream(prompt):
        chunks.append(chunk)
        print(chunk, end='', flush=True)
    full_response = ''.join(chunks)
except Exception as e:
    print(f"Streaming failed: {e}")
    # Handle fallback
```

### 3. Performance Considerations
- Streaming reduces perceived latency but may have higher total latency
- Use for interactive applications where immediate feedback is important
- Consider user experience over raw performance

### 4. Provider Selection
- **Ollama**: Best streaming experience, real-time chunks
- **Anthropic**: Good streaming support via official client
- **Gemini**: Simulated streaming, but still provides progressive output

## Configuration Examples

### Environment Variables
```bash
# Ollama (for true streaming)
OLLAMA_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2

# Other providers work but with simulated streaming
GOOGLE_AI_STUDIO_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### Workflow Configuration
```json
{
  "name": "streaming_workflow",
  "config": {
    "default_llm_provider": "ollama"
  },
  "steps": [
    {
      "id": "stream_response",
      "type": "llm_chat",
      "config": {
        "stream": true,
        "provider": "ollama"
      }
    }
  ]
}
```

## Migration Guide

### Existing Code
No changes needed! All existing code continues to work:

```python
# This still works exactly as before
response = service.generate("Hello world")
```

### Adding Streaming
Simply replace method calls with streaming versions:

```python
# Before
response = service.generate(prompt)

# After  
for chunk in service.generate_stream(prompt):
    print(chunk, end='', flush=True)
```

## Testing

Run the streaming examples:

```bash
# Test all streaming features
python examples/streaming_example.py

# Test agent orchestration streaming
python agent_orchestration/examples/usage_example.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're using the updated imports
2. **Streaming Not Working**: Check if the provider supports streaming
3. **Connection Issues**: Verify Ollama is running for native streaming

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Provider Status
- ✅ **Ollama**: Full streaming support
- 🔄 **Gemini**: Simulated streaming (real streaming coming)
- ✅ **Anthropic**: Native streaming support

## TTS Integration

### Streaming LLM to TTS Pipeline

The streaming system integrates with the TTS module to enable real-time audio generation:

```python
from tts.streaming_tts_service import create_streaming_pipeline

# Create streaming pipeline
pipeline = create_streaming_pipeline(
    llm_provider="ollama",
    tts_provider="auto",
    streaming_config={
        "sentence_buffer_size": 2,
        "min_chunk_length": 20,
        "max_buffer_time": 3.0
    }
)

# Stream LLM response directly to audio
for progress in pipeline.stream_chat_to_audio("Explain quantum computing"):
    print(f"Status: {progress['status']}, Audio files: {progress['audio_files_generated']}")
```

### How It Works

1. **Text Streaming**: LLM generates text chunks in real-time
2. **Sentence Detection**: Complete sentences are identified as they form
3. **Audio Generation**: Each sentence is converted to audio immediately
4. **Progressive Output**: Audio files are created incrementally

### Configuration Options

```python
streaming_config = {
    "sentence_buffer_size": 2,    # Buffer 2 sentences before generating audio
    "min_chunk_length": 20,       # Minimum characters for audio generation
    "max_buffer_time": 3.0,       # Maximum seconds to wait before forcing generation
    "output_dir": "output/audio"  # Directory for audio files
}
```

## Future Enhancements

- **Real Gemini Streaming**: When Google releases streaming API
- **OpenAI Integration**: With native streaming support
- **Structured Streaming**: Streaming for structured responses
- **Agent Streaming**: Real-time streaming in orchestration workflows
- **Advanced TTS Streaming**: Voice cloning and real-time voice synthesis

---

The streaming implementation maintains full backward compatibility while providing a modern, efficient streaming experience for real-time AI applications.