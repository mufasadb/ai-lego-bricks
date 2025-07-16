# TTS (Text-to-Speech) Module

A unified interface for multiple text-to-speech providers supporting OpenAI, Google Cloud, and Coqui-XTTS with streaming capabilities.

## Quick Start

```python
from tts import create_tts_service

# Auto-detect available provider
tts = create_tts_service("auto")

# Generate speech
response = tts.text_to_speech("Hello, world!")
print(f"Audio saved to: {response.audio_file_path}")
```

## Supported Providers

### OpenAI TTS
- **Voices**: alloy, echo, fable, onyx, nova, shimmer
- **Models**: tts-1, tts-1-hd
- **Setup**: Set `OPENAI_API_KEY` in environment

```python
# OpenAI with specific voice
tts = create_tts_service("openai", voice="nova")
response = tts.text_to_speech("Hello from OpenAI!", output_path="output.mp3")
```

### Google Cloud TTS
- **Features**: SSML support, 200+ voices, multiple languages
- **Setup**: Set `GOOGLE_APPLICATION_CREDENTIALS` pointing to service account JSON

```python
# Google TTS with language
tts = create_tts_service("google")
response = tts.text_to_speech(
    "Hello from Google!", 
    language_code="en-US",
    output_path="output.wav"
)
```

### Coqui-XTTS (Local)
- **Features**: Custom voice cloning, multilingual, fast local processing
- **Setup**: Set `COQUI_XTTS_URL` to your local server (e.g., `http://localhost:8020`)

```python
# Coqui-XTTS with custom voice
tts = create_tts_service("coqui_xtts", voice="Claribel Dervla")
response = tts.text_to_speech("Hello from Coqui!", language="en")
```

## Voice Control

```python
# Get available voices
voices = tts.get_available_voices()
print(f"Available voices: {list(voices.keys())}")

# Switch voices dynamically
response1 = tts.text_to_speech("First voice", voice="alloy")
response2 = tts.text_to_speech("Second voice", voice="echo")
```

## Streaming TTS

For real-time applications like voice assistants:

```python
from tts import StreamingTTSService

# Create streaming service
streaming_tts = StreamingTTSService(
    tts_service=tts,
    sentence_buffer_size=2,  # Buffer 2 sentences
    max_buffer_time=3.0      # Max 3 second delay
)

# Stream text chunks to audio
def text_generator():
    yield "Hello there! "
    yield "This is streaming TTS. "
    yield "Each sentence becomes audio as it completes."

for progress in streaming_tts.stream_text_to_audio(text_generator()):
    print(f"Status: {progress['status']}, Files: {progress.get('audio_files_generated', 0)}")
```

## LLM Integration

Stream LLM responses directly to speech:

```python
from tts.streaming_tts_service import create_streaming_pipeline

# Create pipeline
pipeline = create_streaming_pipeline(
    llm_provider="ollama",
    tts_provider="openai"
)

# Stream chat to audio
for progress in pipeline.stream_chat_to_audio("Tell me about AI"):
    print(f"Audio files generated: {progress.get('audio_files_generated', 0)}")
```

## Configuration

```python
# Provider-specific configurations
tts = create_tts_service("openai", 
    voice="nova",
    speed=1.2,              # 1.0 = normal, 0.5-2.0 range
    output_format="mp3",    # mp3, wav, ogg, flac
    model="tts-1-hd"        # Higher quality model
)

# Coqui-XTTS advanced options
tts = create_tts_service("coqui_xtts",
    voice="custom_voice",
    language="en",
    server_url="http://localhost:8020",
    timeout=30
)
```

## Error Handling

```python
response = tts.text_to_speech("Hello world")

if response.success:
    print(f"✅ Generated: {response.audio_file_path}")
    print(f"Duration: {response.duration_ms}ms")
    print(f"Voice: {response.voice_used}")
else:
    print(f"❌ Error: {response.error_message}")
```

## Environment Setup

Create a `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Coqui-XTTS Local Server
COQUI_XTTS_URL=http://localhost:8020
COQUI_XTTS_DEFAULT_VOICE=Claribel Dervla
```

## Use Cases

- **Voice Assistants**: Real-time speech synthesis from chat responses
- **Accessibility**: Convert text content to audio for visually impaired users
- **Content Creation**: Generate narration for videos or presentations
- **Language Learning**: Pronunciation examples in multiple languages
- **Interactive Applications**: Dynamic speech feedback in games or apps

## Provider Detection

The module automatically detects available providers:

```python
from tts import get_available_providers, get_provider_info

# Check what's available
providers = get_available_providers()
# {'openai': True, 'google': False, 'coqui_xtts': True}

# Get detailed info
info = get_provider_info()
for provider, details in info.items():
    print(f"{provider}: {details['description']}")
```

## Audio Formats

Supported formats vary by provider:
- **OpenAI**: mp3, opus, aac, flac
- **Google**: mp3, wav, ogg, flac  
- **Coqui-XTTS**: mp3, wav, ogg

Default format is MP3 for compatibility.