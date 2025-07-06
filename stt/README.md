# Speech-to-Text (STT) Service

The STT service provides a unified interface for speech-to-text operations across multiple providers.

## Supported Providers

### Faster Whisper (Local)
- **Provider**: `faster_whisper`
- **Description**: Local Faster Whisper instance for fast, offline speech recognition
- **Server URL**: Default `http://localhost:10300` (configurable via `FASTER_WHISPER_URL`)
- **Features**: Word timestamps, language detection, multiple model sizes
- **Models**: tiny, base, small, medium, large

### Google Cloud Speech-to-Text
- **Provider**: `google`
- **Description**: Google Cloud Speech-to-Text API
- **Authentication**: Requires `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- **Features**: Speaker diarization, word timestamps, automatic punctuation, many languages
- **Models**: latest_short, latest_long, command_and_search

## Quick Start

### Using the Factory
```python
from stt import create_stt_service

# Auto-detect available provider
stt_service = create_stt_service("auto")

# Specific provider
stt_service = create_stt_service("faster_whisper")
stt_service = create_stt_service("google")

# Transcribe audio
response = stt_service.speech_to_text("path/to/audio.wav")
print(response.transcript)
```

### Using in Agent Workflows
```json
{
  "id": "transcribe_step",
  "type": "stt",
  "config": {
    "provider": "faster_whisper",
    "language": "auto",
    "enable_word_timestamps": true
  },
  "inputs": {
    "audio_file_path": "${input_audio}"
  }
}
```

## Configuration

### Environment Variables
- `FASTER_WHISPER_URL`: URL for Faster Whisper server (default: `http://localhost:10300`)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud service account JSON file
- `FASTER_WHISPER_DEFAULT_LANGUAGE`: Default language for Faster Whisper (default: `auto`)
- `FASTER_WHISPER_DEFAULT_MODEL`: Default model for Faster Whisper (default: `base`)

### Provider Configuration

#### Faster Whisper
```python
config = {
    "provider": "faster_whisper",
    "language": "en",
    "model": "base",
    "enable_word_timestamps": True,
    "temperature": 0.0,
    "beam_size": 5,
    "extra_params": {
        "server_url": "http://localhost:10300",
        "timeout": 120
    }
}
```

#### Google Cloud Speech
```python
config = {
    "provider": "google",
    "language": "en-US",
    "model": "latest_long",
    "enable_word_timestamps": True,
    "enable_speaker_diarization": True,
    "max_speakers": 6,
    "extra_params": {
        "use_enhanced": True,
        "enable_automatic_punctuation": True
    }
}
```

## Response Format

```python
{
    "success": True,
    "transcript": "Hello, this is a test recording.",
    "language_detected": "en",
    "confidence": 0.95,
    "word_timestamps": [
        {"word": "Hello", "start_time": 0.5, "end_time": 1.0},
        {"word": "this", "start_time": 1.2, "end_time": 1.4}
    ],
    "speaker_segments": [
        {
            "speaker_id": "speaker_1",
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Hello, this is a test recording."
        }
    ],
    "duration_seconds": 5.2,
    "provider": "faster_whisper",
    "model_used": "base"
}
```

## Supported Audio Formats

- MP3
- WAV
- OGG
- FLAC
- M4A
- WEBM

## Examples

See the `examples/` directory for complete workflow examples:
- `stt_workflow_example.py` - Python usage examples
- `agent_orchestration/examples/simple_stt_agent.json` - Basic STT workflow
- `agent_orchestration/examples/voice_analysis_agent.json` - STT + LLM analysis
- `agent_orchestration/examples/voice_assistant_agent.json` - Bidirectional voice assistant

## Setup

### Faster Whisper Server
You need to set up a Faster Whisper server on localhost:10300. This is your local server mentioned in the requirements.

### Google Cloud Speech
1. Create a Google Cloud project with Speech-to-Text API enabled
2. Create a service account and download the credentials JSON
3. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the JSON file path

## Error Handling

The service provides detailed error messages for common issues:
- Missing audio files
- Unsupported audio formats
- Network connectivity issues
- Authentication failures
- Provider-specific errors

All errors are returned in a consistent format with `success: False` and an `error_message` field.