# Speech-to-Text (STT) Service

A unified interface for speech-to-text operations supporting multiple providers with automatic fallback and configuration management.

## Quick Start

```python
from stt import create_stt_service

# Auto-detect available provider
stt = create_stt_service("auto")
result = stt.speech_to_text("audio.wav")
print(result.transcript)
```

## Supported Providers

- **Faster Whisper** (`faster_whisper`) - Local offline processing via localhost:10300
- **Google Cloud Speech** (`google`) - Cloud-based with advanced features

## Basic Usage

### Simple Transcription
```python
from stt import create_stt_service

# Create service with specific provider
stt = create_stt_service("faster_whisper")

# Basic transcription
response = stt.speech_to_text("recording.wav")
if response.success:
    print(f"Transcript: {response.transcript}")
    print(f"Confidence: {response.confidence}")
```

### With Configuration
```python
from stt import create_stt_service

# Configure provider options
stt = create_stt_service(
    provider="google",
    language="en-US",
    enable_word_timestamps=True,
    enable_speaker_diarization=True
)

response = stt.speech_to_text("meeting.wav")
if response.success:
    print(f"Transcript: {response.transcript}")
    
    # Word-level timestamps
    for word in response.word_timestamps:
        print(f"{word.word}: {word.start_time}s - {word.end_time}s")
    
    # Speaker segments
    for segment in response.speaker_segments:
        print(f"Speaker {segment.speaker_id}: {segment.text}")
```

## Common Use Cases

### Voice Assistant
```python
# Real-time voice processing
stt = create_stt_service("faster_whisper", language="auto")

def process_voice_command(audio_file):
    result = stt.speech_to_text(audio_file)
    if result.success:
        command = result.transcript.lower().strip()
        # Process command...
        return command
    return None
```

### Meeting Transcription
```python
# Multi-speaker meeting transcription
stt = create_stt_service(
    provider="google",
    enable_speaker_diarization=True,
    max_speakers=6,
    enable_word_timestamps=True
)

def transcribe_meeting(audio_file):
    result = stt.speech_to_text(audio_file)
    if result.success:
        # Format transcript with speakers
        transcript = []
        for segment in result.speaker_segments:
            transcript.append(f"{segment.speaker_id}: {segment.text}")
        return "\n".join(transcript)
    return None
```

### Batch Processing
```python
import os
from pathlib import Path

def batch_transcribe(audio_dir):
    stt = create_stt_service("auto")
    results = {}
    
    for audio_file in Path(audio_dir).glob("*.wav"):
        result = stt.speech_to_text(str(audio_file))
        if result.success:
            results[audio_file.name] = result.transcript
    
    return results
```

## Configuration

### Environment Variables
```bash
# Faster Whisper
FASTER_WHISPER_URL=http://localhost:10300
FASTER_WHISPER_DEFAULT_MODEL=base

# Google Cloud Speech
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Provider-Specific Options

**Faster Whisper:**
```python
config = {
    "provider": "faster_whisper",
    "language": "en",
    "model": "base",  # tiny, base, small, medium, large
    "enable_word_timestamps": True,
    "temperature": 0.0
}
```

**Google Cloud Speech:**
```python
config = {
    "provider": "google",
    "language": "en-US",
    "model": "latest_long",
    "enable_word_timestamps": True,
    "enable_speaker_diarization": True,
    "max_speakers": 6
}
```

## Response Format

```python
STTResponse(
    success=True,
    transcript="Hello, how can I help you today?",
    language_detected="en",
    confidence=0.95,
    word_timestamps=[
        WordTimestamp(word="Hello", start_time=0.5, end_time=1.0),
        WordTimestamp(word="how", start_time=1.2, end_time=1.4),
        # ...
    ],
    speaker_segments=[
        SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=3.0,
            text="Hello, how can I help you today?"
        )
    ],
    duration_seconds=3.2,
    provider="faster_whisper",
    model_used="base"
)
```

## Supported Audio Formats

MP3, WAV, OGG, FLAC, M4A, WEBM

## Setup

### Faster Whisper (Local)
Set up a Faster Whisper server on `localhost:10300`

### Google Cloud Speech
1. Enable Speech-to-Text API in Google Cloud Console
2. Create service account and download credentials JSON
3. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## Error Handling

```python
result = stt.speech_to_text("audio.wav")
if not result.success:
    print(f"Error: {result.error_message}")
    # Handle error appropriately
```

Common errors are returned with descriptive messages for missing files, unsupported formats, network issues, and authentication failures.