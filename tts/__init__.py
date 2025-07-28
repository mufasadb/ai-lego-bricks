"""
Text-to-Speech (TTS) module for AI Lego Bricks

This module provides TTS capabilities with support for multiple providers:
- OpenAI TTS
- Google Text-to-Speech
- Coqui-XTTS (local instance)

Features:
- Standard TTS for complete text
- Streaming TTS for real-time audio generation from LLM streams
- Multiple provider support with auto-detection
- Agent orchestration integration

Usage:
    from tts import create_tts_service, TTSProvider

    # Create TTS service with auto-detection
    tts = create_tts_service("auto")

    # Create specific provider
    tts = create_tts_service("coqui_xtts")

    # Generate speech
    audio_path = tts.text_to_speech("Hello world!", voice="default")

    # Streaming TTS
    from tts.streaming_tts_service import create_streaming_pipeline
    pipeline = create_streaming_pipeline()
    for progress in pipeline.stream_chat_to_audio("Tell me about AI"):
        print(f"Status: {progress['status']}")
"""

from .tts_types import TTSProvider, TTSConfig, TTSResponse
from .tts_service import TTSService
from .tts_factory import TTSServiceFactory
from .streaming_tts_service import (
    StreamingTTSService,
    StreamingLLMToTTSPipeline,
    create_streaming_pipeline,
)


# Convenience functions
def create_tts_service(provider: str = "auto", **kwargs) -> TTSService:
    """Create a TTS service instance (convenience function)"""
    return TTSServiceFactory.create_tts_service(provider, **kwargs)


def get_available_providers():
    """Get available TTS providers"""
    return TTSServiceFactory.get_available_providers()


def get_provider_info():
    """Get detailed provider information"""
    return TTSServiceFactory.get_provider_info()


# Export main classes and functions
__all__ = [
    "TTSService",
    "TTSProvider",
    "TTSConfig",
    "TTSResponse",
    "TTSServiceFactory",
    "StreamingTTSService",
    "StreamingLLMToTTSPipeline",
    "create_tts_service",
    "create_streaming_pipeline",
    "get_available_providers",
    "get_provider_info",
]
