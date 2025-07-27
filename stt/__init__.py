"""
Speech-to-Text (STT) service module

This module provides a unified interface for speech-to-text operations
supporting multiple providers including Faster Whisper and Google Cloud Speech.
"""

from .stt_types import (
    STTProvider,
    AudioFormat,
    STTConfig,
    STTResponse,
    WordTimestamp,
    SpeakerSegment,
    FasterWhisperConfig,
    GoogleSTTConfig,
)
from .stt_service import STTService
from .stt_clients import FasterWhisperClient, GoogleSTTClient
from .stt_factory import (
    STTServiceFactory,
    create_stt_service,
    get_available_providers,
    get_provider_info,
    create_faster_whisper_service,
    create_google_stt_service,
)

__all__ = [
    # Types and enums
    "STTProvider",
    "AudioFormat",
    "STTConfig",
    "STTResponse",
    "WordTimestamp",
    "SpeakerSegment",
    "FasterWhisperConfig",
    "GoogleSTTConfig",
    # Core service
    "STTService",
    # Clients
    "FasterWhisperClient",
    "GoogleSTTClient",
    # Factory and utilities
    "STTServiceFactory",
    "create_stt_service",
    "get_available_providers",
    "get_provider_info",
    "create_faster_whisper_service",
    "create_google_stt_service",
]
