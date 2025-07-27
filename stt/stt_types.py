"""
Types, enums, and configuration models for STT (Speech-to-Text) services
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class STTProvider(str, Enum):
    """Available STT providers"""

    FASTER_WHISPER = "faster_whisper"
    GOOGLE = "google"


class AudioFormat(str, Enum):
    """Supported audio input formats"""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"
    WEBM = "webm"


class STTConfig(BaseModel):
    """Configuration for STT services"""

    provider: STTProvider
    language: Optional[str] = "en-US"
    model: Optional[str] = None  # For Whisper: tiny, base, small, medium, large
    enable_word_timestamps: bool = False
    enable_speaker_diarization: bool = False
    max_speakers: Optional[int] = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    beam_size: int = Field(default=5, ge=1, le=50)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class WordTimestamp(BaseModel):
    """Word-level timestamp information"""

    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


class SpeakerSegment(BaseModel):
    """Speaker diarization segment"""

    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None


class STTResponse(BaseModel):
    """Response from STT service"""

    success: bool
    transcript: Optional[str] = None
    language_detected: Optional[str] = None
    confidence: Optional[float] = None
    word_timestamps: List[WordTimestamp] = Field(default_factory=list)
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    provider: str
    model_used: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"protected_namespaces": ()}


class STTClient(ABC):
    """Abstract base class for STT clients"""

    def __init__(self, config: STTConfig):
        self.config = config

    @abstractmethod
    def speech_to_text(self, audio_file_path: str, **kwargs) -> STTResponse:
        """
        Convert speech to text

        Args:
            audio_file_path: Path to audio file to transcribe
            **kwargs: Additional parameters that override config

        Returns:
            STTResponse with transcript and metadata
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for this provider

        Returns:
            List of supported language codes
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this STT provider is available

        Returns:
            True if provider is available, False otherwise
        """
        pass


# Provider-specific configuration classes
class FasterWhisperConfig(BaseModel):
    """Faster Whisper specific configuration"""

    server_url: Optional[str] = "http://localhost:10300"
    model_size: str = "base"  # tiny, base, small, medium, large
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "auto"  # auto, int8, float16, float32

    model_config = {"protected_namespaces": ()}


class GoogleSTTConfig(BaseModel):
    """Google Cloud Speech-to-Text configuration"""

    model: str = "latest_long"  # latest_short, latest_long, command_and_search, etc.
    use_enhanced: bool = True
    enable_automatic_punctuation: bool = True
    enable_profanity_filter: bool = False
    sample_rate_hertz: Optional[int] = None
    audio_channel_count: Optional[int] = None
