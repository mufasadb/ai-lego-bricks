"""
Types, enums, and configuration models for TTS services
"""

from enum import Enum
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class TTSProvider(str, Enum):
    """Available TTS providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    COQUI_XTTS = "coqui_xtts"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"


class TTSConfig(BaseModel):
    """Configuration for TTS services"""
    provider: TTSProvider
    voice: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.1, le=4.0)
    pitch: Optional[float] = Field(default=None, ge=-20.0, le=20.0)
    output_format: AudioFormat = AudioFormat.MP3
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=48000)
    output_path: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class TTSResponse(BaseModel):
    """Response from TTS service"""
    success: bool
    audio_file_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    duration_ms: Optional[int] = None
    provider: str
    voice_used: Optional[str] = None
    format_used: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TTSClient(ABC):
    """Abstract base class for TTS clients"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
    
    @abstractmethod
    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters that override config
            
        Returns:
            TTSResponse with audio data and metadata
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get available voices for this provider
        
        Returns:
            Dictionary of available voices with metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this TTS provider is available
        
        Returns:
            True if provider is available, False otherwise
        """
        pass


# Voice configuration classes for different providers
class OpenAIVoice(str, Enum):
    """OpenAI TTS voice options"""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class OpenAIModel(str, Enum):
    """OpenAI TTS model options"""
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class GoogleVoiceConfig(BaseModel):
    """Google TTS voice configuration"""
    language_code: str = "en-US"
    name: Optional[str] = None
    ssml_gender: Optional[str] = None  # MALE, FEMALE, NEUTRAL


class CoquiXTTSConfig(BaseModel):
    """Coqui-XTTS specific configuration"""
    server_url: Optional[str] = None
    voice: str = "default"
    language: str = "en"
    temperature: float = 0.7
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    top_k: int = 50
    top_p: float = 0.8