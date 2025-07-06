"""
Factory for creating STT service instances
"""

import os
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

from .stt_types import STTProvider, STTConfig
from .stt_service import STTService
from .stt_clients import FasterWhisperClient, GoogleSTTClient

if TYPE_CHECKING:
    from ..credentials import CredentialManager


class STTServiceFactory:
    """Factory for creating STT service instances"""
    
    @staticmethod
    def create_stt_service(provider: Union[str, STTProvider] = "auto", credential_manager: Optional['CredentialManager'] = None, **kwargs) -> STTService:
        """
        Create an STT service instance
        
        Args:
            provider: Provider type ("auto", "faster_whisper", "google")
            credential_manager: Optional credential manager for explicit credential handling
            **kwargs: Additional configuration parameters
            
        Returns:
            STTService instance
            
        Raises:
            ValueError: If provider is unsupported or unavailable
        """
        if isinstance(provider, str):
            if provider == "auto":
                provider = STTServiceFactory._detect_available_provider()
            else:
                try:
                    provider = STTProvider(provider)
                except ValueError:
                    raise ValueError(f"Unsupported STT provider: {provider}")
        
        # Create configuration
        config = STTServiceFactory._create_config(provider, **kwargs)
        
        # Create appropriate client
        if provider == STTProvider.FASTER_WHISPER:
            client = STTServiceFactory._create_faster_whisper_client(config, credential_manager)
        elif provider == STTProvider.GOOGLE:
            client = STTServiceFactory._create_google_client(config, credential_manager)
        else:
            raise ValueError(f"Unsupported STT provider: {provider}")
        
        return STTService(client)
    
    @staticmethod
    def _detect_available_provider() -> STTProvider:
        """Detect which STT provider is available based on environment and configuration"""
        
        # Check Faster Whisper first (local instance)
        faster_whisper_url = os.getenv("FASTER_WHISPER_URL", "http://localhost:10300")
        if STTServiceFactory._check_faster_whisper_availability(faster_whisper_url):
            return STTProvider.FASTER_WHISPER
        
        # Check Google Speech-to-Text
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return STTProvider.GOOGLE
        
        # If no providers are available, raise an error with helpful message
        raise ValueError(
            "No STT providers are configured. Please set one of the following:\n"
            "- Set up Faster Whisper server on localhost:10300 (or set FASTER_WHISPER_URL)\n" 
            "- GOOGLE_APPLICATION_CREDENTIALS for Google Speech-to-Text"
        )
    
    @staticmethod
    def _check_faster_whisper_availability(server_url: str) -> bool:
        """Check if Faster Whisper server is available"""
        if not server_url:
            return False
            
        try:
            import requests
            # Try health endpoint first
            response = requests.get(f"{server_url}/health", timeout=3)
            if response.status_code == 200:
                return True
        except Exception:
            pass
            
        try:
            # Try root endpoint as fallback
            response = requests.get(f"{server_url}/", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def _create_config(provider: STTProvider, **kwargs) -> STTConfig:
        """Create STTConfig for the specified provider"""
        
        # Default configuration
        config_dict = {
            "provider": provider,
            "language": kwargs.get("language"),
            "model": kwargs.get("model"),
            "enable_word_timestamps": kwargs.get("enable_word_timestamps", False),
            "enable_speaker_diarization": kwargs.get("enable_speaker_diarization", False),
            "max_speakers": kwargs.get("max_speakers"),
            "temperature": kwargs.get("temperature", 0.0),
            "beam_size": kwargs.get("beam_size", 5),
            "extra_params": {}
        }
        
        # Provider-specific defaults and extra parameters
        if provider == STTProvider.FASTER_WHISPER:
            config_dict["extra_params"].update({
                "server_url": kwargs.get("server_url", os.getenv("FASTER_WHISPER_URL", "http://localhost:10300")),
                "timeout": kwargs.get("timeout", 120),
                "model_size": kwargs.get("model_size", "base"),
                "device": kwargs.get("device", "auto"),
                "compute_type": kwargs.get("compute_type", "auto")
            })
            if not config_dict["language"]:
                config_dict["language"] = os.getenv("FASTER_WHISPER_DEFAULT_LANGUAGE", "auto")
            if not config_dict["model"]:
                config_dict["model"] = os.getenv("FASTER_WHISPER_DEFAULT_MODEL", "base")
        
        elif provider == STTProvider.GOOGLE:
            config_dict["extra_params"].update({
                "model": kwargs.get("model", "latest_long"),
                "use_enhanced": kwargs.get("use_enhanced", True),
                "enable_automatic_punctuation": kwargs.get("enable_automatic_punctuation", True),
                "enable_profanity_filter": kwargs.get("enable_profanity_filter", False),
                "sample_rate_hertz": kwargs.get("sample_rate_hertz"),
                "audio_channel_count": kwargs.get("audio_channel_count"),
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            })
            if not config_dict["language"]:
                config_dict["language"] = os.getenv("GOOGLE_STT_DEFAULT_LANGUAGE", "en-US")
        
        # Add any additional extra_params from kwargs
        config_dict["extra_params"].update(kwargs.get("extra_params", {}))
        
        return STTConfig(**config_dict)
    
    @staticmethod
    def _create_faster_whisper_client(config: STTConfig, credential_manager: Optional['CredentialManager'] = None) -> FasterWhisperClient:
        """Create Faster Whisper client"""
        return FasterWhisperClient(config, credential_manager)
    
    @staticmethod
    def _create_google_client(config: STTConfig, credential_manager: Optional['CredentialManager'] = None) -> GoogleSTTClient:
        """Create Google STT client"""
        return GoogleSTTClient(config, credential_manager)
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Check which STT providers are available"""
        providers = {
            "faster_whisper": False,
            "google": False
        }
        
        # Check Faster Whisper
        faster_whisper_url = os.getenv("FASTER_WHISPER_URL", "http://localhost:10300")
        providers["faster_whisper"] = STTServiceFactory._check_faster_whisper_availability(faster_whisper_url)
        
        # Check Google
        providers["google"] = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        
        return providers
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get detailed information about all providers"""
        return {
            "faster_whisper": {
                "name": "Faster Whisper Local",
                "description": "Local Faster Whisper instance for fast speech recognition",
                "server_url": os.getenv("FASTER_WHISPER_URL", "http://localhost:10300"),
                "available": STTServiceFactory.get_available_providers()["faster_whisper"],
                "formats": ["mp3", "wav", "ogg", "flac", "m4a", "webm"],
                "models": ["tiny", "base", "small", "medium", "large"],
                "features": ["Word timestamps", "Language detection", "Fast local processing"]
            },
            "google": {
                "name": "Google Speech-to-Text",
                "description": "Google Cloud Speech-to-Text API",
                "available": STTServiceFactory.get_available_providers()["google"],
                "formats": ["wav", "flac", "mp3", "ogg", "webm"],
                "models": ["latest_short", "latest_long", "command_and_search"],
                "features": ["Speaker diarization", "Word timestamps", "Automatic punctuation", "Many languages"]
            }
        }


# Convenience functions
def create_stt_service(provider: str = "auto", credential_manager: Optional['CredentialManager'] = None, **kwargs) -> STTService:
    """Create an STT service instance (convenience function)"""
    return STTServiceFactory.create_stt_service(provider, credential_manager, **kwargs)


def get_available_providers() -> Dict[str, bool]:
    """Check which STT providers are available (convenience function)"""
    return STTServiceFactory.get_available_providers()


def get_provider_info() -> Dict[str, Any]:
    """Get detailed provider information (convenience function)"""
    return STTServiceFactory.get_provider_info()


# Provider-specific convenience functions
def create_faster_whisper_service(**kwargs) -> STTService:
    """Create Faster Whisper service"""
    return create_stt_service("faster_whisper", **kwargs)


def create_google_stt_service(**kwargs) -> STTService:
    """Create Google STT service"""
    return create_stt_service("google", **kwargs)