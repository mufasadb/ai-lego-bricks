"""
Factory for creating TTS service instances
"""

import os
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

from .tts_types import TTSProvider, TTSConfig, AudioFormat
from .tts_service import TTSService
from .tts_clients import CoquiXTTSClient, OpenAITTSClient, GoogleTTSClient

load_dotenv()


class TTSServiceFactory:
    """Factory for creating TTS service instances"""
    
    @staticmethod
    def create_tts_service(provider: Union[str, TTSProvider] = "auto", **kwargs) -> TTSService:
        """
        Create a TTS service instance
        
        Args:
            provider: Provider type ("auto", "coqui_xtts", "openai", "google")
            **kwargs: Additional configuration parameters
            
        Returns:
            TTSService instance
            
        Raises:
            ValueError: If provider is unsupported or unavailable
        """
        if isinstance(provider, str):
            if provider == "auto":
                provider = TTSServiceFactory._detect_available_provider()
            else:
                try:
                    provider = TTSProvider(provider)
                except ValueError:
                    raise ValueError(f"Unsupported TTS provider: {provider}")
        
        # Create configuration
        config = TTSServiceFactory._create_config(provider, **kwargs)
        
        # Create appropriate client
        if provider == TTSProvider.COQUI_XTTS:
            client = TTSServiceFactory._create_coqui_xtts_client(config)
        elif provider == TTSProvider.OPENAI:
            client = TTSServiceFactory._create_openai_client(config)
        elif provider == TTSProvider.GOOGLE:
            client = TTSServiceFactory._create_google_client(config)
        else:
            raise ValueError(f"Unsupported TTS provider: {provider}")
        
        return TTSService(client)
    
    @staticmethod
    def _detect_available_provider() -> TTSProvider:
        """Detect which TTS provider is available based on environment and configuration"""
        
        # Check Coqui-XTTS first (local instance)
        coqui_url = os.getenv("COQUI_XTTS_URL")
        if coqui_url and TTSServiceFactory._check_coqui_xtts_availability(coqui_url):
            return TTSProvider.COQUI_XTTS
        
        # Check OpenAI TTS
        if os.getenv("OPENAI_API_KEY"):
            return TTSProvider.OPENAI
        
        # Check Google TTS
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return TTSProvider.GOOGLE
        
        # If no providers are available, raise an error with helpful message
        raise ValueError(
            "No TTS providers are configured. Please set one of the following in your .env file:\n"
            "- COQUI_XTTS_URL for Coqui-XTTS\n" 
            "- OPENAI_API_KEY for OpenAI TTS\n"
            "- GOOGLE_APPLICATION_CREDENTIALS for Google TTS"
        )
    
    @staticmethod
    def _check_coqui_xtts_availability(server_url: str) -> bool:
        """Check if Coqui-XTTS server is available"""
        if not server_url:
            return False
            
        try:
            import requests
            # Try studio_speakers endpoint - this is the most reliable test
            response = requests.get(f"{server_url}/studio_speakers", timeout=3)
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
    def _create_config(provider: TTSProvider, **kwargs) -> TTSConfig:
        """Create TTSConfig for the specified provider"""
        
        # Default configuration
        config_dict = {
            "provider": provider,
            "voice": kwargs.get("voice"),
            "speed": kwargs.get("speed", 1.0),
            "pitch": kwargs.get("pitch"),
            "output_format": AudioFormat(kwargs.get("output_format", "mp3")),
            "sample_rate": kwargs.get("sample_rate"),
            "output_path": kwargs.get("output_path"),
            "extra_params": {}
        }
        
        # Provider-specific defaults and extra parameters
        if provider == TTSProvider.COQUI_XTTS:
            config_dict["extra_params"].update({
                "server_url": kwargs.get("server_url", os.getenv("COQUI_XTTS_URL")),
                "timeout": kwargs.get("timeout", 30),
                "language": kwargs.get("language", os.getenv("COQUI_XTTS_DEFAULT_LANGUAGE", "en"))
            })
            if not config_dict["voice"]:
                config_dict["voice"] = os.getenv("COQUI_XTTS_DEFAULT_VOICE", "Claribel Dervla")
        
        elif provider == TTSProvider.OPENAI:
            config_dict["extra_params"].update({
                "model": kwargs.get("model", "tts-1"),
                "api_key": os.getenv("OPENAI_API_KEY")
            })
            if not config_dict["voice"]:
                config_dict["voice"] = "alloy"
        
        elif provider == TTSProvider.GOOGLE:
            config_dict["extra_params"].update({
                "language_code": kwargs.get("language_code", "en-US"),
                "ssml_gender": kwargs.get("ssml_gender", "NEUTRAL"),
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            })
        
        # Add any additional extra_params from kwargs
        config_dict["extra_params"].update(kwargs.get("extra_params", {}))
        
        return TTSConfig(**config_dict)
    
    @staticmethod
    def _create_coqui_xtts_client(config: TTSConfig) -> CoquiXTTSClient:
        """Create Coqui-XTTS client"""
        return CoquiXTTSClient(config)
    
    @staticmethod
    def _create_openai_client(config: TTSConfig) -> OpenAITTSClient:
        """Create OpenAI TTS client"""
        return OpenAITTSClient(config)
    
    @staticmethod
    def _create_google_client(config: TTSConfig) -> GoogleTTSClient:
        """Create Google TTS client"""
        return GoogleTTSClient(config)
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Check which TTS providers are available"""
        providers = {
            "coqui_xtts": False,
            "openai": False,
            "google": False
        }
        
        # Check Coqui-XTTS
        coqui_url = os.getenv("COQUI_XTTS_URL")
        providers["coqui_xtts"] = bool(coqui_url) and TTSServiceFactory._check_coqui_xtts_availability(coqui_url)
        
        # Check OpenAI
        providers["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        
        # Check Google
        providers["google"] = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        
        return providers
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get detailed information about all providers"""
        return {
            "coqui_xtts": {
                "name": "Coqui-XTTS Local",
                "description": "Local Coqui-XTTS instance for high-quality voice synthesis",
                "server_url": os.getenv("COQUI_XTTS_URL"),
                "available": TTSServiceFactory.get_available_providers()["coqui_xtts"],
                "formats": ["mp3", "wav", "ogg"],
                "features": ["Custom voices", "Multiple languages", "Fast local processing"]
            },
            "openai": {
                "name": "OpenAI TTS",
                "description": "OpenAI's high-quality text-to-speech API",
                "available": TTSServiceFactory.get_available_providers()["openai"],
                "formats": ["mp3", "opus", "aac", "flac"],
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                "models": ["tts-1", "tts-1-hd"],
                "features": ["High quality", "Multiple voices", "Speed control"]
            },
            "google": {
                "name": "Google Text-to-Speech",
                "description": "Google Cloud Text-to-Speech API",
                "available": TTSServiceFactory.get_available_providers()["google"],
                "formats": ["mp3", "wav", "ogg", "flac"],
                "features": ["SSML support", "Many languages", "Voice customization"]
            }
        }


# Convenience functions
def create_tts_service(provider: str = "auto", **kwargs) -> TTSService:
    """Create a TTS service instance (convenience function)"""
    return TTSServiceFactory.create_tts_service(provider, **kwargs)


def get_available_providers() -> Dict[str, bool]:
    """Check which TTS providers are available (convenience function)"""
    return TTSServiceFactory.get_available_providers()


def get_provider_info() -> Dict[str, Any]:
    """Get detailed provider information (convenience function)"""
    return TTSServiceFactory.get_provider_info()


# Provider-specific convenience functions
def create_coqui_xtts_service(**kwargs) -> TTSService:
    """Create Coqui-XTTS service"""
    return create_tts_service("coqui_xtts", **kwargs)


def create_openai_tts_service(**kwargs) -> TTSService:
    """Create OpenAI TTS service"""
    return create_tts_service("openai", **kwargs)


def create_google_tts_service(**kwargs) -> TTSService:
    """Create Google TTS service"""
    return create_tts_service("google", **kwargs)