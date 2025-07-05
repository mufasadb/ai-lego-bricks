"""
Main TTS service interface
"""

from typing import Optional, Dict, Any
from .tts_types import TTSConfig, TTSResponse, TTSClient


class TTSService:
    """
    Main TTS service that provides a unified interface for text-to-speech operations
    """
    
    def __init__(self, client: TTSClient):
        """
        Initialize TTS service with a specific client
        
        Args:
            client: TTS client implementation
        """
        self.client = client
        self.config = client.config
    
    def text_to_speech(self, text: str, voice: Optional[str] = None, 
                      output_path: Optional[str] = None, **kwargs) -> TTSResponse:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (overrides config)
            output_path: Where to save audio file (overrides config)
            **kwargs: Additional parameters that override config
            
        Returns:
            TTSResponse with audio data and metadata
        """
        # Validate input
        if not text or not text.strip():
            return TTSResponse(
                success=False,
                error_message="Text cannot be empty",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value
            )
        
        # Override config with provided parameters
        override_kwargs = {}
        if voice:
            override_kwargs['voice'] = voice
        if output_path:
            override_kwargs['output_path'] = output_path
        override_kwargs.update(kwargs)
        
        try:
            response = self.client.text_to_speech(text.strip(), **override_kwargs)
            return response
        except Exception as e:
            return TTSResponse(
                success=False,
                error_message=f"TTS generation failed: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value
            )
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get available voices for the current provider
        
        Returns:
            Dictionary of available voices with metadata
        """
        try:
            return self.client.get_available_voices()
        except Exception as e:
            return {"error": f"Failed to get voices: {str(e)}"}
    
    def is_available(self) -> bool:
        """
        Check if the TTS service is available
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            return self.client.is_available()
        except Exception:
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.config.provider.value,
            "voice": self.config.voice,
            "speed": self.config.speed,
            "output_format": self.config.output_format.value,
            "available": self.is_available(),
            "voices": self.get_available_voices()
        }
    
    def test_synthesis(self, test_text: str = "Hello, this is a test of the text-to-speech system.") -> TTSResponse:
        """
        Test the TTS service with a simple phrase
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            TTSResponse with test results
        """
        return self.text_to_speech(test_text)
    
    def switch_voice(self, voice: str) -> bool:
        """
        Switch to a different voice
        
        Args:
            voice: Voice to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            available_voices = self.get_available_voices()
            if voice in available_voices or "error" not in available_voices:
                self.config.voice = voice
                return True
            return False
        except Exception:
            return False