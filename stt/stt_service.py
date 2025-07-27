"""
Main STT service interface
"""

import os
from typing import Optional, Dict, Any, List
from .stt_types import STTResponse, STTClient


class STTService:
    """
    Main STT service that provides a unified interface for speech-to-text operations
    """

    def __init__(self, client: STTClient):
        """
        Initialize STT service with a specific client

        Args:
            client: STT client implementation
        """
        self.client = client
        self.config = client.config

    def speech_to_text(
        self, audio_file_path: str, language: Optional[str] = None, **kwargs
    ) -> STTResponse:
        """
        Convert speech to text

        Args:
            audio_file_path: Path to audio file to transcribe
            language: Language code (overrides config)
            **kwargs: Additional parameters that override config

        Returns:
            STTResponse with transcript and metadata
        """
        # Validate input file
        if not audio_file_path or not os.path.isfile(audio_file_path):
            return STTResponse(
                success=False,
                error_message=f"Audio file not found: {audio_file_path}",
                provider=self.config.provider.value,
            )

        # Check file format
        os.path.splitext(audio_file_path)[1].lower().lstrip(".")
        [
            fmt.value
            for fmt in self.client.config.provider.__class__.__bases__[0].__dict__.get(
                "AudioFormat", []
            )
        ]

        # Override config with provided parameters
        override_kwargs = {}
        if language:
            override_kwargs["language"] = language
        override_kwargs.update(kwargs)

        try:
            response = self.client.speech_to_text(audio_file_path, **override_kwargs)
            return response
        except Exception as e:
            return STTResponse(
                success=False,
                error_message=f"STT transcription failed: {str(e)}",
                provider=self.config.provider.value,
            )

    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for the current provider

        Returns:
            List of supported language codes
        """
        try:
            return self.client.get_supported_languages()
        except Exception:
            return []

    def is_available(self) -> bool:
        """
        Check if the STT service is available

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
            "language": self.config.language,
            "model": self.config.model,
            "available": self.is_available(),
            "supported_languages": self.get_supported_languages(),
        }

    def test_transcription(self, test_audio_path: Optional[str] = None) -> STTResponse:
        """
        Test the STT service with an audio file

        Args:
            test_audio_path: Path to test audio file

        Returns:
            STTResponse with test results
        """
        if not test_audio_path:
            return STTResponse(
                success=False,
                error_message="No test audio file provided",
                provider=self.config.provider.value,
            )

        return self.speech_to_text(test_audio_path)

    def switch_language(self, language: str) -> bool:
        """
        Switch to a different language

        Args:
            language: Language code to switch to

        Returns:
            True if successful, False otherwise
        """
        try:
            supported_languages = self.get_supported_languages()
            if language in supported_languages:
                self.config.language = language
                return True
            return False
        except Exception:
            return False

    def switch_model(self, model: str) -> bool:
        """
        Switch to a different model (for providers that support multiple models)

        Args:
            model: Model to switch to

        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.model = model
            return True
        except Exception:
            return False
