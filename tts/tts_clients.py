"""
TTS client implementations for different providers
"""

import os
import time
import requests
import tempfile
import base64
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..credentials import CredentialManager
from .tts_types import (
    TTSClient,
    TTSConfig,
    TTSResponse,
    AudioFormat,
    OpenAIVoice,
    OpenAIModel,
)


class CoquiXTTSClient(TTSClient):
    """
    Client for Coqui-XTTS local instance
    """

    def __init__(
        self,
        config: TTSConfig,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        super().__init__(config)
        from ..credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.server_url = config.extra_params.get(
            "server_url", self.credential_manager.get_credential("COQUI_XTTS_URL")
        )
        if not self.server_url:
            raise ValueError(
                "COQUI_XTTS_URL environment variable is required for Coqui-XTTS client. Please set it in your .env file."
            )
        self.timeout = config.extra_params.get("timeout", 30)
        self._speakers_cache = None
        self._languages_cache = None

    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """Convert text to speech using Coqui-XTTS"""

        # Merge config with kwargs
        voice = kwargs.get("voice", self.config.voice or "Claribel Dervla")
        output_path = kwargs.get("output_path", self.config.output_path)
        language = kwargs.get("language", "en")

        # Get speaker data for the requested voice
        try:
            speakers = self._get_speakers()
            if voice not in speakers:
                available_voices = list(speakers.keys())
                if available_voices:
                    voice = available_voices[0]  # Use first available voice as fallback
                else:
                    return TTSResponse(
                        success=False,
                        error_message="No speakers available on Coqui-XTTS server",
                        provider=self.config.provider.value,
                        format_used=self.config.output_format.value,
                    )

            speaker_data = speakers[voice]
            speaker_embedding = speaker_data["speaker_embedding"]
            gpt_cond_latent = speaker_data["gpt_cond_latent"]

        except Exception as e:
            return TTSResponse(
                success=False,
                error_message=f"Failed to get speaker data: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )

        # Prepare request data according to Coqui-XTTS API spec
        request_data = {
            "text": text,
            "language": language,
            "speaker_embedding": speaker_embedding,
            "gpt_cond_latent": gpt_cond_latent,
        }

        try:
            start_time = time.time()

            # Make request to Coqui-XTTS server
            response = requests.post(
                f"{self.server_url}/tts",
                json=request_data,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                return TTSResponse(
                    success=False,
                    error_message=f"Coqui-XTTS server error: {response.status_code} - {response.text}",
                    provider=self.config.provider.value,
                    format_used=self.config.output_format.value,
                )

            # Get audio data and decode from base64
            try:
                # Coqui XTTS returns base64-encoded audio data
                audio_data = base64.b64decode(response.content)
            except Exception:
                # If decoding fails, use raw content as fallback
                audio_data = response.content

            duration_ms = int((time.time() - start_time) * 1000)

            # Save to file if output_path specified
            file_path = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                file_path = os.path.abspath(output_path)
            else:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{self.config.output_format.value}", delete=False
                ) as f:
                    f.write(audio_data)
                    file_path = f.name

            return TTSResponse(
                success=True,
                audio_file_path=file_path,
                audio_data=audio_data,
                duration_ms=duration_ms,
                provider=self.config.provider.value,
                voice_used=voice,
                format_used=self.config.output_format.value,
                metadata={
                    "server_url": self.server_url,
                    "language": language,
                    "text_length": len(text),
                },
            )

        except requests.exceptions.RequestException as e:
            return TTSResponse(
                success=False,
                error_message=f"Connection to Coqui-XTTS server failed: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )
        except Exception as e:
            return TTSResponse(
                success=False,
                error_message=f"Coqui-XTTS error: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )

    def _get_speakers(self) -> Dict[str, Any]:
        """Get speaker data from Coqui-XTTS server with caching"""
        if self._speakers_cache is None:
            try:
                response = requests.get(
                    f"{self.server_url}/studio_speakers", timeout=self.timeout
                )

                if response.status_code == 200:
                    self._speakers_cache = response.json()
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            except Exception as e:
                raise Exception(f"Could not fetch speakers: {str(e)}")

        return self._speakers_cache

    def _get_languages(self) -> list:
        """Get available languages from Coqui-XTTS server with caching"""
        if self._languages_cache is None:
            try:
                response = requests.get(
                    f"{self.server_url}/languages", timeout=self.timeout
                )

                if response.status_code == 200:
                    self._languages_cache = response.json()
                else:
                    self._languages_cache = ["en"]  # Fallback to English
            except Exception:
                self._languages_cache = ["en"]  # Fallback to English

        return self._languages_cache

    def get_available_voices(self) -> Dict[str, Any]:
        """Get available voices from Coqui-XTTS server"""
        try:
            speakers = self._get_speakers()
            languages = self._get_languages()

            result = {}
            for speaker_name in speakers.keys():
                result[speaker_name] = {
                    "name": speaker_name,
                    "languages": languages,
                    "description": f"Coqui XTTS voice: {speaker_name}",
                }

            return result
        except Exception as e:
            return {"error": f"Could not fetch voices: {str(e)}"}

    def is_available(self) -> bool:
        """Check if Coqui-XTTS server is available"""
        try:
            # Check if we can get speakers - this is the most reliable test
            response = requests.get(f"{self.server_url}/studio_speakers", timeout=5)
            return response.status_code == 200
        except Exception:
            # Fallback to checking root endpoint
            try:
                response = requests.get(f"{self.server_url}/", timeout=5)
                return response.status_code == 200
            except Exception:
                return False


class OpenAITTSClient(TTSClient):
    """
    Client for OpenAI TTS API
    """

    def __init__(
        self,
        config: TTSConfig,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        super().__init__(config)
        from ..credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential(
            "OPENAI_API_KEY", "OpenAI TTS"
        )

        self.model = config.extra_params.get("model", OpenAIModel.TTS_1.value)
        self.base_url = "https://api.openai.com/v1"

    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """Convert text to speech using OpenAI TTS"""

        # Merge config with kwargs
        voice = kwargs.get("voice", self.config.voice or OpenAIVoice.ALLOY.value)
        output_path = kwargs.get("output_path", self.config.output_path)
        model = kwargs.get("model", self.model)
        speed = kwargs.get("speed", self.config.speed)

        # OpenAI TTS parameters
        request_data = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": self.config.output_format.value,
            "speed": speed,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/audio/speech",
                json=request_data,
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get(
                        "message", error_detail
                    )
                except Exception:
                    pass

                return TTSResponse(
                    success=False,
                    error_message=f"OpenAI TTS API error: {response.status_code} - {error_detail}",
                    provider=self.config.provider.value,
                    format_used=self.config.output_format.value,
                )

            # Get audio data
            audio_data = response.content
            duration_ms = int((time.time() - start_time) * 1000)

            # Save to file if output_path specified
            file_path = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                file_path = os.path.abspath(output_path)
            else:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{self.config.output_format.value}", delete=False
                ) as f:
                    f.write(audio_data)
                    file_path = f.name

            return TTSResponse(
                success=True,
                audio_file_path=file_path,
                audio_data=audio_data,
                duration_ms=duration_ms,
                provider=self.config.provider.value,
                voice_used=voice,
                format_used=self.config.output_format.value,
                metadata={"model": model, "speed": speed, "text_length": len(text)},
            )

        except requests.exceptions.RequestException as e:
            return TTSResponse(
                success=False,
                error_message=f"OpenAI API request failed: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )
        except Exception as e:
            return TTSResponse(
                success=False,
                error_message=f"OpenAI TTS error: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )

    def get_available_voices(self) -> Dict[str, Any]:
        """Get available OpenAI TTS voices"""
        return {
            voice.value: {
                "name": voice.value.title(),
                "description": f"OpenAI {voice.value} voice",
                "language": "en",
            }
            for voice in OpenAIVoice
        }

    def is_available(self) -> bool:
        """Check if OpenAI TTS is available"""
        if not self.api_key:
            return False

        try:
            # Make a simple request to check API availability
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url}/models", headers=headers, timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


class GoogleTTSClient(TTSClient):
    """
    Client for Google Cloud Text-to-Speech API
    """

    def __init__(
        self,
        config: TTSConfig,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        super().__init__(config)
        from ..credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.credentials_path = self.credential_manager.require_credential(
            "GOOGLE_APPLICATION_CREDENTIALS", "Google TTS"
        )

    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """Convert text to speech using Google TTS"""
        try:
            from google.cloud import texttospeech
        except ImportError:
            return TTSResponse(
                success=False,
                error_message="google-cloud-texttospeech package not installed. Run: pip install google-cloud-texttospeech",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )

        # Merge config with kwargs
        voice = kwargs.get("voice", self.config.voice)
        output_path = kwargs.get("output_path", self.config.output_path)
        language_code = kwargs.get("language_code", "en-US")

        try:
            start_time = time.time()

            # Initialize client
            client = texttospeech.TextToSpeechClient()

            # Configure voice
            voice_config = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice if voice else None,
                ssml_gender=getattr(
                    texttospeech.SsmlVoiceGender, kwargs.get("ssml_gender", "NEUTRAL")
                ),
            )

            # Configure audio
            audio_config = texttospeech.AudioConfig(
                audio_encoding=self._get_audio_encoding(),
                speaking_rate=self.config.speed,
                pitch=self.config.pitch or 0.0,
            )

            # Set input text
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Perform synthesis
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice_config, audio_config=audio_config
            )

            audio_data = response.audio_content
            duration_ms = int((time.time() - start_time) * 1000)

            # Save to file if output_path specified
            file_path = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                file_path = os.path.abspath(output_path)
            else:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{self.config.output_format.value}", delete=False
                ) as f:
                    f.write(audio_data)
                    file_path = f.name

            return TTSResponse(
                success=True,
                audio_file_path=file_path,
                audio_data=audio_data,
                duration_ms=duration_ms,
                provider=self.config.provider.value,
                voice_used=voice or "default",
                format_used=self.config.output_format.value,
                metadata={
                    "language_code": language_code,
                    "speaking_rate": self.config.speed,
                    "pitch": self.config.pitch,
                    "text_length": len(text),
                },
            )

        except Exception as e:
            return TTSResponse(
                success=False,
                error_message=f"Google TTS error: {str(e)}",
                provider=self.config.provider.value,
                format_used=self.config.output_format.value,
            )

    def get_available_voices(self) -> Dict[str, Any]:
        """Get available Google TTS voices"""
        try:
            from google.cloud import texttospeech

            client = texttospeech.TextToSpeechClient()
            voices = client.list_voices()

            available_voices = {}
            for voice in voices.voices:
                for language_code in voice.language_codes:
                    key = f"{voice.name}_{language_code}"
                    available_voices[key] = {
                        "name": voice.name,
                        "language": language_code,
                        "gender": voice.ssml_gender.name,
                        "natural_sample_rate": voice.natural_sample_rate_hertz,
                    }

            return available_voices

        except Exception as e:
            return {"error": f"Could not fetch Google voices: {str(e)}"}

    def is_available(self) -> bool:
        """Check if Google TTS is available"""
        if not self.credentials_path:
            return False

        try:
            from google.cloud import texttospeech

            client = texttospeech.TextToSpeechClient()
            # Try to list voices as a simple test
            client.list_voices()
            return True
        except Exception:
            return False

    def _get_audio_encoding(self):
        """Convert AudioFormat to Google TTS encoding"""
        try:
            from google.cloud import texttospeech

            format_mapping = {
                AudioFormat.MP3: texttospeech.AudioEncoding.MP3,
                AudioFormat.WAV: texttospeech.AudioEncoding.LINEAR16,
                AudioFormat.OGG: texttospeech.AudioEncoding.OGG_OPUS,
                # FLAC is not available in this version, fall back to MP3
                AudioFormat.FLAC: texttospeech.AudioEncoding.MP3,
            }

            return format_mapping.get(
                self.config.output_format, texttospeech.AudioEncoding.MP3
            )
        except ImportError:
            # Fallback if Google client not available
            return "MP3"
