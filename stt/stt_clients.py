"""
STT client implementations for different providers
"""

import os
import time
import requests
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..credentials import CredentialManager
from .stt_types import STTClient, STTConfig, STTResponse, WordTimestamp, SpeakerSegment


class FasterWhisperClient(STTClient):
    """
    Client for Faster Whisper local instance
    """

    def __init__(
        self,
        config: STTConfig,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        super().__init__(config)
        from ..credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.server_url = (
            config.extra_params.get("server_url") or "http://localhost:10300"
        )
        self.timeout = config.extra_params.get(
            "timeout", 120
        )  # Longer timeout for transcription
        self._languages_cache = None

    def speech_to_text(self, audio_file_path: str, **kwargs) -> STTResponse:
        """Convert speech to text using Faster Whisper"""

        # Merge config with kwargs
        language = kwargs.get("language", self.config.language or "auto")
        model = kwargs.get("model", self.config.model or "base")
        enable_word_timestamps = kwargs.get(
            "enable_word_timestamps", self.config.enable_word_timestamps
        )
        temperature = kwargs.get("temperature", self.config.temperature)
        beam_size = kwargs.get("beam_size", self.config.beam_size)

        if not os.path.isfile(audio_file_path):
            return STTResponse(
                success=False,
                error_message=f"Audio file not found: {audio_file_path}",
                provider=self.config.provider.value,
            )

        try:
            start_time = time.time()

            # Prepare the request
            with open(audio_file_path, "rb") as audio_file:
                files = {"audio": audio_file}
                data = {
                    "language": language,
                    "model": model,
                    "temperature": temperature,
                    "beam_size": beam_size,
                    "word_timestamps": enable_word_timestamps,
                }

                # Make request to Faster Whisper server
                response = requests.post(
                    f"{self.server_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

            if response.status_code != 200:
                return STTResponse(
                    success=False,
                    error_message=f"Faster Whisper server error: {response.status_code} - {response.text}",
                    provider=self.config.provider.value,
                )

            result = response.json()
            duration_seconds = time.time() - start_time

            # Parse word timestamps if available
            word_timestamps = []
            if enable_word_timestamps and "segments" in result:
                for segment in result.get("segments", []):
                    for word_info in segment.get("words", []):
                        word_timestamps.append(
                            WordTimestamp(
                                word=word_info["word"],
                                start_time=word_info["start"],
                                end_time=word_info["end"],
                                confidence=word_info.get("probability"),
                            )
                        )

            return STTResponse(
                success=True,
                transcript=result.get("text", ""),
                language_detected=result.get("language"),
                confidence=result.get("confidence"),
                word_timestamps=word_timestamps,
                duration_seconds=duration_seconds,
                provider=self.config.provider.value,
                model_used=model,
                metadata={
                    "segments": result.get("segments", []),
                    "processing_time": duration_seconds,
                },
            )

        except requests.exceptions.RequestException as e:
            return STTResponse(
                success=False,
                error_message=f"Network error connecting to Faster Whisper server: {str(e)}",
                provider=self.config.provider.value,
            )
        except Exception as e:
            return STTResponse(
                success=False,
                error_message=f"Faster Whisper transcription failed: {str(e)}",
                provider=self.config.provider.value,
            )

    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        if self._languages_cache is None:
            try:
                response = requests.get(f"{self.server_url}/languages", timeout=10)
                if response.status_code == 200:
                    self._languages_cache = response.json().get("languages", [])
                else:
                    # Fallback to common Whisper languages
                    self._languages_cache = [
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "ru",
                        "ja",
                        "ko",
                        "zh",
                        "ar",
                        "hi",
                        "tr",
                        "pl",
                        "nl",
                        "sv",
                        "da",
                        "no",
                        "fi",
                    ]
            except Exception:
                self._languages_cache = ["en", "auto"]

        return self._languages_cache

    def is_available(self) -> bool:
        """Check if Faster Whisper server is available"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class GoogleSTTClient(STTClient):
    """
    Client for Google Cloud Speech-to-Text
    """

    def __init__(
        self,
        config: STTConfig,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        super().__init__(config)
        from ..credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self._client = None
        self._languages_cache = None

        # Initialize Google client
        try:
            from google.cloud import speech

            credentials_path = self.credential_manager.get_credential(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
            if credentials_path and os.path.exists(credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            self._client = speech.SpeechClient()
        except ImportError:
            raise ImportError("google-cloud-speech is required for Google STT client")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google STT client: {str(e)}")

    def speech_to_text(self, audio_file_path: str, **kwargs) -> STTResponse:
        """Convert speech to text using Google Cloud Speech"""

        # Merge config with kwargs
        language = kwargs.get("language", self.config.language or "en-US")
        model = kwargs.get("model", self.config.model or "latest_long")
        enable_word_timestamps = kwargs.get(
            "enable_word_timestamps", self.config.enable_word_timestamps
        )
        enable_speaker_diarization = kwargs.get(
            "enable_speaker_diarization", self.config.enable_speaker_diarization
        )

        if not os.path.isfile(audio_file_path):
            return STTResponse(
                success=False,
                error_message=f"Audio file not found: {audio_file_path}",
                provider=self.config.provider.value,
            )

        try:
            from google.cloud import speech

            start_time = time.time()

            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            # Determine audio format
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            encoding_map = {
                ".wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
                ".flac": speech.RecognitionConfig.AudioEncoding.FLAC,
                ".mp3": speech.RecognitionConfig.AudioEncoding.MP3,
                ".ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                ".webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            }

            if file_ext not in encoding_map:
                return STTResponse(
                    success=False,
                    error_message=f"Unsupported audio format: {file_ext}",
                    provider=self.config.provider.value,
                )

            audio = speech.RecognitionAudio(content=content)

            # Build recognition config
            config_kwargs = {
                "encoding": encoding_map[file_ext],
                "language_code": language,
                "model": model,
                "enable_automatic_punctuation": True,
            }

            if enable_word_timestamps:
                config_kwargs["enable_word_time_offsets"] = True

            if enable_speaker_diarization:
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    max_speaker_count=self.config.max_speakers or 6,
                )
                config_kwargs["diarization_config"] = diarization_config

            recognition_config = speech.RecognitionConfig(**config_kwargs)

            # Perform recognition
            response = self._client.recognize(config=recognition_config, audio=audio)
            duration_seconds = time.time() - start_time

            if not response.results:
                return STTResponse(
                    success=True,
                    transcript="",
                    provider=self.config.provider.value,
                    model_used=model,
                    duration_seconds=duration_seconds,
                )

            # Extract transcript and metadata
            transcript = ""
            word_timestamps = []
            speaker_segments = []
            confidence_scores = []

            for result in response.results:
                alternative = result.alternatives[0]
                transcript += alternative.transcript + " "
                confidence_scores.append(alternative.confidence)

                # Extract word timestamps
                if enable_word_timestamps and hasattr(alternative, "words"):
                    for word_info in alternative.words:
                        word_timestamps.append(
                            WordTimestamp(
                                word=word_info.word,
                                start_time=word_info.start_time.total_seconds(),
                                end_time=word_info.end_time.total_seconds(),
                                confidence=getattr(word_info, "confidence", None),
                            )
                        )

                # Extract speaker segments
                if enable_speaker_diarization and hasattr(alternative, "words"):
                    current_speaker = None
                    segment_start = None
                    segment_words = []

                    for word_info in alternative.words:
                        speaker_tag = getattr(word_info, "speaker_tag", None)
                        if speaker_tag != current_speaker:
                            if current_speaker is not None:
                                speaker_segments.append(
                                    SpeakerSegment(
                                        speaker_id=f"speaker_{current_speaker}",
                                        start_time=segment_start,
                                        end_time=word_info.start_time.total_seconds(),
                                        text=" ".join(segment_words),
                                    )
                                )
                            current_speaker = speaker_tag
                            segment_start = word_info.start_time.total_seconds()
                            segment_words = [word_info.word]
                        else:
                            segment_words.append(word_info.word)

                    # Add final segment
                    if current_speaker is not None and segment_words:
                        speaker_segments.append(
                            SpeakerSegment(
                                speaker_id=f"speaker_{current_speaker}",
                                start_time=segment_start,
                                end_time=word_info.end_time.total_seconds(),
                                text=" ".join(segment_words),
                            )
                        )

            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else None
            )

            return STTResponse(
                success=True,
                transcript=transcript.strip(),
                language_detected=language,
                confidence=avg_confidence,
                word_timestamps=word_timestamps,
                speaker_segments=speaker_segments,
                duration_seconds=duration_seconds,
                provider=self.config.provider.value,
                model_used=model,
                metadata={"response": response, "processing_time": duration_seconds},
            )

        except Exception as e:
            return STTResponse(
                success=False,
                error_message=f"Google STT transcription failed: {str(e)}",
                provider=self.config.provider.value,
            )

    def get_supported_languages(self) -> List[str]:
        """Get supported languages for Google Speech"""
        if self._languages_cache is None:
            # Common Google Speech language codes
            self._languages_cache = [
                "en-US",
                "en-GB",
                "es-ES",
                "es-US",
                "fr-FR",
                "de-DE",
                "it-IT",
                "pt-BR",
                "pt-PT",
                "ru-RU",
                "ja-JP",
                "ko-KR",
                "zh-CN",
                "zh-TW",
                "ar-SA",
                "hi-IN",
                "tr-TR",
                "pl-PL",
                "nl-NL",
                "sv-SE",
                "da-DK",
                "no-NO",
                "fi-FI",
                "th-TH",
                "vi-VN",
                "id-ID",
                "ms-MY",
                "he-IL",
            ]

        return self._languages_cache

    def is_available(self) -> bool:
        """Check if Google STT is available"""
        try:
            return self._client is not None
        except Exception:
            return False
