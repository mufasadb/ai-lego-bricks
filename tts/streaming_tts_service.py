"""
Streaming TTS Service for real-time text-to-speech from LLM streams
"""

import re
import threading
from typing import Generator, Optional, Dict, Any, Callable, List
from queue import Queue, Empty
import time
import os

from .tts_service import TTSService


class StreamingTTSService:
    """
    Service that processes streaming text input and generates audio in real-time.

    Features:
    - Buffers streaming text chunks until complete sentences
    - Generates audio for sentence chunks as they complete
    - Supports configurable buffering strategies
    - Provides audio streaming callbacks
    """

    def __init__(
        self,
        tts_service: TTSService,
        sentence_buffer_size: int = 2,
        min_chunk_length: int = 20,
        max_buffer_time: float = 3.0,
        audio_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize streaming TTS service

        Args:
            tts_service: Base TTS service to use
            sentence_buffer_size: Number of sentences to buffer before generating audio
            min_chunk_length: Minimum characters before attempting to generate audio
            max_buffer_time: Maximum time to buffer before forcing audio generation
            audio_callback: Optional callback for when audio files are ready
        """
        self.tts_service = tts_service
        self.sentence_buffer_size = sentence_buffer_size
        self.min_chunk_length = min_chunk_length
        self.max_buffer_time = max_buffer_time
        self.audio_callback = audio_callback

        # Internal state
        self.text_buffer = ""
        self.sentence_buffer = []
        self.audio_files = []
        self.is_streaming = False
        self.last_buffer_time = time.time()

        # Threading for async audio generation
        self.audio_queue = Queue()
        self.audio_thread = None
        self.stop_audio_thread = False

    def stream_text_to_audio(
        self,
        text_generator: Generator[str, None, str],
        output_dir: str = "output/streaming_audio",
        voice: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, List[str]]:
        """
        Process streaming text and generate audio files in real-time

        Args:
            text_generator: Generator yielding text chunks
            output_dir: Directory to save audio files
            voice: Voice to use for TTS

        Yields:
            Dict with status updates and audio file paths

        Returns:
            List of all generated audio file paths
        """
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        self.audio_files = []
        self.text_buffer = ""
        self.sentence_buffer = []
        self.is_streaming = True
        self.last_buffer_time = time.time()

        # Start audio processing thread
        self._start_audio_thread(output_dir, voice)

        try:
            chunk_count = 0
            for chunk in text_generator:
                chunk_count += 1
                self.text_buffer += chunk

                # Check if we should generate audio
                sentences = self._extract_complete_sentences(self.text_buffer)

                if sentences:
                    # Add sentences to buffer
                    self.sentence_buffer.extend(sentences)

                    # Remove processed sentences from buffer
                    for sentence in sentences:
                        self.text_buffer = self.text_buffer.replace(sentence, "", 1)

                    # Generate audio if buffer is full or timeout reached
                    if (
                        len(self.sentence_buffer) >= self.sentence_buffer_size
                        or time.time() - self.last_buffer_time > self.max_buffer_time
                    ):

                        self._queue_audio_generation()
                        yield {
                            "status": "buffering",
                            "chunk_count": chunk_count,
                            "sentences_buffered": len(self.sentence_buffer),
                            "audio_files_generated": len(self.audio_files),
                        }

                # Yield progress update
                yield {
                    "status": "processing",
                    "chunk_count": chunk_count,
                    "buffer_length": len(self.text_buffer),
                    "sentences_buffered": len(self.sentence_buffer),
                    "audio_files_generated": len(self.audio_files),
                }

            # Process any remaining text
            if self.text_buffer.strip() or self.sentence_buffer:
                if self.text_buffer.strip():
                    self.sentence_buffer.append(self.text_buffer.strip())
                self._queue_audio_generation()

                # Wait for final audio generation
                while not self.audio_queue.empty():
                    time.sleep(0.1)

                yield {
                    "status": "finalizing",
                    "audio_files_generated": len(self.audio_files),
                }

        finally:
            # Cleanup
            self.is_streaming = False
            self._stop_audio_thread()

            yield {
                "status": "completed",
                "total_audio_files": len(self.audio_files),
                "audio_files": self.audio_files,
            }

        return self.audio_files

    def _extract_complete_sentences(self, text: str) -> List[str]:
        """
        Extract complete sentences from text buffer

        Args:
            text: Text to process

        Returns:
            List of complete sentences
        """
        # Look for sentence endings
        sentence_endings = re.findall(r"[^.!?]*[.!?]+", text)

        # Filter out very short sentences (likely incomplete)
        complete_sentences = []
        for sentence in sentence_endings:
            sentence = sentence.strip()
            if len(sentence) >= self.min_chunk_length:
                complete_sentences.append(sentence)

        return complete_sentences

    def _queue_audio_generation(self):
        """Queue buffered sentences for audio generation"""
        if self.sentence_buffer:
            text_to_generate = " ".join(self.sentence_buffer)
            self.audio_queue.put(text_to_generate)
            self.sentence_buffer = []
            self.last_buffer_time = time.time()

    def _start_audio_thread(self, output_dir: str, voice: Optional[str]):
        """Start the audio processing thread"""
        self.stop_audio_thread = False
        self.audio_thread = threading.Thread(
            target=self._audio_worker, args=(output_dir, voice), daemon=True
        )
        self.audio_thread.start()

    def _stop_audio_thread(self):
        """Stop the audio processing thread"""
        self.stop_audio_thread = True
        if self.audio_thread:
            self.audio_thread.join(timeout=5.0)

    def _audio_worker(self, output_dir: str, voice: Optional[str]):
        """Worker thread for processing audio generation queue"""
        audio_count = 0

        while not self.stop_audio_thread:
            try:
                # Get text from queue (with timeout)
                text = self.audio_queue.get(timeout=0.5)

                if text:
                    audio_count += 1
                    output_path = os.path.join(
                        output_dir, f"chunk_{audio_count:03d}.wav"
                    )

                    # Generate audio
                    response = self.tts_service.text_to_speech(
                        text=text, voice=voice, output_path=output_path
                    )

                    if response.success:
                        self.audio_files.append(response.audio_file_path)

                        # Call audio callback if provided
                        if self.audio_callback:
                            self.audio_callback(response.audio_file_path)

                    self.audio_queue.task_done()

            except Empty:
                # No text in queue, continue checking
                continue
            except Exception as e:
                print(f"Error in audio worker: {e}")
                continue


class StreamingLLMToTTSPipeline:
    """
    Complete pipeline for streaming LLM output to TTS
    """

    def __init__(
        self,
        llm_service,
        tts_service: TTSService,
        streaming_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pipeline

        Args:
            llm_service: LLM service with streaming capabilities
            tts_service: TTS service for audio generation
            streaming_config: Configuration for streaming behavior
        """
        self.llm_service = llm_service
        self.tts_service = tts_service

        # Default streaming config
        default_config = {
            "sentence_buffer_size": 2,
            "min_chunk_length": 20,
            "max_buffer_time": 3.0,
            "output_dir": "output/streaming_pipeline",
        }

        self.streaming_config = {**default_config, **(streaming_config or {})}

        # Initialize streaming TTS
        self.streaming_tts = StreamingTTSService(
            tts_service=tts_service,
            sentence_buffer_size=self.streaming_config["sentence_buffer_size"],
            min_chunk_length=self.streaming_config["min_chunk_length"],
            max_buffer_time=self.streaming_config["max_buffer_time"],
        )

    def stream_chat_to_audio(
        self,
        message: str,
        voice: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Stream LLM chat response directly to audio generation

        Args:
            message: Message to send to LLM
            voice: Voice to use for TTS
            progress_callback: Optional callback for progress updates

        Yields:
            Progress updates with text and audio status

        Returns:
            Final result with complete text and audio files
        """
        # Generate streaming response from LLM
        if hasattr(self.llm_service, "generate_stream"):
            text_stream = self.llm_service.generate_stream(message)
        elif hasattr(self.llm_service, "chat_stream"):
            text_stream = self.llm_service.chat_stream(message, [])
        else:
            raise ValueError("LLM service does not support streaming")

        # Process through streaming TTS
        complete_text = ""
        audio_files = []

        for progress in self.streaming_tts.stream_text_to_audio(
            text_stream, output_dir=self.streaming_config["output_dir"], voice=voice
        ):
            # Track complete text
            if progress["status"] == "completed":
                audio_files = progress["audio_files"]

            # Add text info to progress
            progress_info = {
                **progress,
                "message": message,
                "voice": voice,
                "complete_text": complete_text,
            }

            if progress_callback:
                progress_callback(progress_info)

            yield progress_info

        # Return final result
        return {
            "success": True,
            "message": message,
            "complete_text": complete_text,
            "audio_files": audio_files,
            "total_audio_files": len(audio_files),
            "voice_used": voice or self.tts_service.config.voice,
            "provider": self.tts_service.config.provider.value,
        }


# Convenience functions for quick usage
def create_streaming_pipeline(
    llm_provider: str = "ollama",
    tts_provider: str = "auto",
    streaming_config: Optional[Dict[str, Any]] = None,
):
    """
    Create a streaming pipeline with default services

    Args:
        llm_provider: LLM provider to use
        tts_provider: TTS provider to use
        streaming_config: Configuration for streaming behavior

    Returns:
        Configured StreamingLLMToTTSPipeline
    """
    # Import here to avoid circular imports
    from llm.generation_service import GenerationService
    from llm.llm_types import LLMProvider
    from tts.tts_factory import create_tts_service

    # Create services
    llm_service = GenerationService(LLMProvider(llm_provider.lower()))
    tts_service = create_tts_service(tts_provider)

    return StreamingLLMToTTSPipeline(llm_service, tts_service, streaming_config)


def quick_stream_chat_to_audio(
    message: str,
    llm_provider: str = "ollama",
    tts_provider: str = "auto",
    voice: Optional[str] = None,
    output_dir: str = "output/quick_streaming",
) -> Dict[str, Any]:
    """
    Quick function to stream chat response to audio

    Args:
        message: Message to send to LLM
        llm_provider: LLM provider to use
        tts_provider: TTS provider to use
        voice: Voice to use for TTS
        output_dir: Directory for audio files

    Returns:
        Result with audio files and metadata
    """
    pipeline = create_streaming_pipeline(
        llm_provider=llm_provider,
        tts_provider=tts_provider,
        streaming_config={"output_dir": output_dir},
    )

    # Process streaming pipeline
    result = None
    for progress in pipeline.stream_chat_to_audio(message, voice):
        print(
            f"Status: {progress['status']}, Audio files: {progress.get('audio_files_generated', 0)}"
        )
        result = progress

    return result
