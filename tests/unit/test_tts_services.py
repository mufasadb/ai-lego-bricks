"""
Unit tests for text-to-speech (TTS) services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Iterator

from tts.tts_service import TTSService
from tts.tts_factory import create_tts_service
from tts.streaming_tts_service import StreamingTTSService
from tts.tts_clients import OpenAITTSClient, GoogleTTSClient, CoquiXTTSClient
from tts.tts_factory import TTSServiceFactory
from tts.tts_types import TTSProvider, TTSConfig, AudioFormat


class TestTTSService:
    """Test suite for TTSService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_client = Mock()
        self.mock_client.generate_speech.return_value = b"fake_audio_data"
        
        self.service = TTSService(self.mock_client)
        
        self.test_config = TTSConfig(
            voice="alloy",
            speed=1.0,
            format=AudioFormat.MP3
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.client == self.mock_client
    
    def test_generate_speech(self):
        """Test basic speech generation."""
        text = "Hello, world!"
        audio_data = self.service.generate_speech(text)
        
        assert audio_data == b"fake_audio_data"
        self.mock_client.generate_speech.assert_called_once_with(text, config=None)
    
    def test_generate_speech_with_config(self):
        """Test speech generation with configuration."""
        text = "Test speech"
        audio_data = self.service.generate_speech(text, config=self.test_config)
        
        assert audio_data == b"fake_audio_data"
        self.mock_client.generate_speech.assert_called_once_with(text, config=self.test_config)
    
    def test_generate_speech_to_file(self):
        """Test generating speech and saving to file."""
        text = "Save this to file"
        output_path = "/tmp/test_audio.mp3"
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result_path = self.service.generate_speech_to_file(text, output_path)
            
            assert result_path == output_path
            mock_file.write.assert_called_once_with(b"fake_audio_data")
            self.mock_client.generate_speech.assert_called_once()
    
    def test_generate_speech_auto_filename(self):
        """Test generating speech with automatic filename."""
        text = "Auto filename test"
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('pathlib.Path.exists', return_value=False):
                result_path = self.service.generate_speech_to_file(text)
                
                assert result_path is not None
                assert result_path.endswith('.mp3')
                mock_open.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in speech generation."""
        self.mock_client.generate_speech.side_effect = Exception("TTS API Error")
        
        with pytest.raises(Exception, match="TTS API Error"):
            self.service.generate_speech("Test text")
    
    def test_text_preprocessing(self):
        """Test text preprocessing before TTS."""
        # Test with text that needs preprocessing
        text_with_issues = "Hello... world!!!   Extra   spaces."
        
        processed_audio = self.service.generate_speech(text_with_issues, preprocess=True)
        
        assert processed_audio == b"fake_audio_data"
        
        # Verify client was called with processed text
        call_args = self.mock_client.generate_speech.call_args[0]
        processed_text = call_args[0]
        
        # Should have normalized punctuation and spaces
        assert "..." not in processed_text
        assert "!!!" not in processed_text
        assert "   " not in processed_text
    
    def test_speech_chunking(self):
        """Test handling long text by chunking."""
        # Create very long text
        long_text = "This is a very long sentence. " * 100  # ~3000 characters
        
        # Mock client to return different audio for each chunk
        self.mock_client.generate_speech.side_effect = [
            b"chunk1_audio",
            b"chunk2_audio",
            b"chunk3_audio"
        ]
        
        audio_data = self.service.generate_speech(long_text, chunk_size=1000)
        
        # Should have been chunked and combined
        assert audio_data == b"chunk1_audiochunk2_audiochunk3_audio"
        assert self.mock_client.generate_speech.call_count == 3


class TestStreamingTTSService:
    """Test suite for StreamingTTSService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_client = Mock()
        self.mock_client.generate_speech_stream.return_value = iter([
            b"audio_chunk_1",
            b"audio_chunk_2", 
            b"audio_chunk_3"
        ])
        
        self.service = StreamingTTSService(self.mock_client)
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.client == self.mock_client
    
    def test_generate_speech_stream(self):
        """Test streaming speech generation."""
        text = "Stream this text"
        stream = self.service.generate_speech_stream(text)
        
        chunks = list(stream)
        assert chunks == [b"audio_chunk_1", b"audio_chunk_2", b"audio_chunk_3"]
        self.mock_client.generate_speech_stream.assert_called_once_with(text, config=None)
    
    def test_stream_to_file(self):
        """Test streaming audio directly to file."""
        text = "Stream to file"
        output_path = "/tmp/stream_audio.mp3"
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result_path = self.service.stream_to_file(text, output_path)
            
            assert result_path == output_path
            
            # Verify all chunks were written
            assert mock_file.write.call_count == 3
            mock_file.write.assert_any_call(b"audio_chunk_1")
            mock_file.write.assert_any_call(b"audio_chunk_2")
            mock_file.write.assert_any_call(b"audio_chunk_3")
    
    def test_real_time_playback(self):
        """Test real-time audio playback."""
        text = "Play this in real time"
        
        with patch('tts.streaming_tts_service.AudioPlayer') as mock_player:
            mock_player_instance = Mock()
            mock_player.return_value = mock_player_instance
            
            self.service.play_realtime(text)
            
            # Verify player was created and audio chunks were fed to it
            mock_player.assert_called_once()
            assert mock_player_instance.play_chunk.call_count == 3
    
    def test_stream_with_sentence_splitting(self):
        """Test streaming with sentence-by-sentence processing."""
        text = "First sentence. Second sentence. Third sentence."
        
        # Mock to return different audio for each sentence
        self.mock_client.generate_speech_stream.side_effect = [
            iter([b"sentence1_audio"]),
            iter([b"sentence2_audio"]),
            iter([b"sentence3_audio"])
        ]
        
        stream = self.service.generate_speech_stream(text, split_sentences=True)
        chunks = list(stream)
        
        # Should have combined all sentence audio
        assert b"sentence1_audio" in chunks
        assert b"sentence2_audio" in chunks
        assert b"sentence3_audio" in chunks
        
        # Should have called TTS for each sentence
        assert self.mock_client.generate_speech_stream.call_count == 3


class TestTTSServiceFactory:
    """Test suite for TTSServiceFactory."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = TTSServiceFactory()
        assert factory is not None
        assert hasattr(factory, 'create_client')
    
    @patch('tts.tts_factory.os.getenv')
    def test_create_openai_client(self, mock_getenv):
        """Test creating OpenAI TTS client."""
        mock_getenv.return_value = "test_api_key"
        
        factory = TTSServiceFactory()
        
        with patch('tts.tts_clients.OpenAITTSClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(TTSProvider.OPENAI)
            
            assert client == mock_client
            mock_client_class.assert_called_once_with("test_api_key")
    
    @patch('tts.tts_factory.os.getenv')
    def test_create_google_client(self, mock_getenv):
        """Test creating Google TTS client.""" 
        mock_getenv.return_value = "/path/to/credentials.json"
        
        factory = TTSServiceFactory()
        
        with patch('tts.tts_clients.GoogleTTSClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(TTSProvider.GOOGLE)
            
            assert client == mock_client
            mock_client_class.assert_called_once_with("/path/to/credentials.json")
    
    def test_create_coqui_client(self):
        """Test creating Coqui TTS client."""
        factory = TTSServiceFactory()
        
        with patch('tts.tts_clients.CoquiTTSClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(TTSProvider.COQUI)
            
            assert client == mock_client
            mock_client_class.assert_called_once()
    
    def test_create_invalid_provider(self):
        """Test creating client with invalid provider."""
        factory = TTSServiceFactory()
        
        with pytest.raises(ValueError, match="Unsupported TTS provider"):
            factory.create_client("invalid_provider")
    
    @patch('tts.tts_factory.os.getenv')
    def test_missing_credentials(self, mock_getenv):
        """Test creation fails when credentials are missing."""
        mock_getenv.return_value = None
        
        factory = TTSServiceFactory()
        
        with pytest.raises(ValueError, match="API key not found"):
            factory.create_client(TTSProvider.OPENAI)


class TestTTSClients:
    """Test suite for individual TTS clients."""
    
    def test_openai_client_initialization(self):
        """Test OpenAI TTS client initialization."""
        with patch('openai.OpenAI') as mock_openai:
            client = OpenAITTSClient("test_api_key")
            
            assert client is not None
            assert client.api_key == "test_api_key"
            mock_openai.assert_called_once_with(api_key="test_api_key")
    
    def test_openai_client_generate_speech(self):
        """Test OpenAI client speech generation."""
        with patch('openai.OpenAI') as mock_openai:
            mock_response = Mock()
            mock_response.content = b"openai_audio_data"
            mock_openai.return_value.audio.speech.create.return_value = mock_response
            
            client = OpenAITTSClient("test_api_key")
            audio_data = client.generate_speech("Hello world")
            
            assert audio_data == b"openai_audio_data"
            mock_openai.return_value.audio.speech.create.assert_called_once()
    
    def test_google_client_initialization(self):
        """Test Google TTS client initialization."""
        with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_google:
            with patch('google.auth.default') as mock_auth:
                mock_auth.return_value = (Mock(), "test_project")
                
                client = GoogleTTSClient("/path/to/credentials.json")
                
                assert client is not None
                mock_google.assert_called_once()
    
    def test_google_client_generate_speech(self):
        """Test Google client speech generation."""
        with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_google:
            with patch('google.auth.default'):
                mock_client = Mock()
                mock_response = Mock()
                mock_response.audio_content = b"google_audio_data"
                mock_client.synthesize_speech.return_value = mock_response
                mock_google.return_value = mock_client
                
                client = GoogleTTSClient("/path/to/credentials.json")
                audio_data = client.generate_speech("Hello world")
                
                assert audio_data == b"google_audio_data"
                mock_client.synthesize_speech.assert_called_once()
    
    def test_coqui_client_initialization(self):
        """Test Coqui TTS client initialization."""
        with patch('TTS.api.TTS') as mock_tts:
            client = CoquiTTSClient()
            
            assert client is not None
            mock_tts.assert_called_once()
    
    def test_coqui_client_generate_speech(self):
        """Test Coqui client speech generation."""
        with patch('TTS.api.TTS') as mock_tts:
            with patch('builtins.open', create=True) as mock_open:
                with patch('pathlib.Path.read_bytes', return_value=b"coqui_audio_data"):
                    mock_tts_instance = Mock()
                    mock_tts.return_value = mock_tts_instance
                    
                    client = CoquiTTSClient()
                    audio_data = client.generate_speech("Hello world")
                    
                    assert audio_data == b"coqui_audio_data"
                    mock_tts_instance.tts_to_file.assert_called_once()
    
    def test_client_error_handling(self):
        """Test client error handling."""
        # Test OpenAI client error
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.return_value.audio.speech.create.side_effect = Exception("API Error")
            
            client = OpenAITTSClient("test_api_key")
            
            with pytest.raises(Exception, match="API Error"):
                client.generate_speech("Hello world")
    
    def test_client_streaming(self):
        """Test client streaming functionality."""
        with patch('openai.OpenAI') as mock_openai:
            # Mock streaming response
            mock_stream = iter([b"chunk1", b"chunk2", b"chunk3"])
            mock_openai.return_value.audio.speech.create.return_value = mock_stream
            
            client = OpenAITTSClient("test_api_key")
            stream = client.generate_speech_stream("Hello world")
            
            chunks = list(stream)
            assert len(chunks) == 3
            assert chunks[0] == b"chunk1"
            assert chunks[1] == b"chunk2"
            assert chunks[2] == b"chunk3"


class TestTTSConfig:
    """Test suite for TTSConfig."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = TTSConfig(
            voice="alloy",
            speed=1.2,
            format=AudioFormat.MP3,
            quality="high"
        )
        
        assert config.voice == "alloy"
        assert config.speed == 1.2
        assert config.format == AudioFormat.MP3
        assert config.quality == "high"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = TTSConfig(speed=1.0)
        assert config.speed == 1.0
        
        # Test invalid speed
        with pytest.raises(ValueError):
            TTSConfig(speed=5.0)  # Should be between 0.25 and 4.0
        
        # Test invalid format
        with pytest.raises(ValueError):
            TTSConfig(format="invalid_format")
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = TTSConfig()
        
        assert config.voice == "alloy"  # Default voice
        assert config.speed == 1.0  # Default speed
        assert config.format == AudioFormat.MP3  # Default format
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = TTSConfig(voice="nova", speed=1.5)
        
        # Test to_dict method
        config_dict = config.to_dict()
        assert config_dict['voice'] == "nova"
        assert config_dict['speed'] == 1.5
        
        # Test from_dict method
        new_config = TTSConfig.from_dict(config_dict)
        assert new_config.voice == "nova"
        assert new_config.speed == 1.5


class TestCreateTTSService:
    """Test suite for create_tts_service factory function."""
    
    @patch('tts.tts_service.TTSServiceFactory')
    def test_create_service_auto_detection(self, mock_factory_class):
        """Test auto provider detection."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        with patch('tts.tts_service._detect_available_provider') as mock_detect:
            mock_detect.return_value = TTSProvider.OPENAI
            
            service = create_tts_service("auto")
            
            assert isinstance(service, TTSService)
            mock_factory.create_client.assert_called_once_with(TTSProvider.OPENAI)
    
    @patch('tts.tts_service.TTSServiceFactory')
    def test_create_service_specific_provider(self, mock_factory_class):
        """Test creating service with specific provider."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        service = create_tts_service("openai")
        
        assert isinstance(service, TTSService)
        mock_factory.create_client.assert_called_once_with(TTSProvider.OPENAI)
    
    @patch('tts.tts_service.StreamingTTSService')
    @patch('tts.tts_service.TTSServiceFactory')
    def test_create_streaming_service(self, mock_factory_class, mock_streaming_class):
        """Test creating streaming TTS service."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        mock_streaming_service = Mock()
        mock_streaming_class.return_value = mock_streaming_service
        
        service = create_tts_service("openai", streaming=True)
        
        assert service == mock_streaming_service
        mock_streaming_class.assert_called_once_with(mock_client)
    
    def test_create_service_invalid_provider(self):
        """Test creating service with invalid provider."""
        with pytest.raises(ValueError, match="Invalid TTS provider"):
            create_tts_service("invalid_provider")
    
    @patch('tts.tts_service.os.getenv')
    def test_provider_auto_detection(self, mock_getenv):
        """Test automatic provider detection logic."""
        # Test OpenAI detection
        mock_getenv.side_effect = lambda key: {
            'OPENAI_API_KEY': 'test_key'
        }.get(key)
        
        from tts.tts_service import _detect_available_provider
        provider = _detect_available_provider()
        assert provider == TTSProvider.OPENAI
        
        # Test Google detection when OpenAI not available
        mock_getenv.side_effect = lambda key: {
            'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json'
        }.get(key)
        
        provider = _detect_available_provider()
        assert provider == TTSProvider.GOOGLE
        
        # Test Coqui fallback when others not available
        mock_getenv.return_value = None
        
        provider = _detect_available_provider()
        assert provider == TTSProvider.COQUI


class TestTTSUtils:
    """Test suite for TTS utility functions."""
    
    def test_audio_format_detection(self):
        """Test audio format detection from file extension."""
        from tts.tts_service import detect_audio_format
        
        assert detect_audio_format("test.mp3") == AudioFormat.MP3
        assert detect_audio_format("test.wav") == AudioFormat.WAV
        assert detect_audio_format("test.ogg") == AudioFormat.OGG
        assert detect_audio_format("test.unknown") == AudioFormat.MP3  # Default
    
    def test_text_preprocessing(self):
        """Test text preprocessing for TTS."""
        from tts.tts_service import preprocess_text
        
        # Test basic preprocessing
        text = "Hello... world!!!   Too   many   spaces."
        processed = preprocess_text(text)
        
        assert "..." not in processed
        assert "!!!" not in processed
        assert "   " not in processed
        assert processed.endswith(".")
    
    def test_text_chunking(self):
        """Test text chunking for long content."""
        from tts.tts_service import chunk_text
        
        long_text = "This is a sentence. " * 100  # Very long text
        chunks = chunk_text(long_text, max_length=500)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500
            # Should end with sentence boundary
            assert chunk.endswith('.') or chunk.endswith('!') or chunk.endswith('?')
    
    def test_ssml_generation(self):
        """Test SSML generation for enhanced TTS."""
        from tts.tts_service import generate_ssml
        
        text = "Hello world"
        config = TTSConfig(voice="alloy", speed=1.2)
        
        ssml = generate_ssml(text, config)
        
        assert ssml.startswith('<speak>')
        assert ssml.endswith('</speak>')
        assert 'rate="1.2"' in ssml
        assert 'Hello world' in ssml
    
    def test_audio_metadata_extraction(self):
        """Test extracting metadata from audio data."""
        from tts.tts_service import extract_audio_metadata
        
        # Mock audio data (would normally be actual audio bytes)
        fake_audio_data = b"fake_mp3_header_data" + b"audio_content" * 100
        
        metadata = extract_audio_metadata(fake_audio_data, AudioFormat.MP3)
        
        assert "size" in metadata
        assert "format" in metadata
        assert metadata["format"] == "mp3"
        assert metadata["size"] == len(fake_audio_data)