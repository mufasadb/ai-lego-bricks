"""
Unit tests for LLM services and text generation functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Iterator

from llm.generation_service import GenerationService, create_generation_service
from llm.text_clients import AnthropicTextClient, GeminiTextClient, OllamaTextClient
from llm.llm_factory import LLMClientFactory
from llm.llm_types import LLMProvider, GenerationConfig
from llm.model_manager import ModelManager


class TestGenerationService:
    """Test suite for GenerationService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_client = Mock()
        self.mock_client.generate.return_value = "Test response"
        self.mock_client.generate_stream.return_value = iter(["Hello", " world", "!"])
        
        self.service = GenerationService(self.mock_client)
        
        self.test_config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.client == self.mock_client
    
    def test_generate_text(self):
        """Test basic text generation."""
        prompt = "Hello, how are you?"
        response = self.service.generate(prompt)
        
        assert response == "Test response"
        self.mock_client.generate.assert_called_once_with(prompt, config=None)
    
    def test_generate_with_config(self):
        """Test text generation with configuration."""
        prompt = "Test prompt"
        response = self.service.generate(prompt, config=self.test_config)
        
        assert response == "Test response"
        self.mock_client.generate.assert_called_once_with(prompt, config=self.test_config)
    
    def test_generate_stream(self):
        """Test streaming text generation."""
        prompt = "Tell me a story"
        stream = self.service.generate_stream(prompt)
        
        chunks = list(stream)
        assert chunks == ["Hello", " world", "!"]
        self.mock_client.generate_stream.assert_called_once_with(prompt, config=None)
    
    def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello"
        
        response = self.service.generate(
            user_prompt, 
            system_prompt=system_prompt,
            config=self.test_config
        )
        
        assert response == "Test response"
        self.mock_client.generate.assert_called_once()
        
        # Verify system prompt was passed
        call_args = self.mock_client.generate.call_args
        assert system_prompt in str(call_args)
    
    def test_error_handling(self):
        """Test error handling in generation."""
        self.mock_client.generate.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            self.service.generate("Test prompt")
    
    def test_retry_mechanism(self):
        """Test retry mechanism on failures."""
        # First call fails, second succeeds
        self.mock_client.generate.side_effect = [
            Exception("Temporary error"),
            "Success response"
        ]
        
        with patch('llm.generation_service.time.sleep'):  # Mock sleep to speed up test
            with patch.object(self.service, 'max_retries', 2):
                response = self.service.generate("Test prompt")
                assert response == "Success response"
                assert self.mock_client.generate.call_count == 2


class TestLLMFactory:
    """Test suite for LLMFactory."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = LLMFactory()
        assert factory is not None
        assert hasattr(factory, 'create_client')
    
    @patch('llm.llm_factory.os.getenv')
    def test_create_anthropic_client(self, mock_getenv):
        """Test creating Anthropic client."""
        mock_getenv.return_value = "test_api_key"
        
        factory = LLMFactory()
        
        with patch('llm.text_clients.AnthropicClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(LLMProvider.ANTHROPIC)
            
            assert client == mock_client
            mock_client_class.assert_called_once_with("test_api_key")
    
    @patch('llm.llm_factory.os.getenv')
    def test_create_openai_client(self, mock_getenv):
        """Test creating OpenAI client."""
        mock_getenv.return_value = "test_api_key"
        
        factory = LLMFactory()
        
        with patch('llm.text_clients.OpenAIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(LLMProvider.OPENAI)
            
            assert client == mock_client
            mock_client_class.assert_called_once_with("test_api_key")
    
    @patch('llm.llm_factory.os.getenv')
    def test_create_google_client(self, mock_getenv):
        """Test creating Google client."""
        mock_getenv.return_value = "test_api_key"
        
        factory = LLMFactory()
        
        with patch('llm.text_clients.GoogleClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(LLMProvider.GOOGLE)
            
            assert client == mock_client
            mock_client_class.assert_called_once_with("test_api_key")
    
    @patch('llm.llm_factory.os.getenv')
    def test_create_ollama_client(self, mock_getenv):
        """Test creating Ollama client."""
        mock_getenv.side_effect = lambda key: {
            'OLLAMA_URL': 'http://localhost:11434',
            'OLLAMA_DEFAULT_MODEL': 'llama2'
        }.get(key)
        
        factory = LLMFactory()
        
        with patch('llm.text_clients.OllamaClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = factory.create_client(LLMProvider.OLLAMA)
            
            assert client == mock_client
            mock_client_class.assert_called_once()
    
    def test_create_invalid_provider(self):
        """Test creating client with invalid provider."""
        factory = LLMFactory()
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            factory.create_client("invalid_provider")
    
    @patch('llm.llm_factory.os.getenv')
    def test_missing_api_key(self, mock_getenv):
        """Test creation fails when API key is missing."""
        mock_getenv.return_value = None
        
        factory = LLMFactory()
        
        with pytest.raises(ValueError, match="API key not found"):
            factory.create_client(LLMProvider.ANTHROPIC)


class TestLLMClients:
    """Test suite for individual LLM clients."""
    
    def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization."""
        with patch('anthropic.Client') as mock_anthropic:
            client = AnthropicClient("test_api_key")
            
            assert client is not None
            assert client.api_key == "test_api_key"
            mock_anthropic.assert_called_once_with(api_key="test_api_key")
    
    def test_anthropic_client_generate(self):
        """Test Anthropic client text generation."""
        with patch('anthropic.Client') as mock_anthropic:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_anthropic.return_value.messages.create.return_value = mock_response
            
            client = AnthropicClient("test_api_key")
            response = client.generate("Hello world")
            
            assert response == "Test response"
            mock_anthropic.return_value.messages.create.assert_called_once()
    
    def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        with patch('openai.OpenAI') as mock_openai:
            client = OpenAIClient("test_api_key")
            
            assert client is not None
            assert client.api_key == "test_api_key"
            mock_openai.assert_called_once_with(api_key="test_api_key")
    
    def test_openai_client_generate(self):
        """Test OpenAI client text generation."""
        with patch('openai.OpenAI') as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Test response"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            
            client = OpenAIClient("test_api_key")
            response = client.generate("Hello world")
            
            assert response == "Test response"
            mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_google_client_initialization(self):
        """Test Google client initialization."""
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                client = GoogleClient("test_api_key")
                
                assert client is not None
                assert client.api_key == "test_api_key"
                mock_configure.assert_called_once_with(api_key="test_api_key")
                mock_model.assert_called_once()
    
    def test_google_client_generate(self):
        """Test Google client text generation."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model
                
                client = GoogleClient("test_api_key")
                response = client.generate("Hello world")
                
                assert response == "Test response"
                mock_model.generate_content.assert_called_once_with("Hello world")
    
    def test_ollama_client_initialization(self):
        """Test Ollama client initialization."""
        client = OllamaClient("http://localhost:11434", "llama2")
        
        assert client is not None
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama2"
    
    def test_ollama_client_generate(self):
        """Test Ollama client text generation."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            client = OllamaClient("http://localhost:11434", "llama2")
            response = client.generate("Hello world")
            
            assert response == "Test response"
            mock_post.assert_called_once()
    
    def test_client_error_handling(self):
        """Test client error handling."""
        # Test Anthropic client error
        with patch('anthropic.Client') as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
            
            client = AnthropicClient("test_api_key")
            
            with pytest.raises(Exception, match="API Error"):
                client.generate("Hello world")
    
    def test_client_streaming(self):
        """Test client streaming functionality."""
        with patch('anthropic.Client') as mock_anthropic:
            # Mock streaming response
            mock_stream = [
                Mock(delta=Mock(text="Hello")),
                Mock(delta=Mock(text=" world")),
                Mock(delta=Mock(text="!"))
            ]
            mock_anthropic.return_value.messages.create.return_value = iter(mock_stream)
            
            client = AnthropicClient("test_api_key")
            stream = client.generate_stream("Hello world")
            
            chunks = list(stream)
            assert len(chunks) == 3
            assert chunks[0] == "Hello"
            assert chunks[1] == " world"
            assert chunks[2] == "!"


class TestModelManager:
    """Test suite for ModelManager."""
    
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        manager = ModelManager()
        assert manager is not None
        assert hasattr(manager, 'get_available_models')
        assert hasattr(manager, 'switch_model')
    
    def test_get_available_models(self):
        """Test getting available models."""
        manager = ModelManager()
        
        with patch.object(manager, '_check_provider_availability') as mock_check:
            mock_check.return_value = True
            
            models = manager.get_available_models()
            
            assert isinstance(models, dict)
            assert len(models) > 0
            
            # Check that common providers are included
            for provider in ['anthropic', 'openai', 'google']:
                if provider in models:
                    assert isinstance(models[provider], list)
    
    def test_switch_model(self):
        """Test switching between models."""
        manager = ModelManager()
        
        # Test switching to valid model
        with patch.object(manager, '_validate_model') as mock_validate:
            mock_validate.return_value = True
            
            result = manager.switch_model('anthropic', 'claude-3-sonnet')
            assert result is True
    
    def test_model_validation(self):
        """Test model validation."""
        manager = ModelManager()
        
        # Test invalid provider
        with pytest.raises(ValueError, match="Invalid provider"):
            manager.switch_model('invalid_provider', 'some_model')
        
        # Test invalid model
        with patch.object(manager, '_validate_model') as mock_validate:
            mock_validate.return_value = False
            
            with pytest.raises(ValueError, match="Invalid model"):
                manager.switch_model('anthropic', 'invalid_model')


class TestCreateGenerationService:
    """Test suite for create_generation_service factory function."""
    
    @patch('llm.generation_service.LLMFactory')
    def test_create_service_auto_detection(self, mock_factory_class):
        """Test auto provider detection."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        with patch('llm.generation_service._detect_available_provider') as mock_detect:
            mock_detect.return_value = LLMProvider.ANTHROPIC
            
            service = create_generation_service("auto")
            
            assert isinstance(service, GenerationService)
            mock_factory.create_client.assert_called_once_with(LLMProvider.ANTHROPIC)
    
    @patch('llm.generation_service.LLMFactory')
    def test_create_service_specific_provider(self, mock_factory_class):
        """Test creating service with specific provider."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        service = create_generation_service("anthropic")
        
        assert isinstance(service, GenerationService)
        mock_factory.create_client.assert_called_once_with(LLMProvider.ANTHROPIC)
    
    def test_create_service_invalid_provider(self):
        """Test creating service with invalid provider."""
        with pytest.raises(ValueError, match="Invalid provider"):
            create_generation_service("invalid_provider")
    
    @patch('llm.generation_service.os.getenv')
    def test_provider_auto_detection(self, mock_getenv):
        """Test automatic provider detection logic."""
        # Test Anthropic detection
        mock_getenv.side_effect = lambda key: {
            'ANTHROPIC_API_KEY': 'test_key'
        }.get(key)
        
        from llm.generation_service import _detect_available_provider
        provider = _detect_available_provider()
        assert provider == LLMProvider.ANTHROPIC
        
        # Test OpenAI detection when Anthropic not available
        mock_getenv.side_effect = lambda key: {
            'OPENAI_API_KEY': 'test_key'
        }.get(key)
        
        provider = _detect_available_provider()
        assert provider == LLMProvider.OPENAI
    
    @patch('llm.generation_service.os.getenv')
    def test_no_providers_available(self, mock_getenv):
        """Test behavior when no providers are available."""
        mock_getenv.return_value = None
        
        from llm.generation_service import _detect_available_provider
        
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            _detect_available_provider()


class TestGenerationConfig:
    """Test suite for GenerationConfig."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1
        )
        
        assert config.max_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = GenerationConfig(temperature=0.7)
        assert config.temperature == 0.7
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            GenerationConfig(temperature=2.0)  # Should be between 0 and 1
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            GenerationConfig(top_p=1.5)  # Should be between 0 and 1
        
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            GenerationConfig(max_tokens=-1)  # Should be positive
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = GenerationConfig()
        
        assert config.max_tokens == 1000  # Default value
        assert config.temperature == 0.7  # Default value
        assert config.top_p == 1.0  # Default value
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = GenerationConfig(max_tokens=500, temperature=0.5)
        
        # Test to_dict method
        config_dict = config.to_dict()
        assert config_dict['max_tokens'] == 500
        assert config_dict['temperature'] == 0.5
        
        # Test from_dict method
        new_config = GenerationConfig.from_dict(config_dict)
        assert new_config.max_tokens == 500
        assert new_config.temperature == 0.5