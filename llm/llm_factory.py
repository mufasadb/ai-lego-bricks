from __future__ import annotations
from typing import Optional, Dict, Any, Union, Type, TypeVar, TYPE_CHECKING
from pydantic import BaseModel
from .llm_types import (
    LLMProvider,
    VisionProvider,
    LLMConfig,
    VisionConfig,
    TextLLMClient,
    VisionLLMClient,
    EmbeddingClient,
    StructuredLLMWrapper,
    StructuredResponseConfig,
)
from .text_clients import (
    OllamaTextClient,
    GeminiTextClient,
    AnthropicTextClient,
    OpenRouterTextClient,
)
from .vision_clients import GeminiVisionClient, LLaVAClient
from .embedding_client import SentenceTransformerEmbeddingClient

# Conditional import for credentials
try:
    from credentials import CredentialManager
except ImportError:
    try:
        from credentials import CredentialManager
    except ImportError:
        CredentialManager = None

T = TypeVar("T", bound=BaseModel)

# Import the new services - delay import to avoid circular imports
if TYPE_CHECKING:
    from .generation_service import GenerationService
    from chat.conversation_service import ConversationService


class LLMClientFactory:
    """Factory for creating LLM clients with unified interface"""

    @staticmethod
    def create_text_client(
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        credential_manager: Optional[CredentialManager] = None,
        **kwargs,
    ) -> TextLLMClient:
        """
        Create a text LLM client

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            credential_manager: Optional credential manager for explicit credential handling
            **kwargs: Additional configuration parameters

        Returns:
            TextLLMClient instance
        """
        config = LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )

        if provider == LLMProvider.OLLAMA:
            return OllamaTextClient(config, credential_manager)
        elif provider == LLMProvider.GEMINI:
            return GeminiTextClient(config, credential_manager)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicTextClient(config, credential_manager)
        elif provider == LLMProvider.OPENROUTER:
            return OpenRouterTextClient(config, credential_manager)
        else:
            raise ValueError(f"Unsupported text provider: {provider}")

    @staticmethod
    def create_vision_client(
        provider: VisionProvider,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ) -> VisionLLMClient:
        """
        Create a vision LLM client

        Args:
            provider: Vision provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            VisionLLMClient instance
        """
        config = VisionConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )

        if provider == VisionProvider.GEMINI_VISION:
            return GeminiVisionClient(config)
        elif provider == VisionProvider.LLAVA:
            return LLaVAClient(config)
        else:
            raise ValueError(f"Unsupported vision provider: {provider}")

    @staticmethod
    def create_embedding_client(model_name: Optional[str] = None) -> EmbeddingClient:
        """
        Create an embedding client

        Args:
            model_name: Embedding model name (optional, uses defaults)

        Returns:
            EmbeddingClient instance
        """
        return SentenceTransformerEmbeddingClient(model_name)

    @staticmethod
    def create_structured_client(
        provider: LLMProvider,
        schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        config: Optional[StructuredResponseConfig] = None,
        **kwargs,
    ) -> StructuredLLMWrapper[T]:
        """
        Create a structured LLM client that enforces a specific response schema

        Args:
            provider: LLM provider to use
            schema: Pydantic model defining the expected response structure
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            config: Configuration for structured response behavior
            **kwargs: Additional configuration parameters

        Returns:
            StructuredLLMWrapper instance that returns validated Pydantic models
        """
        # Create the base text client
        base_client = LLMClientFactory.create_text_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Wrap it with structured response enforcement
        return StructuredLLMWrapper(base_client, schema, config)

    @staticmethod
    def create_structured_client_from_client(
        client: TextLLMClient,
        schema: Type[T],
        config: Optional[StructuredResponseConfig] = None,
    ) -> StructuredLLMWrapper[T]:
        """
        Wrap an existing text client with structured response enforcement

        Args:
            client: Existing TextLLMClient to wrap
            schema: Pydantic model defining the expected response structure
            config: Configuration for structured response behavior

        Returns:
            StructuredLLMWrapper instance
        """
        return StructuredLLMWrapper(client, schema, config)

    @staticmethod
    def get_available_providers() -> Dict[str, Any]:
        """Get information about available providers"""
        return {
            "text_providers": [p.value for p in LLMProvider],
            "vision_providers": [p.value for p in VisionProvider],
            "embedding_models": [
                "all-MiniLM-L6-v2",  # Default
                "all-mpnet-base-v2",  # Better quality
                "multi-qa-MiniLM-L6-cos-v1",  # Good for Q&A
                "paraphrase-MiniLM-L6-v2",  # Good for paraphrasing
            ],
        }

    @staticmethod
    def switch_client_model(
        client: Union[TextLLMClient, VisionLLMClient], new_model: str
    ) -> bool:
        """
        Switch the model for an existing client

        Args:
            client: The LLM client to update
            new_model: The new model name to switch to

        Returns:
            True if successful, False otherwise
        """
        if hasattr(client, "switch_model"):
            return client.switch_model(new_model)
        else:
            raise ValueError(
                f"Client {type(client).__name__} does not support model switching"
            )

    @staticmethod
    def get_client_model(
        client: Union[TextLLMClient, VisionLLMClient]
    ) -> Optional[str]:
        """
        Get the current model name from a client

        Args:
            client: The LLM client to query

        Returns:
            Current model name or None if not available
        """
        if hasattr(client, "get_current_model"):
            return client.get_current_model()
        else:
            return getattr(client, "model", None)

    @staticmethod
    def create_switchable_text_client(
        provider: LLMProvider,
        initial_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> TextLLMClient:
        """
        Create a text client specifically designed for model switching

        Args:
            provider: LLM provider to use
            initial_model: Initial model name (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            TextLLMClient instance with switching capabilities
        """
        if provider not in [LLMProvider.OLLAMA, LLMProvider.ANTHROPIC, LLMProvider.OPENROUTER]:
            raise ValueError(
                f"Model switching only supported for Ollama, Anthropic, and OpenRouter providers, not {provider}"
            )

        return LLMClientFactory.create_text_client(
            provider, initial_model, temperature, max_tokens, **kwargs
        )

    @staticmethod
    def create_switchable_vision_client(
        provider: VisionProvider,
        initial_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ) -> VisionLLMClient:
        """
        Create a vision client specifically designed for model switching

        Args:
            provider: Vision provider to use
            initial_model: Initial model name (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            VisionLLMClient instance with switching capabilities
        """
        if provider != VisionProvider.LLAVA:
            raise ValueError(
                f"Model switching only supported for LLaVA provider, not {provider}"
            )

        return LLMClientFactory.create_vision_client(
            provider, initial_model, temperature, max_tokens, **kwargs
        )

    @staticmethod
    def create_generation_service(
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> "GenerationService":
        """
        Create a generation service for stateless one-shot LLM interactions.

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            GenerationService instance
        """
        from .generation_service import GenerationService

        return GenerationService(provider, model, temperature, max_tokens, **kwargs)

    @staticmethod
    def create_conversation_service(
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> "ConversationService":
        """
        Create a conversation service for multi-turn conversations with rich state management.

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            conversation_id: Optional existing conversation ID to continue
            **kwargs: Additional configuration parameters

        Returns:
            ConversationService instance
        """
        try:
            from chat.conversation_service import ConversationService
        except ImportError:
            from chat.conversation_service import ConversationService

        return ConversationService(
            provider, model, temperature, max_tokens, conversation_id, **kwargs
        )


# Convenience functions for quick access
def create_text_client(
    provider: str, credential_manager: Optional[CredentialManager] = None, **kwargs
) -> TextLLMClient:
    """Convenience function to create text client from string"""
    return LLMClientFactory.create_text_client(
        LLMProvider(provider), credential_manager=credential_manager, **kwargs
    )


def create_vision_client(provider: str, **kwargs) -> VisionLLMClient:
    """Convenience function to create vision client from string"""
    return LLMClientFactory.create_vision_client(VisionProvider(provider), **kwargs)


def create_embedding_client(**kwargs) -> EmbeddingClient:
    """Convenience function to create embedding client"""
    return LLMClientFactory.create_embedding_client(**kwargs)


# Model switching convenience functions
def switch_client_model(
    client: Union[TextLLMClient, VisionLLMClient], new_model: str
) -> bool:
    """Convenience function to switch model for any client"""
    return LLMClientFactory.switch_client_model(client, new_model)


def get_client_model(client: Union[TextLLMClient, VisionLLMClient]) -> Optional[str]:
    """Convenience function to get current model from any client"""
    return LLMClientFactory.get_client_model(client)


def create_switchable_text_client(provider: str, **kwargs) -> TextLLMClient:
    """Convenience function to create switchable text client from string"""
    return LLMClientFactory.create_switchable_text_client(
        LLMProvider(provider), **kwargs
    )


def create_switchable_vision_client(provider: str, **kwargs) -> VisionLLMClient:
    """Convenience function to create switchable vision client from string"""
    return LLMClientFactory.create_switchable_vision_client(
        VisionProvider(provider), **kwargs
    )


# Structured response convenience functions
def create_structured_client(
    provider: str, schema: Type[T], **kwargs
) -> StructuredLLMWrapper[T]:
    """Convenience function to create structured client from string"""
    return LLMClientFactory.create_structured_client(
        LLMProvider(provider), schema, **kwargs
    )


def create_structured_client_from_client(
    client: TextLLMClient, schema: Type[T], **kwargs
) -> StructuredLLMWrapper[T]:
    """Convenience function to wrap existing client with structured responses"""
    return LLMClientFactory.create_structured_client_from_client(
        client, schema, **kwargs
    )


# New service convenience functions
def create_generation_service(
    provider: str, model: Optional[str] = None, **kwargs
) -> "GenerationService":
    """Convenience function to create generation service from string"""
    return LLMClientFactory.create_generation_service(
        LLMProvider(provider), model, **kwargs
    )


def create_conversation_service(
    provider: str, model: Optional[str] = None, **kwargs
) -> "ConversationService":
    """Convenience function to create conversation service from string"""
    return LLMClientFactory.create_conversation_service(
        LLMProvider(provider), model, **kwargs
    )


# Provider-specific convenience functions
def create_ollama_generation(
    model: Optional[str] = None, **kwargs
) -> "GenerationService":
    """Create Ollama generation service"""
    return create_generation_service("ollama", model, **kwargs)


def create_gemini_generation(
    model: Optional[str] = None, **kwargs
) -> "GenerationService":
    """Create Gemini generation service"""
    return create_generation_service("gemini", model, **kwargs)


def create_anthropic_generation(
    model: Optional[str] = None, **kwargs
) -> "GenerationService":
    """Create Anthropic generation service"""
    return create_generation_service("anthropic", model, **kwargs)


def create_ollama_conversation(
    model: Optional[str] = None, **kwargs
) -> "ConversationService":
    """Create Ollama conversation service"""
    return create_conversation_service("ollama", model, **kwargs)


def create_gemini_conversation(
    model: Optional[str] = None, **kwargs
) -> "ConversationService":
    """Create Gemini conversation service"""
    return create_conversation_service("gemini", model, **kwargs)


def create_anthropic_conversation(
    model: Optional[str] = None, **kwargs
) -> "ConversationService":
    """Create Anthropic conversation service"""
    return create_conversation_service("anthropic", model, **kwargs)


def create_openrouter_generation(
    model: Optional[str] = None, **kwargs
) -> "GenerationService":
    """Create OpenRouter generation service"""
    return create_generation_service("openrouter", model, **kwargs)


def create_openrouter_conversation(
    model: Optional[str] = None, **kwargs
) -> "ConversationService":
    """Create OpenRouter conversation service"""
    return create_conversation_service("openrouter", model, **kwargs)
