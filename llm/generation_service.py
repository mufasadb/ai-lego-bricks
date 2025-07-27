"""
Pure generation service for stateless LLM interactions.
Optimized for one-shot prompt → response without conversation history.
"""

from typing import Optional, Any, Dict, Generator
from .llm_types import LLMProvider
from .llm_factory import LLMClientFactory
from .thinking_tokens_service import ThinkingTokensService, ThinkingTokensMode


class GenerationService:
    """
    Stateless LLM generation service for single prompt-response interactions.
    No conversation history or state management - optimized for speed and simplicity.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        thinking_tokens_mode: ThinkingTokensMode = ThinkingTokensMode.AUTO,
        thinking_tokens_delimiters: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize the generation service with a specific LLM client.

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            thinking_tokens_mode: How to handle thinking tokens in responses
            thinking_tokens_delimiters: Custom delimiters for thinking tokens
            **kwargs: Additional configuration parameters
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_tokens_mode = thinking_tokens_mode
        self.extra_params = kwargs

        # Initialize thinking tokens service
        self.thinking_tokens_service = ThinkingTokensService(
            delimiters=thinking_tokens_delimiters
        )

        # Create the underlying LLM client
        self.client = LLMClientFactory.create_text_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        thinking_tokens_mode: Optional[ThinkingTokensMode] = None,
        preserve_for_structured: bool = False,
        **override_params,
    ) -> str:
        """
        Generate a response to a single prompt.

        Args:
            prompt: The input prompt
            thinking_tokens_mode: Override thinking tokens mode for this call
            preserve_for_structured: Whether to preserve clean response for structured parsing
            **override_params: Optional parameters to override defaults for this call

        Returns:
            Generated response as string
        """
        # Use the client directly - no conversation context needed
        raw_response = self.client.chat(prompt)

        # Process thinking tokens
        mode = thinking_tokens_mode or self.thinking_tokens_mode
        result = self.thinking_tokens_service.process_response(
            raw_response, mode=mode, preserve_for_structured=preserve_for_structured
        )

        # Return appropriate response based on mode
        if mode == ThinkingTokensMode.SHOW:
            return result.original_response
        elif mode == ThinkingTokensMode.EXTRACT:
            return result.clean_response  # Return clean by default for extract mode
        else:
            return result.clean_response

    def generate_stream(
        self, prompt: str, **override_params
    ) -> Generator[str, None, str]:
        """
        Generate a streaming response to a single prompt.

        Args:
            prompt: The input prompt
            **override_params: Optional parameters to override defaults for this call

        Yields:
            str: Partial response chunks as they arrive

        Returns:
            str: Complete response when streaming is done
        """
        # Use the client's streaming method
        return self.client.chat_stream(prompt)

    def generate_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        thinking_tokens_mode: Optional[ThinkingTokensMode] = None,
        preserve_for_structured: bool = False,
        **override_params,
    ) -> str:
        """
        Generate a response with a system prompt.

        Args:
            prompt: The user prompt
            system_prompt: System instruction/context
            thinking_tokens_mode: Override thinking tokens mode for this call
            preserve_for_structured: Whether to preserve clean response for structured parsing
            **override_params: Optional parameters to override defaults for this call

        Returns:
            Generated response as string
        """
        # Combine system and user prompts appropriately for the provider
        if hasattr(self.client, "chat_with_system_prompt"):
            raw_response = self.client.chat_with_system_prompt(prompt, system_prompt)
        else:
            # Fallback: prepend system prompt to user prompt
            combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            raw_response = self.client.chat(combined_prompt)

        # Process thinking tokens
        mode = thinking_tokens_mode or self.thinking_tokens_mode
        result = self.thinking_tokens_service.process_response(
            raw_response, mode=mode, preserve_for_structured=preserve_for_structured
        )

        # Return appropriate response based on mode
        if mode == ThinkingTokensMode.SHOW:
            return result.original_response
        elif mode == ThinkingTokensMode.EXTRACT:
            return result.clean_response  # Return clean by default for extract mode
        else:
            return result.clean_response

    def generate_with_system_prompt_stream(
        self, prompt: str, system_prompt: str, **override_params
    ) -> Generator[str, None, str]:
        """
        Generate a streaming response with a system prompt.

        Args:
            prompt: The user prompt
            system_prompt: System instruction/context
            **override_params: Optional parameters to override defaults for this call

        Yields:
            str: Partial response chunks as they arrive

        Returns:
            str: Complete response when streaming is done
        """
        # Combine system and user prompts appropriately for the provider
        if hasattr(self.client, "chat_with_system_prompt_stream"):
            return self.client.chat_with_system_prompt_stream(prompt, system_prompt)
        else:
            # Fallback: prepend system prompt to user prompt
            combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return self.client.chat_stream(combined_prompt)

    def batch_generate(self, prompts: list[str], **override_params) -> list[str]:
        """
        Generate responses for multiple prompts independently.

        Args:
            prompts: List of input prompts
            **override_params: Optional parameters to override defaults for this call

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **override_params)
            responses.append(response)
        return responses

    def generate_with_thinking_tokens(
        self, prompt: str, system_prompt: Optional[str] = None, **override_params
    ) -> Dict[str, Any]:
        """
        Generate a response and return detailed thinking tokens information.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction/context
            **override_params: Optional parameters to override defaults for this call

        Returns:
            Dictionary with detailed thinking tokens information
        """
        # Generate raw response
        if system_prompt:
            if hasattr(self.client, "chat_with_system_prompt"):
                raw_response = self.client.chat_with_system_prompt(
                    prompt, system_prompt
                )
            else:
                combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                raw_response = self.client.chat(combined_prompt)
        else:
            raw_response = self.client.chat(prompt)

        # Process thinking tokens with EXTRACT mode to get all details
        result = self.thinking_tokens_service.process_response(
            raw_response, mode=ThinkingTokensMode.EXTRACT
        )

        # Return formatted result
        return self.thinking_tokens_service.format_extracted_response(result)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the generation service.

        Returns:
            Configuration dictionary
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking_tokens_mode": self.thinking_tokens_mode,
            "extra_params": self.extra_params,
        }

    def update_config(self, **config_updates):
        """
        Update the configuration and recreate the client if needed.

        Args:
            **config_updates: Configuration parameters to update
        """
        # Update instance variables
        for key, value in config_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_params[key] = value

        # Update thinking tokens service if delimiters changed
        if "thinking_tokens_delimiters" in config_updates:
            self.thinking_tokens_service = ThinkingTokensService(
                delimiters=config_updates["thinking_tokens_delimiters"]
            )

        # Recreate client with new configuration (exclude thinking tokens params)
        client_params = {
            k: v
            for k, v in self.extra_params.items()
            if k not in ["thinking_tokens_delimiters"]
        }
        self.client = LLMClientFactory.create_text_client(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **client_params,
        )


# Convenience functions for quick one-shot generation
def quick_generate(
    prompt: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> str:
    """
    Quick one-shot generation without creating a service instance.

    Args:
        prompt: The input prompt
        provider: LLM provider to use
        model: Model name (optional)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters

    Returns:
        Generated response as string
    """
    service = GenerationService(provider, model, temperature, max_tokens, **kwargs)
    return service.generate(prompt)


def quick_generate_ollama(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Quick generation using Ollama"""
    return quick_generate(prompt, LLMProvider.OLLAMA, model, **kwargs)


def quick_generate_gemini(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Quick generation using Gemini"""
    return quick_generate(prompt, LLMProvider.GEMINI, model, **kwargs)


def quick_generate_anthropic(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Quick generation using Anthropic"""
    return quick_generate(prompt, LLMProvider.ANTHROPIC, model, **kwargs)


# Streaming convenience functions
def quick_generate_stream(
    prompt: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> Generator[str, None, str]:
    """
    Quick one-shot streaming generation without creating a service instance.

    Args:
        prompt: The input prompt
        provider: LLM provider to use
        model: Model name (optional)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters

    Yields:
        str: Partial response chunks as they arrive

    Returns:
        str: Complete response when streaming is done
    """
    service = GenerationService(provider, model, temperature, max_tokens, **kwargs)
    return service.generate_stream(prompt)


def quick_generate_ollama_stream(
    prompt: str, model: Optional[str] = None, **kwargs
) -> Generator[str, None, str]:
    """Quick streaming generation using Ollama"""
    return quick_generate_stream(prompt, LLMProvider.OLLAMA, model, **kwargs)


def quick_generate_gemini_stream(
    prompt: str, model: Optional[str] = None, **kwargs
) -> Generator[str, None, str]:
    """Quick streaming generation using Gemini"""
    return quick_generate_stream(prompt, LLMProvider.GEMINI, model, **kwargs)


def quick_generate_anthropic_stream(
    prompt: str, model: Optional[str] = None, **kwargs
) -> Generator[str, None, str]:
    """Quick streaming generation using Anthropic"""
    return quick_generate_stream(prompt, LLMProvider.ANTHROPIC, model, **kwargs)


# Factory function for creating GenerationService instances
def create_generation_service(
    provider: str = "auto",
    model: Optional[str] = None,
    thinking_tokens_mode: ThinkingTokensMode = ThinkingTokensMode.AUTO,
    **kwargs,
) -> GenerationService:
    """
    Factory function to create a GenerationService instance.

    Args:
        provider: Provider name ("auto", "anthropic", "openai", "google", "ollama")
        model: Model name (optional)
        thinking_tokens_mode: How to handle thinking tokens in responses
        **kwargs: Additional configuration parameters

    Returns:
        GenerationService instance
    """
    # Map string provider to enum
    from .llm_types import LLMProvider

    provider_map = {
        "auto": LLMProvider.ANTHROPIC,  # Default to Anthropic
        "anthropic": LLMProvider.ANTHROPIC,
        "openai": LLMProvider.OPENAI,
        "google": LLMProvider.GEMINI,
        "gemini": LLMProvider.GEMINI,
        "ollama": LLMProvider.OLLAMA,
    }

    if provider in provider_map:
        llm_provider = provider_map[provider]
    else:
        # Default to anthropic if unknown
        llm_provider = LLMProvider.ANTHROPIC

    return GenerationService(
        llm_provider, model, thinking_tokens_mode=thinking_tokens_mode, **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Create a generation service and use it
    print("=== Generation Service Example ===")
    gen_service = GenerationService(LLMProvider.GEMINI, temperature=0.7)

    response = gen_service.generate("What is the capital of France?")
    print(f"Response: {response}")

    # Example 2: Use system prompt
    response_with_system = gen_service.generate_with_system_prompt(
        "What is the capital?", "You are a geography expert. Answer concisely."
    )
    print(f"With system prompt: {response_with_system}")

    # Example 3: Batch generation
    prompts = [
        "What is 2+2?",
        "What is the largest planet?",
        "What is the speed of light?",
    ]
    batch_responses = gen_service.batch_generate(prompts)
    for i, response in enumerate(batch_responses):
        print(f"Batch {i+1}: {response}")

    # Example 4: Quick generation functions
    print("\n=== Quick Generation Functions ===")
    quick_response = quick_generate_gemini("Tell me a fun fact about cats")
    print(f"Quick response: {quick_response}")
