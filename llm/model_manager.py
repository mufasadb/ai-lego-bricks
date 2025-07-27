import os
from typing import List, Dict, Any, Optional, Union
import httpx
from .llm_types import TextLLMClient, VisionLLMClient


class ModelManager:
    """Manages Ollama model operations and switching"""

    def __init__(self, ollama_url: Optional[str] = None):
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434"
        )

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama

        Returns:
            List of model dictionaries with name, size, and modified date
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()
                return response.json().get("models", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")

    def get_model_names(self) -> List[str]:
        """
        Get just the model names as a list of strings

        Returns:
            List of model names
        """
        models = self.list_available_models()
        return [model["name"] for model in models]

    def validate_model(self, model_name: str) -> bool:
        """
        Check if a model exists in Ollama

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model exists, False otherwise
        """
        try:
            available_models = self.get_model_names()
            return model_name in available_models
        except Exception:
            return False

    def switch_model(
        self, client: Union[TextLLMClient, VisionLLMClient], new_model: str
    ) -> bool:
        """
        Switch the model for an existing client

        Args:
            client: The LLM client to update
            new_model: The new model name to switch to

        Returns:
            True if successful, False otherwise
        """
        if not self.validate_model(new_model):
            raise ValueError(f"Model '{new_model}' is not available in Ollama")

        if hasattr(client, "model"):
            client.model = new_model
            return True
        else:
            raise ValueError(
                f"Client {type(client).__name__} does not support model switching"
            )

    def get_current_model(
        self, client: Union[TextLLMClient, VisionLLMClient]
    ) -> Optional[str]:
        """
        Get the current model name from a client

        Args:
            client: The LLM client to query

        Returns:
            Current model name or None if not available
        """
        return getattr(client, "model", None)

    def get_models_by_type(self) -> Dict[str, List[str]]:
        """
        Categorize models by type (text, vision, code, etc.)

        Returns:
            Dictionary with model categories as keys and model lists as values
        """
        models = self.get_model_names()
        categorized = {"text": [], "vision": [], "code": [], "other": []}

        for model in models:
            model_lower = model.lower()
            # Vision/VL models - expanded detection
            if any(
                vl_indicator in model_lower
                for vl_indicator in [
                    "llava",
                    "vision",
                    "vl",
                    "multimodal",
                    "bakllava",
                    "moondream",
                    "gemma-3n",
                    "3n",
                ]
            ):
                categorized["vision"].append(model)
            # Code models
            elif "code" in model_lower or "codellama" in model_lower:
                categorized["code"].append(model)
            # Text models - expanded detection including mistral
            elif any(
                text_indicator in model_lower
                for text_indicator in [
                    "llama",
                    "mistral",
                    "gemma",
                    "phi",
                    "qwen",
                    "neural",
                    "chat",
                ]
            ):
                categorized["text"].append(model)
            else:
                categorized["other"].append(model)

        return categorized

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            with httpx.Client(timeout=300.0) as client:  # Longer timeout for downloads
                response = client.post(
                    f"{self.ollama_url}/api/pull", json={"name": model_name}
                )
                response.raise_for_status()
                return True
        except Exception as e:
            print(f"Failed to pull model '{model_name}': {e}")
            return False


# Global instance for convenience
_model_manager = ModelManager()


# Convenience functions
def list_available_models() -> List[Dict[str, Any]]:
    """List all available models in Ollama"""
    return _model_manager.list_available_models()


def get_model_names() -> List[str]:
    """Get just the model names as a list of strings"""
    return _model_manager.get_model_names()


def validate_model(model_name: str) -> bool:
    """Check if a model exists in Ollama"""
    return _model_manager.validate_model(model_name)


def switch_model(client: Union[TextLLMClient, VisionLLMClient], new_model: str) -> bool:
    """Switch the model for an existing client"""
    return _model_manager.switch_model(client, new_model)


def get_current_model(client: Union[TextLLMClient, VisionLLMClient]) -> Optional[str]:
    """Get the current model name from a client"""
    return _model_manager.get_current_model(client)


def get_models_by_type() -> Dict[str, List[str]]:
    """Categorize models by type"""
    return _model_manager.get_models_by_type()


def pull_model(model_name: str) -> bool:
    """Pull/download a model from Ollama registry"""
    return _model_manager.pull_model(model_name)
