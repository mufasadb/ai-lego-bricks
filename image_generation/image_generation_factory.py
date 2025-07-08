from typing import Union, Optional, Dict
import os

from .image_generation_types import (
    ImageGenerationProvider,
    ImageGenerationConfig,
    ImageSize,
    ImageQuality,
    ImageStyle
)
from .image_generation_clients import (
    OpenAIImageGenerationClient,
    GoogleImageGenerationClient,
    StabilityAIImageGenerationClient,
    LocalImageGenerationClient
)
from .image_generation_service import ImageGenerationService
from credentials import CredentialManager, default_credential_manager


class ImageGenerationServiceFactory:
    @staticmethod
    def create_image_generation_service(
        provider: Union[str, ImageGenerationProvider] = "auto",
        credential_manager: Optional[CredentialManager] = None,
        **kwargs
    ) -> ImageGenerationService:
        if provider == "auto":
            provider = ImageGenerationServiceFactory._detect_available_provider()
        
        if isinstance(provider, str):
            provider = ImageGenerationProvider(provider)
        
        config = ImageGenerationServiceFactory._create_config(provider, **kwargs)
        
        client_map = {
            ImageGenerationProvider.OPENAI: OpenAIImageGenerationClient,
            ImageGenerationProvider.GOOGLE: GoogleImageGenerationClient,
            ImageGenerationProvider.STABILITY_AI: StabilityAIImageGenerationClient,
            ImageGenerationProvider.LOCAL: LocalImageGenerationClient
        }
        
        if provider not in client_map:
            raise ValueError(f"Unsupported image generation provider: {provider}")
        
        client_class = client_map[provider]
        client = client_class(config, credential_manager)
        
        return ImageGenerationService(client)
    
    @staticmethod
    def _detect_available_provider() -> ImageGenerationProvider:
        credential_manager = default_credential_manager
        
        # Check local service first
        try:
            import requests
            local_url = os.getenv("LOCAL_IMAGE_GEN_URL", "http://192.168.0.96:8000")
            response = requests.get(f"{local_url}/health", timeout=5)
            if response.status_code == 200:
                return ImageGenerationProvider.LOCAL
        except Exception:
            pass
        
        # Check API-based providers
        if credential_manager.get_credential("OPENAI_API_KEY"):
            return ImageGenerationProvider.OPENAI
        
        if credential_manager.get_credential("STABILITY_API_KEY"):
            return ImageGenerationProvider.STABILITY_AI
        
        if credential_manager.get_credential("GOOGLE_AI_STUDIO_KEY"):
            return ImageGenerationProvider.GOOGLE
        
        raise RuntimeError(
            "No image generation provider available. Please set up one of the following:\n"
            "- OPENAI_API_KEY for OpenAI DALL-E\n"
            "- STABILITY_API_KEY for Stability AI\n"
            "- GOOGLE_AI_STUDIO_KEY for Google Imagen\n"
            "- Local service at LOCAL_IMAGE_GEN_URL"
        )
    
    @staticmethod
    def _create_config(provider: ImageGenerationProvider, **kwargs) -> ImageGenerationConfig:
        return ImageGenerationConfig(
            provider=provider,
            model=kwargs.get("model"),
            size=kwargs.get("size", ImageSize.SQUARE_1024),
            quality=kwargs.get("quality", ImageQuality.STANDARD),
            style=kwargs.get("style", ImageStyle.NATURAL),
            output_dir=kwargs.get("output_dir", "./generated_images"),
            num_images=kwargs.get("num_images", 1),
            extra_params=kwargs.get("extra_params", {})
        )
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        credential_manager = default_credential_manager
        providers = {}
        
        # Check local service
        try:
            import requests
            local_url = os.getenv("LOCAL_IMAGE_GEN_URL", "http://192.168.0.96:8000")
            response = requests.get(f"{local_url}/health", timeout=5)
            providers["local"] = response.status_code == 200
        except Exception:
            providers["local"] = False
        
        # Check API providers
        providers["openai"] = bool(credential_manager.get_credential("OPENAI_API_KEY"))
        providers["stability_ai"] = bool(credential_manager.get_credential("STABILITY_API_KEY"))
        providers["google"] = bool(credential_manager.get_credential("GOOGLE_AI_STUDIO_KEY"))
        
        return providers


# Convenience functions
def create_image_generation_service(provider: str = "auto", **kwargs) -> ImageGenerationService:
    return ImageGenerationServiceFactory.create_image_generation_service(provider, **kwargs)


def create_openai_image_service(**kwargs) -> ImageGenerationService:
    return create_image_generation_service("openai", **kwargs)


def create_stability_image_service(**kwargs) -> ImageGenerationService:
    return create_image_generation_service("stability_ai", **kwargs)


def create_google_image_service(**kwargs) -> ImageGenerationService:
    return create_image_generation_service("google", **kwargs)


def create_local_image_service(**kwargs) -> ImageGenerationService:
    return create_image_generation_service("local", **kwargs)


def quick_image_generation(
    prompt: str,
    provider: str = "auto",
    **kwargs
) -> str:
    service = create_image_generation_service(provider)
    response = service.generate_image(prompt, **kwargs)
    
    if response.success and response.images:
        return response.images[0]
    else:
        raise RuntimeError(f"Image generation failed: {response.error_message}")


def get_available_providers() -> Dict[str, bool]:
    return ImageGenerationServiceFactory.get_available_providers()