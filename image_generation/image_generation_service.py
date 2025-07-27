from typing import Optional, Dict, Any, List

from .image_generation_types import (
    ImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageSize,
    ImageQuality,
    ImageStyle,
)


class ImageGenerationService:
    def __init__(self, client: ImageGenerationClient):
        self.client = client
        self.config = client.config

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        size: Optional[ImageSize] = None,
        quality: Optional[ImageQuality] = None,
        style: Optional[ImageStyle] = None,
        num_images: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        **kwargs,
    ) -> ImageGenerationResponse:
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            size=size,
            quality=quality,
            style=style,
            num_images=num_images,
            seed=seed,
            guidance_scale=guidance_scale,
            steps=steps,
            extra_params=kwargs,
        )

        return self.client.generate_image(request)

    def generate_image_from_request(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        return self.client.generate_image(request)

    def is_available(self) -> bool:
        return self.client.is_available()

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider.value,
            "model": getattr(self.client, "model", None),
            "output_dir": self.config.output_dir,
            "default_size": self.config.size.value,
            "default_quality": self.config.quality.value,
            "default_style": self.config.style.value,
            "available": self.is_available(),
        }

    def generate_variations(
        self, base_prompt: str, variations: List[str], **kwargs
    ) -> List[ImageGenerationResponse]:
        responses = []

        for variation in variations:
            full_prompt = f"{base_prompt}, {variation}"
            response = self.generate_image(prompt=full_prompt, **kwargs)
            responses.append(response)

        return responses

    def batch_generate(
        self, prompts: List[str], **kwargs
    ) -> List[ImageGenerationResponse]:
        responses = []

        for prompt in prompts:
            response = self.generate_image(prompt=prompt, **kwargs)
            responses.append(response)

        return responses
