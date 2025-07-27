import os
import base64
import requests
import uuid
from datetime import datetime
from typing import Optional

from .image_generation_types import (
    ImageGenerationClient,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationProvider,
)

# Conditional import for credentials
try:
    from credentials import CredentialManager, default_credential_manager
except ImportError:
    try:
        from credentials import CredentialManager, default_credential_manager
    except ImportError:
        CredentialManager = None
        default_credential_manager = None


class OpenAIImageGenerationClient(ImageGenerationClient):
    def __init__(
        self,
        config: ImageGenerationConfig,
        credential_manager: Optional[CredentialManager] = None,
    ):
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential(
            "OPENAI_API_KEY", "OpenAI Image Generation"
        )
        self.base_url = "https://api.openai.com/v1/images/generations"
        self.model = config.model or os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        try:
            self._ensure_output_dir()

            # Prepare API request
            payload = {
                "model": self.model,
                "prompt": request.prompt,
                "n": request.num_images or self.config.num_images,
                "size": (request.size or self.config.size).value,
                "quality": (request.quality or self.config.quality).value,
                "response_format": "b64_json",
            }

            if self.model == "dall-e-3":
                payload["style"] = (request.style or self.config.style).value

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()

            data = response.json()
            images = []
            revised_prompt = None

            for i, img_data in enumerate(data.get("data", [])):
                # Save image
                image_data = base64.b64decode(img_data["b64_json"])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"openai_image_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
                filepath = os.path.join(self.config.output_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_data)

                images.append(filepath)

                # Get revised prompt if available
                if not revised_prompt and "revised_prompt" in img_data:
                    revised_prompt = img_data["revised_prompt"]

            return ImageGenerationResponse(
                success=True,
                images=images,
                prompt=request.prompt,
                revised_prompt=revised_prompt,
                provider=ImageGenerationProvider.OPENAI.value,
                model=self.model,
                metadata={"api_response": data},
            )

        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                images=[],
                prompt=request.prompt,
                error_message=f"OpenAI image generation failed: {str(e)}",
                provider=ImageGenerationProvider.OPENAI.value,
                model=self.model,
            )

    def is_available(self) -> bool:
        try:
            return bool(self.credential_manager.get_credential("OPENAI_API_KEY"))
        except Exception:
            return False


class GoogleImageGenerationClient(ImageGenerationClient):
    def __init__(
        self,
        config: ImageGenerationConfig,
        credential_manager: Optional[CredentialManager] = None,
    ):
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential(
            "GOOGLE_AI_STUDIO_KEY", "Google Image Generation"
        )
        self.model = config.model or "imagegeneration@006"
        self.project_id = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
        self.location = "us-central1"

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        try:

            self._ensure_output_dir()

            # This is a simplified implementation - you'd need proper Google Cloud setup
            return ImageGenerationResponse(
                success=False,
                images=[],
                prompt=request.prompt,
                error_message="Google Image Generation requires Google Cloud setup with Vertex AI",
                provider=ImageGenerationProvider.GOOGLE.value,
                model=self.model,
            )

        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                images=[],
                prompt=request.prompt,
                error_message=f"Google image generation failed: {str(e)}",
                provider=ImageGenerationProvider.GOOGLE.value,
                model=self.model,
            )

    def is_available(self) -> bool:
        try:
            return bool(self.credential_manager.get_credential("GOOGLE_AI_STUDIO_KEY"))
        except Exception:
            return False


class StabilityAIImageGenerationClient(ImageGenerationClient):
    def __init__(
        self,
        config: ImageGenerationConfig,
        credential_manager: Optional[CredentialManager] = None,
    ):
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential(
            "STABILITY_API_KEY", "Stability AI Image Generation"
        )
        self.base_url = "https://api.stability.ai/v1/generation"
        self.model = config.model or os.getenv(
            "STABILITY_AI_MODEL", "stable-diffusion-xl-1024-v1-0"
        )

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        try:
            self._ensure_output_dir()

            # Parse size
            size = request.size or self.config.size
            width, height = map(int, size.value.split("x"))

            payload = {
                "text_prompts": [{"text": request.prompt}],
                "cfg_scale": request.guidance_scale or 7,
                "height": height,
                "width": width,
                "samples": request.num_images or self.config.num_images,
                "steps": request.steps or 30,
            }

            if request.negative_prompt:
                payload["text_prompts"].append(
                    {"text": request.negative_prompt, "weight": -1}
                )

            if request.seed:
                payload["seed"] = request.seed

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            response = requests.post(
                f"{self.base_url}/{self.model}/text-to-image",
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()

            data = response.json()
            images = []

            for i, artifact in enumerate(data.get("artifacts", [])):
                if artifact.get("finishReason") == "SUCCESS":
                    # Save image
                    image_data = base64.b64decode(artifact["base64"])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = (
                        f"stability_image_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
                    )
                    filepath = os.path.join(self.config.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    images.append(filepath)

            return ImageGenerationResponse(
                success=True,
                images=images,
                prompt=request.prompt,
                provider=ImageGenerationProvider.STABILITY_AI.value,
                model=self.model,
                metadata={"api_response": data},
            )

        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                images=[],
                prompt=request.prompt,
                error_message=f"Stability AI image generation failed: {str(e)}",
                provider=ImageGenerationProvider.STABILITY_AI.value,
                model=self.model,
            )

    def is_available(self) -> bool:
        try:
            return bool(self.credential_manager.get_credential("STABILITY_API_KEY"))
        except Exception:
            return False


class LocalImageGenerationClient(ImageGenerationClient):
    def __init__(
        self,
        config: ImageGenerationConfig,
        credential_manager: Optional[CredentialManager] = None,
    ):
        super().__init__(config)
        self.base_url = os.getenv("LOCAL_IMAGE_GEN_URL", "http://192.168.0.96:8000")
        self.model = config.model or os.getenv(
            "LOCAL_IMAGE_GEN_DEFAULT_MODEL", "stable-diffusion"
        )

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        try:
            self._ensure_output_dir()

            # Parse size
            size = request.size or self.config.size
            width, height = map(int, size.value.split("x"))

            payload = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "width": width,
                "height": height,
                "num_images": request.num_images or self.config.num_images,
                "guidance_scale": request.guidance_scale or 7.5,
                "num_inference_steps": request.steps or 20,
                "model": self.model,
            }

            if request.seed:
                payload["seed"] = request.seed

            # Add any extra parameters
            payload.update(request.extra_params)

            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=300,  # Local generation can take longer
            )
            response.raise_for_status()

            data = response.json()
            images = []

            # Handle different response formats
            if "images" in data:
                for i, img_b64 in enumerate(data["images"]):
                    # Save image
                    image_data = base64.b64decode(img_b64)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"local_image_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
                    filepath = os.path.join(self.config.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    images.append(filepath)

            return ImageGenerationResponse(
                success=True,
                images=images,
                prompt=request.prompt,
                provider=ImageGenerationProvider.LOCAL.value,
                model=self.model,
                metadata={"api_response": data},
            )

        except Exception as e:
            return ImageGenerationResponse(
                success=False,
                images=[],
                prompt=request.prompt,
                error_message=f"Local image generation failed: {str(e)}",
                provider=ImageGenerationProvider.LOCAL.value,
                model=self.model,
            )

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
