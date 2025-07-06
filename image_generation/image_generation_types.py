from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import os


class ImageGenerationProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    STABILITY_AI = "stability_ai"
    LOCAL = "local"


class ImageSize(str, Enum):
    SQUARE_256 = "256x256"
    SQUARE_512 = "512x512" 
    SQUARE_1024 = "1024x1024"
    PORTRAIT_512_768 = "512x768"
    PORTRAIT_768_1024 = "768x1024"
    LANDSCAPE_768_512 = "768x512"
    LANDSCAPE_1024_768 = "1024x768"
    WIDE_1792_1024 = "1792x1024"
    TALL_1024_1792 = "1024x1792"


class ImageQuality(str, Enum):
    STANDARD = "standard"
    HD = "hd"
    ULTRA = "ultra"


class ImageStyle(str, Enum):
    NATURAL = "natural"
    VIVID = "vivid"
    ARTISTIC = "artistic"
    PHOTOGRAPHIC = "photographic"
    DIGITAL_ART = "digital_art"
    CARTOON = "cartoon"
    SKETCH = "sketch"
    PAINTING = "painting"


class ImageGenerationConfig(BaseModel):
    provider: ImageGenerationProvider
    model: Optional[str] = None
    size: ImageSize = ImageSize.SQUARE_1024
    quality: ImageQuality = ImageQuality.STANDARD
    style: ImageStyle = ImageStyle.NATURAL
    output_dir: str = Field(default="./generated_images")
    num_images: int = Field(default=1, ge=1, le=10)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    size: Optional[ImageSize] = None
    quality: Optional[ImageQuality] = None
    style: Optional[ImageStyle] = None
    num_images: Optional[int] = Field(None, ge=1, le=10)
    seed: Optional[int] = None
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0)
    steps: Optional[int] = Field(None, ge=1, le=150)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ImageGenerationResponse(BaseModel):
    success: bool
    images: List[str] = Field(default_factory=list)  # File paths to generated images
    prompt: str
    revised_prompt: Optional[str] = None  # Some providers modify the prompt
    error_message: Optional[str] = None
    provider: str
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageGenerationClient(ABC):
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        
    @abstractmethod
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    def _ensure_output_dir(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)