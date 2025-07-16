# Image Generation

A modular image generation service that supports multiple providers with automatic provider detection and unified configuration.

## Supported Providers

- **OpenAI DALL-E** - Requires `OPENAI_API_KEY`
- **Google Imagen** - Requires `GOOGLE_AI_STUDIO_KEY`
- **Stability AI** - Requires `STABILITY_API_KEY`
- **Local Service** - Requires local server at `LOCAL_IMAGE_GEN_URL`

## Quick Start

### Simple Image Generation

```python
from image_generation import quick_image_generation, ImageSize, ImageQuality

# Generate with auto-detected provider
image_path = quick_image_generation(
    "A serene mountain landscape at sunset",
    size=ImageSize.SQUARE_1024,
    quality=ImageQuality.HD
)
print(f"Image saved to: {image_path}")
```

### Service-Based Generation

```python
from image_generation import create_image_generation_service, ImageStyle

# Create service with auto-detection
service = create_image_generation_service(
    provider="auto",
    output_dir="./my_images",
    size=ImageSize.SQUARE_512,
    style=ImageStyle.VIVID
)

# Generate single image
response = service.generate_image(
    prompt="A futuristic cityscape with flying cars",
    negative_prompt="blurry, low quality",
    num_images=1
)

if response.success:
    print(f"Generated: {response.images[0]}")
```

### Specific Provider

```python
from image_generation import create_openai_image_service

# Use specific provider
service = create_openai_image_service(
    model="dall-e-3",
    output_dir="./openai_images"
)

response = service.generate_image("A cute robot playing with a cat")
```

## Batch Operations

### Multiple Prompts

```python
service = create_image_generation_service("auto")

responses = service.batch_generate([
    "A red apple on wooden table",
    "A blue ocean with white waves",
    "A green forest in morning mist"
])

for i, response in enumerate(responses):
    if response.success:
        print(f"Image {i+1}: {response.images[0]}")
```

### Prompt Variations

```python
responses = service.generate_variations(
    base_prompt="A majestic dragon",
    variations=[
        "in a medieval castle",
        "flying over a modern city",
        "sleeping in a crystal cave"
    ]
)
```

## Configuration Options

### Image Sizes
- `SQUARE_256`, `SQUARE_512`, `SQUARE_1024`
- `PORTRAIT_512_768`, `PORTRAIT_768_1024`
- `LANDSCAPE_768_512`, `LANDSCAPE_1024_768`
- `WIDE_1792_1024`, `TALL_1024_1792`

### Quality Levels
- `STANDARD` - Basic quality
- `HD` - High definition
- `ULTRA` - Ultra high quality (provider dependent)

### Styles
- `NATURAL` - Natural photography style
- `VIVID` - Enhanced colors and contrast
- `ARTISTIC` - Artistic interpretation
- `PHOTOGRAPHIC` - Photorealistic
- `DIGITAL_ART` - Digital art style
- `CARTOON`, `SKETCH`, `PAINTING` - Artistic styles

## Advanced Configuration

```python
from image_generation import ImageGenerationConfig, ImageGenerationProvider

# Custom configuration
config = ImageGenerationConfig(
    provider=ImageGenerationProvider.OPENAI,
    model="dall-e-3",
    size=ImageSize.SQUARE_1024,
    quality=ImageQuality.HD,
    style=ImageStyle.VIVID,
    output_dir="./custom_images",
    num_images=1
)

service = create_image_generation_service(config=config)
```

## Local Service Setup

For local image generation, ensure your local service is running:

```python
# Check if local service is available
from image_generation import get_available_providers

providers = get_available_providers()
if providers["local"]:
    service = create_local_image_service(
        output_dir="./local_images",
        model="stable-diffusion"
    )
    
    response = service.generate_image(
        prompt="A cute robot in a garden",
        guidance_scale=7.5,
        steps=30,
        seed=12345
    )
```

## Error Handling

```python
response = service.generate_image("A beautiful sunset")

if response.success:
    print(f"Generated: {response.images[0]}")
    if response.revised_prompt:
        print(f"Revised prompt: {response.revised_prompt}")
else:
    print(f"Generation failed: {response.error_message}")
```

## Provider Detection

The service automatically detects available providers based on credentials:

```python
from image_generation import get_available_providers

providers = get_available_providers()
for provider, available in providers.items():
    status = "✓" if available else "✗"
    print(f"{status} {provider}")
```

## Requirements

- Set appropriate API keys as environment variables
- For local service: ensure server is running at configured URL
- PIL (Pillow) for image processing
- requests for HTTP communication