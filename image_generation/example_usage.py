#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_generation import (
    create_image_generation_service,
    create_local_image_service,
    create_openai_image_service,
    quick_image_generation,
    get_available_providers,
    ImageSize,
    ImageQuality,
    ImageStyle
)


def main():
    print("=== Image Generation Service Examples ===\n")
    
    # Check available providers
    print("Available providers:")
    providers = get_available_providers()
    for provider, available in providers.items():
        status = "✓" if available else "✗"
        print(f"  {status} {provider}")
    print()
    
    # Example 1: Quick image generation with auto-detection
    print("1. Quick image generation (auto-detect provider):")
    try:
        image_path = quick_image_generation(
            "A serene mountain landscape at sunset with a crystal clear lake",
            size=ImageSize.SQUARE_1024,
            quality=ImageQuality.STANDARD
        )
        print(f"   Generated image saved to: {image_path}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 2: Using specific provider with custom configuration
    print("2. Using auto-detected service with custom settings:")
    try:
        service = create_image_generation_service(
            provider="auto",
            output_dir="./my_images",
            size=ImageSize.SQUARE_512,
            quality=ImageQuality.HD,
            style=ImageStyle.VIVID
        )
        
        print(f"   Provider info: {service.get_provider_info()}")
        
        response = service.generate_image(
            prompt="A futuristic cityscape with flying cars and neon lights",
            negative_prompt="blurry, low quality, watermark",
            num_images=2
        )
        
        if response.success:
            print(f"   Generated {len(response.images)} images:")
            for img_path in response.images:
                print(f"     - {img_path}")
            if response.revised_prompt:
                print(f"   Revised prompt: {response.revised_prompt}")
        else:
            print(f"   Error: {response.error_message}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 3: Local service (if available)
    print("3. Using local image generation service:")
    try:
        local_service = create_local_image_service(
            output_dir="./local_images",
            model="stable-diffusion"
        )
        
        if local_service.is_available():
            response = local_service.generate_image(
                prompt="A cute robot playing with a cat in a garden",
                size=ImageSize.SQUARE_512,
                guidance_scale=7.5,
                steps=30,
                seed=12345
            )
            
            if response.success:
                print(f"   Local generation successful: {response.images[0]}")
            else:
                print(f"   Local generation failed: {response.error_message}")
        else:
            print("   Local service not available")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 4: Batch generation
    print("4. Batch image generation:")
    try:
        service = create_image_generation_service("auto")
        
        prompts = [
            "A red apple on a wooden table",
            "A blue ocean with white waves",
            "A green forest in the morning mist"
        ]
        
        responses = service.batch_generate(
            prompts,
            size=ImageSize.SQUARE_512,
            quality=ImageQuality.STANDARD
        )
        
        print(f"   Generated {len(responses)} images from batch:")
        for i, response in enumerate(responses):
            if response.success:
                print(f"     {i+1}. {response.images[0]} (prompt: {response.prompt[:50]}...)")
            else:
                print(f"     {i+1}. Failed: {response.error_message}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 5: Variations
    print("5. Generate variations of a base prompt:")
    try:
        service = create_image_generation_service("auto")
        
        base_prompt = "A majestic dragon"
        variations = [
            "in a medieval castle",
            "flying over a modern city",
            "sleeping in a crystal cave",
            "breathing colorful flames"
        ]
        
        responses = service.generate_variations(
            base_prompt,
            variations,
            size=ImageSize.SQUARE_512
        )
        
        print(f"   Generated {len(responses)} variations:")
        for i, response in enumerate(responses):
            if response.success:
                print(f"     {i+1}. {response.images[0]}")
                print(f"        Prompt: {response.prompt}")
            else:
                print(f"     {i+1}. Failed: {response.error_message}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    print("=== Examples completed ===")


if __name__ == "__main__":
    main()