"""
Quick example of model switching - common use cases.

This demonstrates the most common model switching scenarios:
1. Switch between text models for different tasks
2. Switch to vision models when needed
3. List available models
"""

from llm.llm_factory import create_switchable_text_client, create_switchable_vision_client
from llm.model_manager import get_model_names, get_models_by_type


def quick_text_switching():
    """Quick example of text model switching"""
    print("=== Quick Text Model Switching ===")
    
    try:
        # Create a text client
        client = create_switchable_text_client("ollama")
        print(f"Started with model: {client.get_current_model()}")
        
        # Get available models
        models = get_models_by_type()
        text_models = models.get("text", [])
        code_models = models.get("code", [])
        
        print(f"Available text models: {text_models}")
        print(f"Available code models: {code_models}")
        
        # Example: Switch to a code model for programming tasks
        if code_models:
            print(f"\nSwitching to code model: {code_models[0]}")
            client.switch_model(code_models[0])
            response = client.chat("Write a Python function to reverse a string")
            print(f"Code model response: {response[:100]}...")
        
        # Switch back to a general text model
        if text_models:
            print(f"\nSwitching to text model: {text_models[0]}")
            client.switch_model(text_models[0])
            response = client.chat("Tell me about artificial intelligence")
            print(f"Text model response: {response[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")


def quick_vision_switching():
    """Quick example of vision model switching"""
    print("\n=== Quick Vision Model Switching ===")
    
    try:
        # Check for vision models
        models = get_models_by_type()
        vision_models = models.get("vision", [])
        
        if not vision_models:
            print("No vision models available. Try: ollama pull llava")
            return
        
        # Create a vision client
        client = create_switchable_vision_client("llava")
        print(f"Vision model: {client.get_current_model()}")
        
        # Switch between vision models if multiple available
        if len(vision_models) > 1:
            for model in vision_models:
                print(f"Switching to: {model}")
                client.switch_model(model)
                # In real usage, you would call:
                # result = client.process_image(base64_image, "What's in this image?")
                
    except Exception as e:
        print(f"Error: {e}")


def list_all_models():
    """Simple function to list all available models"""
    print("\n=== Available Models ===")
    
    try:
        model_names = get_model_names()
        print(f"Total models: {len(model_names)}")
        for i, name in enumerate(model_names, 1):
            print(f"{i}. {name}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_all_models()
    quick_text_switching()
    quick_vision_switching()