"""
Example usage of model switching functionality in the LLM service.

This example demonstrates how to:
1. List available models
2. Create switchable clients
3. Switch models on existing clients
4. Use different models for different tasks

Prerequisites:
- Ollama running locally
- At least one text model (e.g., llama2, codellama)
- At least one vision model (e.g., llava)
"""

import os
from llm.llm_factory import (
    create_switchable_text_client,
    create_switchable_vision_client,
    switch_client_model,
    get_client_model
)
from llm.model_manager import (
    list_available_models,
    get_model_names,
    get_models_by_type,
    validate_model,
    pull_model
)
from llm.llm_types import LLMProvider, VisionProvider


def demonstrate_model_discovery():
    """Demonstrate model discovery and categorization"""
    print("=== Model Discovery ===")
    
    # List all available models
    print("\n1. All available models:")
    try:
        models = list_available_models()
        for model in models:
            print(f"  - {model['name']} (size: {model.get('size', 'unknown')})")
    except Exception as e:
        print(f"  Error: {e}")
        return
    
    # Get just model names
    print("\n2. Model names only:")
    model_names = get_model_names()
    for name in model_names:
        print(f"  - {name}")
    
    # Categorize models by type
    print("\n3. Models by type:")
    categorized = get_models_by_type()
    for category, models in categorized.items():
        if models:
            print(f"  {category.title()}: {', '.join(models)}")


def demonstrate_text_model_switching():
    """Demonstrate switching between text models"""
    print("\n=== Text Model Switching ===")
    
    try:
        # Create a switchable text client
        print("\n1. Creating switchable text client...")
        client = create_switchable_text_client("ollama", initial_model="llama2")
        print(f"   Initial model: {get_client_model(client)}")
        
        # Test chat with initial model
        print("\n2. Testing chat with initial model...")
        response = client.chat("Hello! What model are you?")
        print(f"   Response: {response[:100]}...")
        
        # Switch to a different model (if available)
        available_models = get_model_names()
        text_models = [m for m in available_models if any(indicator in m.lower() 
                      for indicator in ["llama", "mistral", "codellama", "gemma"])]
        
        if len(text_models) > 1:
            new_model = next((m for m in text_models if m != get_client_model(client)), None)
            if new_model:
                print(f"\n3. Switching to {new_model}...")
                switch_client_model(client, new_model)
                print(f"   Current model: {get_client_model(client)}")
                
                # Test chat with new model
                print("\n4. Testing chat with new model...")
                response = client.chat("Hello! What model are you now?")
                print(f"   Response: {response[:100]}...")
        else:
            print("\n3. Only one text model available, skipping switch demo")
            
    except Exception as e:
        print(f"   Error: {e}")


def demonstrate_vision_model_switching():
    """Demonstrate switching between vision models"""
    print("\n=== Vision Model Switching ===")
    
    try:
        # Check if vision models are available
        categorized = get_models_by_type()
        vision_models = categorized.get("vision", [])
        
        if not vision_models:
            print("   No vision models available. Try running: ollama pull llava")
            return
        
        # Create a switchable vision client
        print("\n1. Creating switchable vision client...")
        client = create_switchable_vision_client("llava")
        print(f"   Initial model: {get_client_model(client)}")
        
        # Note: Vision testing would require actual image data
        print("\n2. Vision client created successfully!")
        print("   (Image processing would require base64 image data)")
        
        # Switch between vision models if multiple available
        if len(vision_models) > 1:
            new_model = next((m for m in vision_models if m != get_client_model(client)), None)
            if new_model:
                print(f"\n3. Switching to {new_model}...")
                switch_client_model(client, new_model)
                print(f"   Current model: {get_client_model(client)}")
        else:
            print("\n3. Only one vision model available, skipping switch demo")
            
    except Exception as e:
        print(f"   Error: {e}")


def demonstrate_model_validation():
    """Demonstrate model validation and error handling"""
    print("\n=== Model Validation ===")
    
    # Test valid model
    print("\n1. Testing valid model validation...")
    valid_models = get_model_names()
    if valid_models:
        test_model = valid_models[0]
        is_valid = validate_model(test_model)
        print(f"   Model '{test_model}' is valid: {is_valid}")
    
    # Test invalid model
    print("\n2. Testing invalid model validation...")
    invalid_model = "definitely-not-a-real-model"
    is_valid = validate_model(invalid_model)
    print(f"   Model '{invalid_model}' is valid: {is_valid}")
    
    # Test model switching with invalid model
    print("\n3. Testing error handling for invalid model switch...")
    try:
        client = create_switchable_text_client("ollama")
        switch_client_model(client, "invalid-model")
    except ValueError as e:
        print(f"   Caught expected error: {e}")


def demonstrate_model_pulling():
    """Demonstrate pulling new models"""
    print("\n=== Model Pulling ===")
    
    print("\n1. Note: Model pulling can take a long time and requires internet connection")
    print("   Example of how to pull a model:")
    print("   success = pull_model('llama2')")
    print("   if success:")
    print("       print('Model pulled successfully')")
    print("   else:")
    print("       print('Failed to pull model')")
    
    # Don't actually pull models in the example as it takes too long
    print("\n   (Skipping actual model pull in demo)")


def main():
    """Main function to run all demonstrations"""
    print("Model Switching Example")
    print("=" * 50)
    
    # Check if Ollama is running
    try:
        get_model_names()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running and accessible at http://localhost:11434")
        return
    
    # Run demonstrations
    demonstrate_model_discovery()
    demonstrate_text_model_switching()
    demonstrate_vision_model_switching()
    demonstrate_model_validation()
    demonstrate_model_pulling()
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()