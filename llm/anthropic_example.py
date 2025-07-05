#!/usr/bin/env python3
"""
Example usage of Anthropic Claude models with the LLM factory.

Before running this example, make sure you have:
1. Installed the anthropic package: pip install anthropic
2. Set your ANTHROPIC_API_KEY environment variable
3. Optionally set ANTHROPIC_DEFAULT_MODEL environment variable
"""

import os
from llm_factory import LLMClientFactory
from llm_types import LLMProvider

def main():
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it with: export ANTHROPIC_API_KEY=your_api_key_here")
        return
    
    print("Creating Anthropic text client...")
    
    # Create Anthropic client with default model
    client = LLMClientFactory.create_text_client(
        provider=LLMProvider.ANTHROPIC,
        temperature=0.7,
        max_tokens=1000
    )
    
    print(f"Using model: {client.get_current_model()}")
    
    # Simple chat example
    print("\n--- Simple Chat Example ---")
    response = client.chat("Hello! What can you help me with today?")
    print(f"Claude: {response}")
    
    # Chat with history example
    print("\n--- Chat with History Example ---")
    from llm_types import ChatMessage
    
    history = [
        ChatMessage(role="user", content="I'm working on a Python project."),
        ChatMessage(role="assistant", content="That's great! I'd be happy to help you with your Python project. What specific aspects are you working on or need assistance with?"),
        ChatMessage(role="user", content="Can you help me understand decorators?")
    ]
    
    response = client.chat_with_messages(history)
    print(f"Claude: {response}")
    
    # Model switching example
    print("\n--- Model Switching Example ---")
    print(f"Current model: {client.get_current_model()}")
    
    # Switch to Claude 3 Haiku (faster, cheaper)
    success = client.switch_model("claude-3-haiku-20240307")
    if success:
        print(f"Switched to: {client.get_current_model()}")
        quick_response = client.chat("Give me a brief summary of Python in one sentence.")
        print(f"Claude Haiku: {quick_response}")
    
    # Switch back to Sonnet (more capable)
    client.switch_model("claude-3-5-sonnet-20241022")
    print(f"Switched back to: {client.get_current_model()}")

if __name__ == "__main__":
    main()