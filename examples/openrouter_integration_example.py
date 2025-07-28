#!/usr/bin/env python3
"""
OpenRouter Integration Example for AI Lego Bricks

This example demonstrates how to use OpenRouter as an LLM provider to access
100+ models through a single API key with unified billing.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable or in .env file
- Optional: Set OPENROUTER_DEFAULT_MODEL (defaults to anthropic/claude-3.5-sonnet)
- Optional: Set OPENROUTER_APP_NAME (for HTTP-Referer header)

OpenRouter provides access to models from:
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- OpenAI (GPT-4, GPT-3.5-turbo, etc.)
- Meta (Llama 3.1, Llama 2, etc.)
- Google (Gemini Pro, PaLM, etc.)
- Mistral, Cohere, and many more
"""

import os
import sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import AI Lego Bricks components
try:
    from llm.llm_factory import (
        LLMClientFactory,
        create_openrouter_generation,
        create_openrouter_conversation,
    )
    from llm.llm_types import LLMProvider
    from credentials import CredentialManager
except ImportError as e:
    print("❌ Error: AI Lego Bricks components not available")
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    exit(1)


def demo_basic_openrouter_usage():
    """Demonstrate basic OpenRouter usage patterns"""
    print("🔄 Demo: Basic OpenRouter Usage")
    print("=" * 50)

    # Method 1: Using factory with explicit provider
    print("\n1. Using LLMClientFactory:")
    try:
        client = LLMClientFactory.create_text_client(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-3.5-sonnet",
            temperature=0.7,
        )
        print(f"✅ Created OpenRouter client with model: {client.get_current_model()}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    # Method 2: Using convenience functions
    print("\n2. Using convenience functions:")
    try:
        gen_service = create_openrouter_generation(
            model="meta-llama/llama-3.1-8b-instruct", temperature=0.8
        )
        print(
            f"✅ Created generation service with model: {gen_service.get_current_model()}"
        )

        conv_service = create_openrouter_conversation(
            model="openai/gpt-4", temperature=0.6
        )
        print(
            f"✅ Created conversation service with model: {conv_service.get_current_model()}"
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        return


def demo_model_switching():
    """Demonstrate OpenRouter's key benefit: easy model switching"""
    print("\n🔄 Demo: Model Switching")
    print("=" * 50)

    try:
        # Create client with initial model
        client = LLMClientFactory.create_text_client(
            provider=LLMProvider.OPENROUTER,
            model="meta-llama/llama-3.1-8b-instruct",  # Start with fast model
        )

        print(f"Initial model: {client.get_current_model()}")

        # Switch to different models for different tasks
        models_to_test = [
            "anthropic/claude-3.5-sonnet",  # Smart model for analysis
            "openai/gpt-4",  # Good for general tasks
            "google/gemini-1.5-pro",  # Good for long context
            "meta-llama/llama-3.1-70b-instruct",  # Powerful open source
        ]

        for model in models_to_test:
            success = client.switch_model(model)
            if success:
                print(f"✅ Switched to: {client.get_current_model()}")
            else:
                print(f"❌ Failed to switch to: {model}")

    except ValueError as e:
        print(f"❌ Error: {e}")


def demo_conversation_with_model_switching():
    """Demonstrate conversation with strategic model switching"""
    print("\n🔄 Demo: Conversation with Strategic Model Switching")
    print("=" * 50)

    try:
        # Create conversation service
        conv = create_openrouter_conversation(
            model="meta-llama/llama-3.1-8b-instruct",  # Start with fast model
            conversation_id="openrouter_demo",
        )

        conv.add_system_message("You are a helpful AI assistant.")

        # Start with fast model for simple greeting
        print(f"\n🤖 Using {conv.get_current_model()} for greeting:")
        response1 = conv.send_message(
            "Hello! I need help with a complex data analysis."
        )
        print(f"Assistant: {response1}")

        # Switch to smart model for complex analysis
        conv.switch_model("anthropic/claude-3.5-sonnet")
        print(f"\n🧠 Switched to {conv.get_current_model()} for analysis:")
        response2 = conv.send_message(
            "Can you explain the differences between supervised and unsupervised learning?"
        )
        print(f"Assistant: {response2}")

        # Show conversation history
        print("\n📊 Conversation Stats:")
        stats = conv.get_conversation_stats()
        print(f"Total messages: {stats['total_messages']}")
        print(f"User messages: {stats['user_messages']}")
        print(f"Assistant messages: {stats['assistant_messages']}")

    except ValueError as e:
        print(f"❌ Error: {e}")


def demo_generation_vs_conversation():
    """Demonstrate when to use Generation vs Conversation services"""
    print("\n🔄 Demo: Generation vs Conversation Services")
    print("=" * 50)

    try:
        # Generation Service: For one-shot tasks
        print("\n⚡ Generation Service (Stateless, optimized for speed):")
        gen_service = create_openrouter_generation(
            model="meta-llama/llama-3.1-8b-instruct"
        )

        # Good for: analysis, classification, transformation
        analysis = gen_service.generate(
            "Classify this text as positive, negative, or neutral: 'This product works great!'"
        )
        print(f"Analysis result: {analysis}")

        # Conversation Service: For multi-turn interactions
        print("\n💬 Conversation Service (Stateful, rich context management):")
        conv_service = create_openrouter_conversation(
            model="anthropic/claude-3.5-sonnet"
        )

        conv_service.add_system_message("You are a helpful coding tutor.")

        # Good for: interactive assistance, context-dependent tasks
        response1 = conv_service.send_message("What is Python?")
        print(f"Response 1: {response1}")

        response2 = conv_service.send_message("Can you show me how to create a list?")
        print(f"Response 2: {response2}")

        # Rich conversation access
        print(f"\nFirst question: {conv_service.get_first_prompt()}")
        print(f"Total messages: {conv_service.get_conversation_length()}")

    except ValueError as e:
        print(f"❌ Error: {e}")


def demo_credential_management():
    """Demonstrate credential management patterns"""
    print("\n🔄 Demo: Credential Management")
    print("=" * 50)

    # Method 1: Environment variables (default)
    print("\n1. Using environment variables:")
    try:
        create_openrouter_generation()
        print("✅ Client created using environment credentials")
    except ValueError:
        print("❌ No OPENROUTER_API_KEY in environment")

    # Method 2: Explicit credentials (library-safe)
    print("\n2. Using explicit credentials:")
    try:
        explicit_creds = CredentialManager(
            {
                "OPENROUTER_API_KEY": "explicit-api-key-here",
                "OPENROUTER_DEFAULT_MODEL": "anthropic/claude-3.5-sonnet",
                "OPENROUTER_APP_NAME": "My Custom App",
            },
            load_env=False,  # Don't load .env file
        )

        LLMClientFactory.create_text_client(
            provider=LLMProvider.OPENROUTER, credential_manager=explicit_creds
        )
        print("✅ Client created with explicit credentials")
    except ValueError as e:
        print(f"Expected error with dummy credentials: {e}")

    # Method 3: Mixed approach
    print("\n3. Using mixed approach (some explicit, some from env):")
    try:
        mixed_creds = CredentialManager(
            {"OPENROUTER_APP_NAME": "Override App Name"},  # Override specific settings
            load_env=True,  # Still load other credentials from environment
        )

        LLMClientFactory.create_text_client(
            provider=LLMProvider.OPENROUTER, credential_manager=mixed_creds
        )
        print("✅ Client created with mixed credentials")
    except ValueError as e:
        print(f"❌ Error: {e}")


def demo_agent_orchestration():
    """Demonstrate OpenRouter in agent workflows"""
    print("\n🔄 Demo: Agent Orchestration Integration")
    print("=" * 50)

    workflow_example = {
        "name": "openrouter_workflow",
        "description": "Example workflow using OpenRouter models",
        "steps": [
            {
                "id": "fast_analysis",
                "type": "llm_chat",
                "config": {
                    "provider": "openrouter",
                    "model": "meta-llama/llama-3.1-8b-instruct",  # Fast model
                    "use_conversation": False,  # Generation service
                    "system_message": "You are a content analyzer.",
                },
                "inputs": {"message": "Analyze the sentiment of: 'AI is amazing!'"},
            },
            {
                "id": "detailed_response",
                "type": "llm_chat",
                "config": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-3.5-sonnet",  # Smart model
                    "use_conversation": True,  # Conversation service
                    "conversation_id": "detailed_analysis",
                    "system_message": "You are a detailed AI assistant.",
                },
                "inputs": {"message": "Explain the analysis in detail."},
            },
            {
                "id": "model_switch_in_conversation",
                "type": "llm_chat",
                "config": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4",  # Different model, same conversation
                    "use_conversation": True,
                    "conversation_id": "detailed_analysis",  # Continue same conversation
                },
                "inputs": {"message": "Can you summarize our discussion?"},
            },
        ],
    }

    print("✅ Sample workflow definition:")
    print(f"  - Steps: {len(workflow_example['steps'])}")
    print("  - Models used: llama-3.1-8b, claude-3.5-sonnet, gpt-4")
    print("  - Services: Generation and Conversation")
    print("  - Features: Model switching within conversation")

    print("\n📝 Workflow JSON structure created successfully!")
    print("   This workflow demonstrates:")
    print("   • Using fast models for simple analysis")
    print("   • Switching to smart models for detailed work")
    print("   • Maintaining conversation context across model switches")


def main():
    """Run all OpenRouter integration examples"""
    print("🚀 AI Lego Bricks - OpenRouter Integration Examples")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  Warning: OPENROUTER_API_KEY not found in environment")
        print("   Set your OpenRouter API key to run actual API calls:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        print("\n   For now, running configuration and structure demos...\n")

    # Run demos
    demo_basic_openrouter_usage()
    demo_model_switching()
    demo_conversation_with_model_switching()
    demo_generation_vs_conversation()
    demo_credential_management()
    demo_agent_orchestration()

    print("\n🎉 OpenRouter Integration Examples Complete!")
    print("\nKey Benefits:")
    print("• 100+ models accessible through single API key")
    print("• Unified billing across all providers")
    print("• Easy model switching within conversations")
    print("• Full integration with Generation and Conversation services")
    print("• Works seamlessly with agent orchestration workflows")
    print("\nFor more models and pricing: https://openrouter.ai/models")


if __name__ == "__main__":
    main()
