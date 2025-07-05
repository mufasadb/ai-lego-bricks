#!/usr/bin/env python3
"""
Example demonstrating streaming capabilities for chat and generation services.
Shows how to use the new streaming APIs alongside the existing regular APIs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.generation_service import quick_generate_ollama_stream, quick_generate_gemini_stream
from chat.chat_service import quick_chat_ollama_stream, quick_chat_gemini_stream
from chat.conversation_service import create_ollama_conversation, create_gemini_conversation
from llm.llm_types import LLMProvider


def demo_streaming_generation():
    """Demonstrate streaming generation service"""
    print("=" * 60)
    print("🚀 STREAMING GENERATION SERVICE DEMO")
    print("=" * 60)
    
    prompt = "Write a short poem about artificial intelligence"
    
    print(f"Prompt: {prompt}\n")
    
    # Test Ollama streaming (if available)
    try:
        print("📡 Ollama Streaming Response:")
        print("-" * 30)
        for chunk in quick_generate_ollama_stream(prompt):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"❌ Ollama streaming failed: {e}\n")
    
    # Test Gemini streaming (if available)
    try:
        print("📡 Gemini Streaming Response:")
        print("-" * 30)
        for chunk in quick_generate_gemini_stream(prompt):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"❌ Gemini streaming failed: {e}\n")


def demo_streaming_chat():
    """Demonstrate streaming chat service"""
    print("=" * 60)
    print("💬 STREAMING CHAT SERVICE DEMO")
    print("=" * 60)
    
    message = "Tell me an interesting fact about the ocean"
    
    print(f"Message: {message}\n")
    
    # Test Ollama streaming chat (if available)
    try:
        print("📡 Ollama Chat Streaming:")
        print("-" * 30)
        for chunk in quick_chat_ollama_stream(message):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"❌ Ollama chat streaming failed: {e}\n")
    
    # Test Gemini streaming chat (if available)
    try:
        print("📡 Gemini Chat Streaming:")
        print("-" * 30)
        for chunk in quick_chat_gemini_stream(message):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"❌ Gemini chat streaming failed: {e}\n")


def demo_streaming_conversation():
    """Demonstrate streaming conversation service"""
    print("=" * 60)
    print("🗣️ STREAMING CONVERSATION SERVICE DEMO")
    print("=" * 60)
    
    try:
        # Create a conversation service
        conv = create_ollama_conversation()
        conv.add_system_message("You are a helpful assistant that gives concise answers.")
        
        print("📡 Conversation Streaming:")
        print("-" * 30)
        
        # Send streaming message
        message = "What are the benefits of renewable energy?"
        print(f"User: {message}\n")
        print("Assistant: ", end='', flush=True)
        
        for chunk in conv.send_message_stream(message):
            print(chunk, end='', flush=True)
        print("\n")
        
        # Show conversation stats
        stats = conv.get_conversation_stats()
        print(f"📊 Conversation Stats: {stats['total_messages']} messages")
        
    except Exception as e:
        print(f"❌ Conversation streaming failed: {e}\n")


def compare_streaming_vs_regular():
    """Compare streaming vs regular responses"""
    print("=" * 60)
    print("⚖️ STREAMING VS REGULAR COMPARISON")
    print("=" * 60)
    
    from llm.generation_service import quick_generate_ollama
    
    prompt = "Explain quantum computing in one paragraph"
    
    print(f"Prompt: {prompt}\n")
    
    try:
        # Regular response
        print("🔄 Regular Response:")
        print("-" * 30)
        regular_response = quick_generate_ollama(prompt)
        print(regular_response)
        print()
        
        # Streaming response
        print("📡 Streaming Response:")
        print("-" * 30)
        streaming_chunks = []
        for chunk in quick_generate_ollama_stream(prompt):
            streaming_chunks.append(chunk)
            print(chunk, end='', flush=True)
        
        streaming_response = ''.join(streaming_chunks)
        print("\n")
        
        # Compare
        print("📋 Comparison:")
        print(f"Regular length: {len(regular_response)} chars")
        print(f"Streaming length: {len(streaming_response)} chars")
        print(f"Responses match: {regular_response.strip() == streaming_response.strip()}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


if __name__ == "__main__":
    print("🧪 STREAMING CAPABILITIES TEST SUITE")
    print("This example demonstrates the new streaming features.")
    print("Note: Some tests may fail if the respective LLM services are not configured.\n")
    
    try:
        demo_streaming_generation()
        demo_streaming_chat()
        demo_streaming_conversation()
        compare_streaming_vs_regular()
        
        print("✅ Streaming examples completed!")
        print("\n💡 Usage Tips:")
        print("- Use streaming for real-time user interfaces")
        print("- Fall back to regular APIs when streaming isn't supported")
        print("- Streaming maintains backward compatibility with existing code")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")