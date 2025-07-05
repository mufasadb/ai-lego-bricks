"""
Example of how to use the new conversation and generation services
"""

# Import both old and new services for comparison
try:
    from .chat_service import ChatService, quick_chat_ollama, quick_chat_gemini
    OLD_SERVICE_AVAILABLE = True
except ImportError:
    OLD_SERVICE_AVAILABLE = False

from .conversation_service import ConversationService, create_gemini_conversation, create_ollama_conversation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))
from generation_service import GenerationService, quick_generate_gemini, quick_generate_ollama
from llm_factory import create_generation_service, create_conversation_service

def demo_new_architecture():
    print("=== NEW ARCHITECTURE: Generation vs Conversation Services ===\n")
    
    # Example 1: One-shot generation (no conversation history needed)
    print("1. One-shot generation (stateless):")
    gen_service = create_generation_service("gemini", temperature=0.7)
    
    response1 = gen_service.generate("What's the capital of France?")
    print(f"Generation service: {response1}")
    
    # Quick generation
    response2 = quick_generate_gemini("What's 2+2?")
    print(f"Quick generation: {response2}")
    print()
    
    # Example 2: Multi-turn conversation with rich state management
    print("2. Multi-turn conversation with state:")
    conversation = create_gemini_conversation(temperature=0.7)
    
    # Add system context
    conversation.add_system_message("You are a helpful Python programming tutor.")
    
    # Have a conversation
    response1 = conversation.send_message("Hi, I'm working on a Python project")
    print(f"User: Hi, I'm working on a Python project")
    print(f"AI: {response1}")
    
    response2 = conversation.send_message("Can you help me with debugging?")
    print(f"User: Can you help me with debugging?")
    print(f"AI: {response2}")
    
    # Demonstrate rich conversation access
    print(f"\nConversation insights:")
    print(f"  First prompt: {conversation.get_first_prompt()}")
    print(f"  Last response: {conversation.get_last_response()[:100]}...")
    print(f"  Total messages: {conversation.get_conversation_length()}")
    print(f"  Conversation ID: {conversation.conversation.id}")
    print()
    
    # Example 3: Choosing the right service for the task
    print("3. Choosing the right service:")
    
    # Use generation for analysis tasks
    document = "The quarterly revenue increased by 15%."
    analysis = gen_service.generate_with_system_prompt(
        f"Analyze: {document}",
        "You are a business analyst."
    )
    print(f"Document analysis (generation): {analysis}")
    
    # Use conversation for interactive planning
    planning_conv = create_gemini_conversation()
    planning_conv.add_system_message("You are a project planning assistant.")
    
    plan_response = planning_conv.send_message("I need to plan a migration project")
    print(f"Interactive planning (conversation): {plan_response}")
    
    # Continue the planning conversation
    details = planning_conv.send_message("What are the key phases?")
    print(f"Planning details: {details[:100]}...")
    print()
    
    # Example 4: Export and summarization
    print("4. Conversation export and summarization:")
    
    # Export conversation
    summary_text = conversation.get_conversation_summary()
    
    # Use generation service to summarize
    summary = gen_service.generate_with_system_prompt(
        f"Summarize this conversation: {summary_text}",
        "You are a summarization expert."
    )
    print(f"Conversation summary: {summary}")
    
    # Export as markdown
    markdown = conversation.export_conversation('markdown')
    print(f"\nMarkdown export preview:")
    print(markdown[:200] + "...")

def demo_backward_compatibility():
    """Show backward compatibility with old chat service"""
    if not OLD_SERVICE_AVAILABLE:
        print("Old chat service not available for comparison")
        return
        
    print("\n=== BACKWARD COMPATIBILITY COMPARISON ===\n")
    
    # Old way
    print("Old ChatService way:")
    old_chat = ChatService("gemini")
    history = []
    
    response1, history = old_chat.chat_with_history("What is Python?", history)
    print(f"Old service response: {response1}")
    print(f"History length: {len(history)}")
    
    # New way with same functionality
    print("\nNew ConversationService way:")
    new_conv = create_gemini_conversation()
    
    response2 = new_conv.send_message("What is Python?")
    print(f"New service response: {response2}")
    print(f"Total messages: {new_conv.get_conversation_length()}")
    print(f"Rich access - first prompt: {new_conv.get_first_prompt()}")

def main():
    print("CHAT SERVICE EVOLUTION DEMONSTRATION")
    print("From mixed single/multi-turn to clean separation\n")
    
    try:
        # Demonstrate new architecture
        demo_new_architecture()
        
        # Show backward compatibility
        demo_backward_compatibility()
        
        print("\n=== ARCHITECTURE BENEFITS ===")
        print("✓ Clear separation: Generation (stateless) vs Conversation (stateful)")
        print("✓ Rich conversation access: first prompt, last response, search, export")
        print("✓ Agent orchestrator can reference any conversation element")
        print("✓ Performance: No conversation overhead for simple generation")
        print("✓ Flexibility: Choose the right tool for each task")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have proper LLM credentials configured")

if __name__ == "__main__":
    main()