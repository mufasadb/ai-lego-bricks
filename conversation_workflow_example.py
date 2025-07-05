#!/usr/bin/env python3
"""
Example of using conversation workflows with the agent orchestrator

This demonstrates how conversation threads work within workflows,
providing lightweight conversation memory without requiring persistent storage.
"""

import json
from agent_orchestration.orchestrator import AgentOrchestrator

def run_simple_conversation():
    """Run a simple conversation workflow"""
    print("=== Simple Conversation Workflow ===")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Load conversation workflow
    workflow = orchestrator.load_workflow_from_file(
        "agent_orchestration/examples/conversation_chat_agent.json"
    )
    
    # Simulate user input (in real usage, this would come from actual user input)
    initial_inputs = {
        "user_query": "What is the difference between lists and tuples in Python?"
    }
    
    # Execute workflow
    result = orchestrator.execute_workflow(workflow, initial_inputs)
    
    if result.success:
        print("✓ Conversation completed successfully")
        print(f"Response: {result.final_output}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Show conversation state
        if 'continue_conversation' in result.step_outputs:
            conversation_result = result.step_outputs['continue_conversation']
            print(f"Total messages in conversation: {conversation_result.get('total_messages', 'N/A')}")
    else:
        print(f"✗ Workflow failed: {result.error}")

def run_multi_turn_conversation():
    """Run a multi-turn conversation workflow"""
    print("\n=== Multi-Turn Conversation Workflow ===")
    
    orchestrator = AgentOrchestrator()
    
    workflow = orchestrator.load_workflow_from_file(
        "agent_orchestration/examples/multi_turn_conversation.json"
    )
    
    # Simulate a learning session
    initial_inputs = {
        "initial_question": "What are Python decorators?",
        "follow_up": "Can you show me a simple example of creating my own decorator?"
    }
    
    result = orchestrator.execute_workflow(workflow, initial_inputs)
    
    if result.success:
        print("✓ Multi-turn conversation completed")
        
        # Parse the JSON output
        if isinstance(result.final_output, str):
            try:
                session_data = json.loads(result.final_output)
                print(f"\nLearning Session Summary:")
                print(f"Initial Question: {session_data.get('initial_question', 'N/A')}")
                print(f"Follow-up: {session_data.get('follow_up_question', 'N/A')}")
                print(f"Total Messages: {session_data.get('total_messages', 'N/A')}")
                print(f"Session Summary: {session_data.get('session_summary', 'N/A')[:200]}...")
            except json.JSONDecodeError:
                print("Could not parse session output as JSON")
                print(f"Raw output: {result.final_output}")
        else:
            print(f"Session output: {result.final_output}")
    else:
        print(f"✗ Multi-turn workflow failed: {result.error}")

def run_manual_message_conversation():
    """Run conversation with manual message management"""
    print("\n=== Conversation with Manual Messages ===")
    
    orchestrator = AgentOrchestrator()
    
    workflow = orchestrator.load_workflow_from_file(
        "agent_orchestration/examples/conversation_with_manual_messages.json"
    )
    
    # Simulate customer support scenario
    initial_inputs = {
        "customer_issue": "I was charged twice for my monthly subscription this month. Can you help me get a refund for the duplicate charge?"
    }
    
    result = orchestrator.execute_workflow(workflow, initial_inputs)
    
    if result.success:
        print("✓ Support conversation completed")
        
        if isinstance(result.final_output, str):
            try:
                support_data = json.loads(result.final_output)
                print(f"\nSupport Session Summary:")
                print(f"Customer Issue: {support_data.get('customer_issue', 'N/A')}")
                print(f"Agent Response: {support_data.get('agent_response', 'N/A')[:200]}...")
                print(f"Escalation Needed: {support_data.get('escalation_decision', 'N/A')}")
                print(f"Total Messages: {support_data.get('total_messages', 'N/A')}")
            except json.JSONDecodeError:
                print(f"Raw output: {result.final_output}")
        else:
            print(f"Support output: {result.final_output}")
    else:
        print(f"✗ Support workflow failed: {result.error}")

def demonstrate_conversation_benefits():
    """Explain the benefits of conversation threads"""
    print("\n=== Conversation Thread Benefits ===")
    print("✓ Lightweight: No persistent storage required")
    print("✓ Context-aware: Each message has access to conversation history")
    print("✓ Flexible: Manually add messages or use automatic conversation steps")
    print("✓ Workflow-integrated: Conversation state managed within execution context")
    print("✓ Multi-conversation: Can handle multiple conversation threads in one workflow")
    print("✓ JSON-configurable: Easily define conversation workflows in JSON")
    print("\nStep Types Available:")
    print("• start_conversation: Initialize a new conversation thread")
    print("• add_to_conversation: Manually add messages to a conversation")
    print("• continue_conversation: Add user message and get AI response")
    print("• llm_chat (with use_conversation): Use conversation context in any LLM step")

if __name__ == "__main__":
    print("🗣️  Conversation Workflow Examples")
    print("=" * 50)
    
    try:
        # Run examples
        run_simple_conversation()
        run_multi_turn_conversation()
        run_manual_message_conversation()
        demonstrate_conversation_benefits()
        
        print("\n✅ All conversation examples completed!")
        print("\n💡 Next Steps:")
        print("1. Try modifying the JSON workflows to change conversation behavior")
        print("2. Experiment with different LLM providers (gemini, ollama, anthropic)")
        print("3. Add conditional logic based on conversation content")
        print("4. Combine conversations with memory storage for long-term context")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("• Make sure you have the required environment variables set")
        print("• Check that the LLM services are available")
        print("• Verify the JSON workflow files exist and are valid")