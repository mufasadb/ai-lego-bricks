"""
Example usage of the Agent Orchestration System

This demonstrates how to use the JSON-driven agent workflows
to create sophisticated AI agents using the existing building blocks.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the orchestration system
from agent_orchestration import AgentOrchestrator


def simple_chat_example():
    """Example: Simple chat agent"""
    print("=== Simple Chat Agent Example ===")
    
    # Create a simple workflow that works without interactive input
    simple_workflow = {
        "name": "simple_chat_agent",
        "description": "A simple chat agent that responds to queries",
        "config": {
            "default_llm_provider": "gemini"
        },
        "steps": [
            {
                "id": "generate_response",
                "type": "llm_chat",
                "description": "Generate AI response",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "$user_query"
                },
                "outputs": ["response"]
            }
        ]
    }
    
    # Create orchestrator and load workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(simple_workflow)
    
    # Test with multiple queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning briefly",
        "What are the benefits of Python programming?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")
        
        result = orchestrator.execute_workflow(workflow, {"user_query": query})
        
        if result.success:
            # Get the actual AI response from step outputs
            ai_response = result.step_outputs.get("generate_response.response", "No response found")
            print(f"AI Response: {ai_response}")
            print(f"Execution time: {result.execution_time:.2f}s")
        else:
            print(f"Error: {result.error}")


def document_analysis_example():
    """Example: Document analysis agent"""
    print("\n=== Document Analysis Agent Example ===")
    
    # Load workflow configuration
    config_path = Path(__file__).parent / "document_analysis_agent.json"
    
    # Create orchestrator and load workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_file(str(config_path))
    
    # Execute with initial inputs (you would provide a real PDF path)
    initial_inputs = {
        "document_path": "/path/to/your/document.pdf",
        "question": "What are the main findings in this document?"
    }
    
    result = orchestrator.execute_workflow(workflow, initial_inputs)
    
    if result.success:
        print(f"Analysis Result: {result.final_output}")
        print(f"Steps completed: {len(result.step_outputs)}")
        print(f"Execution time: {result.execution_time:.2f}s")
    else:
        print(f"Error: {result.error}")


def multi_step_analysis_example():
    """Example: Multi-step analysis workflow"""
    print("\n=== Multi-Step Analysis Example ===")
    
    # Define a multi-step analysis workflow
    analysis_workflow = {
        "name": "topic_analyzer",
        "description": "Analyze a topic from multiple perspectives",
        "config": {
            "default_llm_provider": "gemini"
        },
        "steps": [
            {
                "id": "define_topic",
                "type": "llm_chat",
                "description": "Define the topic",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "Provide a clear definition of: renewable energy"
                },
                "outputs": ["definition"]
            },
            {
                "id": "list_benefits",
                "type": "llm_chat",
                "description": "List benefits",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "List 3 main benefits of: renewable energy"
                },
                "outputs": ["benefits"]
            },
            {
                "id": "identify_challenges",
                "type": "llm_chat",
                "description": "Identify challenges",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "What are 3 main challenges of: renewable energy"
                },
                "outputs": ["challenges"]
            }
        ]
    }
    
    # Create orchestrator and execute workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(analysis_workflow)
    
    print("Analyzing topic: Renewable Energy")
    result = orchestrator.execute_workflow(workflow)
    
    if result.success:
        print(f"\n📋 ANALYSIS RESULTS")
        print(f"Topic: Renewable Energy")
        print(f"\n🔍 Definition:")
        print(result.step_outputs.get("define_topic.definition", "N/A"))
        print(f"\n✅ Benefits:")
        print(result.step_outputs.get("list_benefits.benefits", "N/A"))
        print(f"\n⚠️  Challenges:")
        print(result.step_outputs.get("identify_challenges.challenges", "N/A"))
        print(f"\n⏱️  Total time: {result.execution_time:.2f} seconds")
    else:
        print(f"Error: {result.error}")


def research_agent_example():
    """Example: Research agent with multiple documents"""
    print("\n=== Research Agent Example ===")
    
    # Load workflow configuration
    config_path = Path(__file__).parent / "research_agent.json"
    
    # Create orchestrator and load workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_file(str(config_path))
    
    # Execute with initial inputs
    initial_inputs = {
        "document_paths": [
            "/path/to/research_paper1.pdf",
            "/path/to/research_paper2.pdf"
        ],
        "research_query": "What are the latest developments in machine learning?"
    }
    
    result = orchestrator.execute_workflow(workflow, initial_inputs)
    
    if result.success:
        print(f"Research Report: {result.final_output}")
        print(f"Execution time: {result.execution_time:.2f}s")
    else:
        print(f"Error: {result.error}")


def loop_back_example():
    """Example: Simple loop-back workflow with iteration counting"""
    print("\n=== Loop-Back Workflow Example ===")
    
    # Define a simple loop-back workflow that tries to generate a good response
    loop_workflow = {
        "name": "loop_back_test",
        "description": "Test loop-back functionality with iteration tracking",
        "config": {
            "default_llm_provider": "gemini",
            "max_iterations": 3  # Global limit
        },
        "steps": [
            {
                "id": "generate_number",
                "type": "llm_chat",
                "description": "Generate a random number between 1-10",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "Generate a random number between 1 and 10. Just respond with the number only. This is attempt #$iteration_context.iteration_count"
                },
                "outputs": ["number"],
                "max_iterations": 3,
                "preserve_previous_results": True
            },
            {
                "id": "check_number",
                "type": "condition",
                "description": "Check if number is greater than 5",
                "config": {
                    "condition_type": "llm_decision",
                    "condition_prompt": "Is this number greater than 5?",
                    "route_options": ["yes", "no"],
                    "provider": "gemini"
                },
                "inputs": {
                    "number": {"from_step": "generate_number", "field": "number"}
                },
                "routes": {
                    "yes": "success",
                    "no": "generate_number"  # Loop back for another try
                }
            },
            {
                "id": "success",
                "type": "output",
                "description": "Output successful result",
                "inputs": {
                    "winning_number": {"from_step": "generate_number", "field": "number"},
                    "attempts_taken": "$iteration_context.iteration_count"
                }
            }
        ]
    }
    
    # Create orchestrator and execute
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(loop_workflow)
    
    print("Testing loop-back workflow (trying to get number > 5)...")
    result = orchestrator.execute_workflow(workflow)
    
    if result.success:
        print(f"✅ Loop-back test successful!")
        print(f"Final result: {result.final_output}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Show iteration details if available
        if hasattr(result, 'step_outputs'):
            print(f"Total steps executed: {len(result.step_outputs)}")
    else:
        print(f"❌ Loop-back test failed: {result.error}")


def show_available_step_types():
    """Show all available step types and their purposes"""
    print("\n=== Available Step Types ===")
    
    step_descriptions = {
        "input": "Collect user input or external data",
        "document_processing": "Extract and enhance text from PDFs",
        "memory_store": "Store content in vector/graph memory",
        "memory_retrieve": "Search and retrieve relevant memories",
        "llm_chat": "Generate text responses using LLM",
        "llm_structured": "Generate structured, validated LLM responses",
        "llm_vision": "Analyze images using vision models",
        "chunk_text": "Break text into semantic chunks",
        "condition": "Conditional execution based on criteria",
        "loop": "Iterate over collections or repeat steps (ENHANCED: now supports loop-back)",
        "output": "Format and return final results",
        "file_output": "Write workflow results to files (JSON, text, markdown, CSV)",
        "human_approval": "Human-in-the-loop approval and feedback collection"
    }
    
    for step_type, description in step_descriptions.items():
        print(f"  {step_type}: {description}")
    
    print("\n=== NEW: Loop-Back Features ===")
    print("  • max_iterations: Limit iterations per step")
    print("  • preserve_previous_results: Keep history of attempts")
    print("  • $iteration_context.iteration_count: Current attempt number")
    print("  • $iteration_context.previous_result: Result from last attempt")
    print("  • Routes can loop back to any previous step ID")


if __name__ == "__main__":
    print("Agent Orchestration System - Usage Examples")
    print("=" * 50)
    
    # Show available step types
    show_available_step_types()
    
    # Run working examples
    simple_chat_example()
    multi_step_analysis_example()
    loop_back_example()  # Test the new loop-back functionality
    
    print("\n" + "=" * 60)
    print("🎉 EXAMPLES COMPLETE!")
    print("=" * 60)
    print("Successfully demonstrated:")
    print("✓ Simple chat agent functionality")
    print("✓ Multi-step workflow execution")
    print("✓ JSON-driven agent configuration")
    print("✓ AI response generation and processing")
    print("✓ Enhanced loop-back system with iteration tracking")
    print("✓ Conditional routing to previous steps")
    print("✓ Iteration counting and max iteration limits")
    
    print("\nTo create your own agents:")
    print("1. Define workflows in JSON format")
    print("2. Use available step types (llm_chat, memory_store, etc.)")
    print("3. Chain steps together with input/output references")
    print("4. Execute with AgentOrchestrator.execute_workflow()")