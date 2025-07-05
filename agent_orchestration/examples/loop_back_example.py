"""
Example demonstrating the enhanced loop-back system with iteration tracking.

This example shows how to create workflows that can loop back to previous steps
based on conditional evaluation, with proper iteration counting and context preservation.
"""

from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import AgentOrchestrator


def topic_evaluation_loop_example():
    """
    Example: PDF analysis with topic evaluation loop
    
    This workflow:
    1. Extracts text from PDF
    2. Finds 3 topics from the text
    3. Evaluates if topics are interesting
    4. If not interesting, loops back to step 2 with context from previous attempt
    5. Continues until interesting topics are found or max iterations reached
    """
    print("=== Topic Evaluation Loop Example ===")
    
    # Define the loop-back workflow
    loop_workflow = {
        "name": "topic_evaluation_loop",
        "description": "Find interesting topics from a PDF with loop-back capability",
        "config": {
            "default_llm_provider": "gemini",
            "max_iterations": 5  # Global max iterations per step
        },
        "steps": [
            {
                "id": "extract_text",
                "type": "input",
                "description": "Simulate PDF text extraction",
                "config": {
                    "value": """
                    Climate change represents one of the most pressing challenges of our time. 
                    The increasing global temperatures are causing significant changes in weather 
                    patterns, ice cap melting, and rising sea levels. Renewable energy sources 
                    such as solar, wind, and hydroelectric power offer promising solutions to 
                    reduce greenhouse gas emissions. Governments worldwide are implementing 
                    policies to transition away from fossil fuels and promote sustainable 
                    energy alternatives. The economic implications of this transition are 
                    substantial, requiring significant investments in new infrastructure 
                    and technology development.
                    """
                },
                "outputs": ["text"]
            },
            {
                "id": "find_topics",
                "type": "llm_structured",
                "description": "Find 3 interesting topics from the text",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "response_schema": {
                        "name": "TopicsResponse", 
                        "fields": {
                            "topics": {
                                "type": "list",
                                "description": "List of 3 interesting topics found in the text"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of why these topics were chosen"
                            }
                        }
                    }
                },
                "inputs": {
                    "message": "Analyze this text and find 3 specific, interesting topics. If this is a retry (iteration > 1), try to find different topics than before. Text: $text. Previous attempt result: $iteration_context.previous_result"
                },
                "outputs": ["topics", "reasoning"],
                "max_iterations": 4,  # Allow up to 4 attempts to find good topics
                "preserve_previous_results": True
            },
            {
                "id": "evaluate_topics",
                "type": "condition",
                "description": "Evaluate if the topics are interesting enough",
                "config": {
                    "condition_type": "llm_decision",
                    "condition_prompt": "Evaluate these topics for interestingness and relevance. Are they specific, actionable, and engaging?",
                    "route_options": ["interesting", "not_interesting"],
                    "provider": "gemini",
                    "temperature": 0.3
                },
                "inputs": {
                    "topics": {"from_step": "find_topics", "field": "topics"},
                    "reasoning": {"from_step": "find_topics", "field": "reasoning"},
                    "attempt_number": "$iteration_context.iteration_count"
                },
                "routes": {
                    "interesting": "final_output",
                    "not_interesting": "find_topics"  # Loop back to find different topics
                }
            },
            {
                "id": "final_output",
                "type": "output",
                "description": "Output the final interesting topics",
                "inputs": {
                    "final_topics": {"from_step": "find_topics", "field": "topics"},
                    "evaluation_reasoning": {"from_step": "evaluate_topics", "field": "reasoning"},
                    "total_attempts": {"from_step": "find_topics", "field": "topics"}  # This will trigger iteration context
                }
            }
        ]
    }
    
    # Create orchestrator and execute workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(loop_workflow)
    
    print("Starting topic evaluation loop workflow...")
    print("This will attempt to find interesting topics, looping back if needed.\n")
    
    # Execute with initial input
    result = orchestrator.execute_workflow(workflow, {
        "text": "Climate change, renewable energy, and policy implementation text..."
    })
    
    if result.success:
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {result.execution_time:.2f}s")
        print(f"📊 Final output: {result.final_output}")
        
        # Show iteration details
        print("\n📋 STEP EXECUTION SUMMARY:")
        for step_id, output in result.step_outputs.items():
            if isinstance(output, dict):
                print(f"  {step_id}: {type(output).__name__} with {len(output)} fields")
            else:
                print(f"  {step_id}: {str(output)[:100]}...")
        
    else:
        print(f"❌ Workflow failed: {result.error}")


def simple_counter_loop_example():
    """
    Simple example showing iteration counting and context preservation
    """
    print("\n=== Simple Counter Loop Example ===")
    
    counter_workflow = {
        "name": "counter_loop",
        "description": "Count up to a target number with loop-back",
        "config": {
            "default_llm_provider": "gemini",
            "max_iterations": 10
        },
        "steps": [
            {
                "id": "initialize",
                "type": "input",
                "config": {"value": 0},
                "outputs": ["current_count"]
            },
            {
                "id": "increment",
                "type": "llm_chat",
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-flash"
                },
                "inputs": {
                    "message": "Current count is $current_count. Add 1 to it and respond with just the number. Iteration: $iteration_context.iteration_count"
                },
                "outputs": ["new_count"],
                "max_iterations": 6,
                "preserve_previous_results": True
            },
            {
                "id": "check_target",
                "type": "condition",
                "config": {
                    "condition_type": "simple_comparison",
                    "left_value": {"from_step": "increment", "field": "new_count"},
                    "operator": ">=",
                    "right_value": 5
                },
                "routes": {
                    "true": "done",
                    "false": "increment"  # Loop back
                }
            },
            {
                "id": "done", 
                "type": "output",
                "inputs": {
                    "final_count": {"from_step": "increment", "field": "new_count"},
                    "iterations_taken": "$iteration_context.iteration_count"
                }
            }
        ]
    }
    
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(counter_workflow)
    
    print("Starting counter loop (count from 0 to 5)...")
    result = orchestrator.execute_workflow(workflow, {"current_count": 0})
    
    if result.success:
        print(f"✅ Counter reached target! Final result: {result.final_output}")
    else:
        print(f"❌ Counter failed: {result.error}")


if __name__ == "__main__":
    print("Enhanced Loop-Back System Examples")
    print("=" * 50)
    
    # Run the examples
    topic_evaluation_loop_example()
    simple_counter_loop_example()
    
    print("\n" + "=" * 60)
    print("🎉 LOOP-BACK EXAMPLES COMPLETE!")
    print("=" * 60)
    print("Successfully demonstrated:")
    print("✓ Conditional loop-back to previous steps")
    print("✓ Iteration counting and max iteration limits")
    print("✓ Context preservation between iterations")
    print("✓ Access to previous attempt results")
    print("✓ Flexible routing based on conditions")