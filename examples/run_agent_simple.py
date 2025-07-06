#!/usr/bin/env python3
"""
Simple Agent Runner

Clean, minimal script to run JSON agent configurations and save results.
This is the main way to execute agents defined in the agent_orchestration/examples/ directory.

Usage:
    python run_agent_simple.py agent_orchestration/examples/basic_chat_agent.json
    python run_agent_simple.py agent_orchestration/examples/document_analysis_agent.json --input document.pdf
    python run_agent_simple.py agent_orchestration/examples/voice_assistant_agent.json --input voice.wav
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_orchestration import AgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Run a JSON-defined agent workflow")
    parser.add_argument("agent_file", help="Path to JSON agent configuration")
    parser.add_argument("--input", help="Input data for the agent (file path or text)")
    parser.add_argument("--user-input", help="User input/query for the agent")
    parser.add_argument("--save-output", action="store_true", help="Save results to output folder")
    
    args = parser.parse_args()
    
    # Check if agent file exists
    if not os.path.exists(args.agent_file):
        print(f"❌ Agent file not found: {args.agent_file}")
        print(f"💡 Available agents:")
        examples_dir = Path("agent_orchestration/examples")
        if examples_dir.exists():
            for agent_file in examples_dir.glob("*.json"):
                print(f"   📋 {agent_file}")
        return 1
    
    try:
        # Load agent configuration
        agent_name = Path(args.agent_file).stem
        print(f"🤖 Running agent: {agent_name}")
        
        # Create orchestrator and load workflow
        orchestrator = AgentOrchestrator()
        workflow = orchestrator.load_workflow_from_file(args.agent_file)
        
        # Prepare inputs
        inputs = {}
        if args.input:
            if os.path.exists(args.input):
                inputs["input_file"] = args.input
                inputs["document_path"] = args.input
                inputs["pdf_path"] = args.input
                inputs["voice_input_path"] = args.input
            else:
                inputs["input_text"] = args.input
                inputs["user_input"] = args.input
        
        if args.user_input:
            inputs["user_input"] = args.user_input
            inputs["user_query"] = args.user_input
        
        # Execute the workflow
        print(f"🚀 Executing workflow...")
        result = orchestrator.execute_workflow(workflow, inputs)
        
        # Display results
        if result.success:
            print(f"\n✅ Agent completed successfully!")
            print(f"⏱️  Execution time: {result.execution_time:.2f}s")
            print(f"📄 Results:")
            print("-" * 50)
            
            if isinstance(result.final_output, dict):
                for key, value in result.final_output.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"{key}: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
            else:
                result_str = str(result.final_output)
                if len(result_str) > 500:
                    print(f"{result_str[:500]}...")
                else:
                    print(result_str)
            
            # Save output if requested
            if args.save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"output/{agent_name}_results_{timestamp}.json"
                os.makedirs("output", exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump({
                        "agent": agent_name,
                        "timestamp": timestamp,
                        "inputs": inputs,
                        "results": result.final_output,
                        "execution_time": result.execution_time,
                        "step_count": len(result.step_outputs) if hasattr(result, 'step_outputs') else 0
                    }, f, indent=2, default=str)
                
                print(f"💾 Results saved to: {output_file}")
        else:
            print(f"❌ Agent failed: {result.error}")
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in agent file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())