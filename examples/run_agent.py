#!/usr/bin/env python3
"""
Simple Agent Runner

Clean, minimal script to run JSON agent configurations and see results.
This is the main way to execute agents defined in the /agents/ directory.

Usage:
    python run_agent.py agents/simple_chat_agent.json
    python run_agent.py agents/document_analysis_agent.json --input document.pdf
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_orchestration import AgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Run a JSON-defined agent workflow")
    parser.add_argument("agent_file", help="Path to JSON agent configuration")
    parser.add_argument("--input", help="Input data for the agent (file path or text)")
    parser.add_argument("--user-input", help="User input/query for the agent")
    parser.add_argument("--show-config", action="store_true", help="Show agent configuration before running")
    
    args = parser.parse_args()
    
    # Check if agent file exists
    if not os.path.exists(args.agent_file):
        print(f"❌ Agent file not found: {args.agent_file}")
        print(f"💡 Available agents in agents/ directory:")
        agents_dir = Path("agents")
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.json"):
                print(f"   📋 {agent_file.name}")
        return 1
    
    try:
        # Load and optionally display agent configuration
        print(f"🤖 Loading agent: {args.agent_file}")
        
        if args.show_config:
            with open(args.agent_file, 'r') as f:
                config = json.load(f)
                print(f"📋 Agent Configuration:")
                print(f"   Name: {config.get('name', 'Unnamed')}")
                print(f"   Description: {config.get('description', 'No description')}")
                print(f"   Steps: {len(config.get('steps', []))}")
                print()
        
        # Create orchestrator and load workflow
        orchestrator = AgentOrchestrator()
        workflow = orchestrator.load_workflow_from_file(args.agent_file)
        
        # Prepare inputs
        inputs = {}
        if args.input:
            # Check if it's a file path or direct text
            if os.path.exists(args.input):
                inputs["input_file"] = args.input
                inputs["document_path"] = args.input  # Common alias
            else:
                inputs["input_text"] = args.input
                inputs["user_input"] = args.input
        
        if args.user_input:
            inputs["user_input"] = args.user_input
            inputs["user_query"] = args.user_input  # Common alias
        
        # If no inputs provided, check what the agent expects
        if not inputs:
            # Look at first step to see what inputs it expects
            first_step = workflow.steps[0] if workflow.steps else None
            if first_step and hasattr(first_step, 'inputs'):
                print("💡 This agent appears to expect inputs. Use:")
                print(f"   --input 'your text here' or --input /path/to/file")
                print(f"   --user-input 'your question here'")
        
        # Execute the workflow
        print(f"🚀 Running agent...")
        result = orchestrator.execute_workflow(workflow, inputs)
        
        # Display results
        print(f"\n✅ Agent completed successfully!")
        print(f"📄 Results:")
        print("-" * 50)
        
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print(result)
            
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