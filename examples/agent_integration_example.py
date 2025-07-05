#!/usr/bin/env python3
"""
Agent Integration Example

Shows how to integrate JSON-defined agents into your own Python applications.
Demonstrates programmatic usage, error handling, and result processing.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_orchestration import AgentOrchestrator

def main():
    print("🤖 Agent Integration Example")
    print("How to use JSON agents in your Python applications")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Try to load a simple agent
    try:
        workflow = orchestrator.load_workflow_from_file("agents/simple_qa_agent.json")
        
        # Execute with inputs
        result = orchestrator.execute_workflow(workflow, {
            "user_input": "Hello from integration example!"
        })
        
        print(f"✅ Agent executed successfully!")
        print(f"📄 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the agents/ directory with JSON configurations")

if __name__ == "__main__":
    main()