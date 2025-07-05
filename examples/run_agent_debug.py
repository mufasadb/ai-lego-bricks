#!/usr/bin/env python3
"""
Debug Agent Runner

Detailed logging and debugging script for JSON agent configurations.
Shows step-by-step execution, data flow, and internal state.

Usage:
    python run_agent_debug.py agents/simple_chat_agent.json
    python run_agent_debug.py agents/document_analysis_agent.json --input document.pdf
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_orchestration import AgentOrchestrator

def print_step_info(step, step_num, total_steps):
    """Print detailed information about a workflow step"""
    print(f"\n{'='*60}")
    print(f"🔧 STEP {step_num}/{total_steps}: {step.id}")
    print(f"{'='*60}")
    print(f"Type: {step.type}")
    if hasattr(step, 'inputs') and step.inputs:
        print(f"Inputs: {json.dumps(step.inputs, indent=2)}")
    if hasattr(step, 'config') and step.config:
        print(f"Config: {json.dumps(step.config, indent=2)}")
    if hasattr(step, 'outputs') and step.outputs:
        print(f"Outputs: {step.outputs}")
    print(f"{'='*60}")

def print_data_flow(context, step_id):
    """Print current workflow context/data flow"""
    print(f"\n📊 DATA FLOW AFTER STEP '{step_id}':")
    print("-" * 40)
    for key, value in context.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... ({len(value)} chars)")
        else:
            print(f"  {key}: {value}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Debug run a JSON-defined agent workflow")
    parser.add_argument("agent_file", help="Path to JSON agent configuration")
    parser.add_argument("--input", help="Input data for the agent (file path or text)")
    parser.add_argument("--user-input", help="User input/query for the agent")
    parser.add_argument("--step-by-step", action="store_true", help="Pause after each step")
    parser.add_argument("--save-log", help="Save debug log to file")
    
    args = parser.parse_args()
    
    # Set up logging
    log_output = []
    def log(message):
        print(message)
        log_output.append(message)
    
    # Check if agent file exists
    if not os.path.exists(args.agent_file):
        log(f"❌ Agent file not found: {args.agent_file}")
        return 1
    
    try:
        start_time = time.time()
        
        # Load and display agent configuration
        log(f"🤖 LOADING AGENT: {args.agent_file}")
        log(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with open(args.agent_file, 'r') as f:
            config = json.load(f)
        
        log(f"\n📋 AGENT CONFIGURATION:")
        log(f"   Name: {config.get('name', 'Unnamed')}")
        log(f"   Description: {config.get('description', 'No description')}")
        log(f"   Steps: {len(config.get('steps', []))}")
        log(f"   Config: {config.get('config', {})}")
        
        # Show full JSON if requested
        log(f"\n📄 Full Configuration:")
        log(json.dumps(config, indent=2))
        
        # Create orchestrator and load workflow
        log(f"\n🏗️ CREATING ORCHESTRATOR...")
        orchestrator = AgentOrchestrator()
        workflow = orchestrator.load_workflow_from_file(args.agent_file)
        
        # Prepare inputs
        inputs = {}
        if args.input:
            if os.path.exists(args.input):
                inputs["input_file"] = args.input
                inputs["document_path"] = args.input
                log(f"📁 Input file: {args.input}")
            else:
                inputs["input_text"] = args.input
                inputs["user_input"] = args.input
                log(f"💬 Input text: {args.input}")
        
        if args.user_input:
            inputs["user_input"] = args.user_input
            inputs["user_query"] = args.user_input
            log(f"❓ User input: {args.user_input}")
        
        log(f"\n🔤 WORKFLOW INPUTS:")
        log(json.dumps(inputs, indent=2))
        
        # Show workflow steps
        log(f"\n📋 WORKFLOW STEPS:")
        for i, step in enumerate(workflow.steps, 1):
            log(f"  {i}. {step.id} ({step.type})")
        
        if args.step_by_step:
            input("\n⏸️  Press Enter to start execution...")
        
        # Execute the workflow with step-by-step logging
        log(f"\n🚀 STARTING EXECUTION...")
        
        # We'll need to create a custom execution method or hook into the orchestrator
        # For now, let's execute normally and show the result
        result = orchestrator.execute_workflow(workflow, inputs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display results
        log(f"\n✅ EXECUTION COMPLETED!")
        log(f"⏰ Execution time: {execution_time:.2f} seconds")
        log(f"📄 FINAL RESULTS:")
        log("=" * 50)
        
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 200:
                    log(f"{key}: {value[:200]}... ({len(value)} chars)")
                else:
                    log(f"{key}: {value}")
        else:
            log(str(result))
        
        log("=" * 50)
        
        # Save log if requested
        if args.save_log:
            with open(args.save_log, 'w') as f:
                f.write('\n'.join(log_output))
            log(f"\n💾 Debug log saved to: {args.save_log}")
            
    except FileNotFoundError as e:
        log(f"❌ File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        log(f"❌ Invalid JSON in agent file: {e}")
        return 1
    except Exception as e:
        log(f"❌ Error running agent: {e}")
        log(f"🐛 Exception type: {type(e).__name__}")
        import traceback
        log(f"📍 Traceback:")
        log(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())