#!/usr/bin/env python3
"""
Debug Agent Runner

Verbose debugging script for agent development and troubleshooting.
Shows detailed step-by-step execution, intermediate outputs, and timing information.

Usage:
    python run_agent_debug.py agent_orchestration/examples/basic_chat_agent.json --input "test question"
    python run_agent_debug.py agent_orchestration/examples/document_analysis_agent.json --input document.pdf --debug-level 2
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_orchestration import AgentOrchestrator

def print_step_details(step, step_output, step_time, debug_level=1):
    """Print detailed step information based on debug level"""
    print(f"\n{'='*60}")
    print(f"🔧 STEP: {step.id}")
    print(f"📝 Type: {step.type}")
    print(f"📄 Description: {step.description}")
    print(f"⏱️  Execution time: {step_time:.3f}s")
    
    if debug_level >= 2:
        print(f"⚙️  Config: {json.dumps(step.config, indent=2)}")
        if hasattr(step, 'inputs') and step.inputs:
            print(f"📥 Inputs: {json.dumps(step.inputs, indent=2, default=str)}")
    
    if debug_level >= 1:
        print("📤 Output:")
        if isinstance(step_output, dict):
            for key, value in step_output.items():
                if isinstance(value, str) and len(value) > 300:
                    print(f"  {key}: {value[:300]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        else:
            output_str = str(step_output)
            if len(output_str) > 500:
                print(f"  {output_str[:500]}... (truncated)")
            else:
                print(f"  {output_str}")

def main():
    parser = argparse.ArgumentParser(description="Debug a JSON-defined agent workflow")
    parser.add_argument("agent_file", help="Path to JSON agent configuration")
    parser.add_argument("--input", help="Input data for the agent (file path or text)")
    parser.add_argument("--user-input", help="User input/query for the agent")
    parser.add_argument("--debug-level", type=int, choices=[1, 2, 3], default=1,
                       help="Debug verbosity: 1=outputs only, 2=configs+inputs, 3=full details")
    parser.add_argument("--save-debug", action="store_true", help="Save debug log to file")
    parser.add_argument("--step-pause", action="store_true", help="Pause after each step")
    
    args = parser.parse_args()
    
    # Check if agent file exists
    if not os.path.exists(args.agent_file):
        print(f"❌ Agent file not found: {args.agent_file}")
        return 1
    
    
    try:
        # Load and display agent configuration
        agent_name = Path(args.agent_file).stem
        print(f"🐛 DEBUG MODE: Running agent '{agent_name}'")
        print(f"📁 Agent file: {args.agent_file}")
        print(f"🔍 Debug level: {args.debug_level}")
        
        with open(args.agent_file, 'r') as f:
            config = json.load(f)
            print("\n📋 Agent Configuration:")
            print(f"   Name: {config.get('name', 'Unnamed')}")
            print(f"   Description: {config.get('description', 'No description')}")
            print(f"   Steps: {len(config.get('steps', []))}")
            
            if args.debug_level >= 2:
                print(f"   Global Config: {json.dumps(config.get('config', {}), indent=4)}")
        
        # Create orchestrator and inject credential manager
        orchestrator = AgentOrchestrator()
        
        # Initialize credential manager for .env support
        try:
            from credentials import CredentialManager
            orchestrator._credential_manager = CredentialManager(load_env=True)
            print("✅ Credential manager initialized")
            
            # Check which credentials are available
            test_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"]
            print("🔍 Available credentials:")
            for key in test_keys:
                value = orchestrator._credential_manager.get_credential(key)
                status = "✅" if value else "❌"
                print(f"   {status} {key}")
                
        except ImportError:
            print("⚠️  Credential manager not available")
        
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
        
        print(f"\n🔧 Prepared inputs: {json.dumps(inputs, indent=2, default=str)}")
        
        # Execute with detailed monitoring
        print("\n🚀 Starting workflow execution...")
        start_time = time.time()
        
        # TODO: Implement step-by-step execution monitoring
        # For now, run normally and show results
        result = orchestrator.execute_workflow(workflow, inputs)
        
        total_time = time.time() - start_time
        
        # Show detailed results
        if result.success:
            print(f"\n{'='*60}")
            print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {total_time:.3f}s")
            
            if hasattr(result, 'step_outputs') and result.step_outputs:
                print(f"📊 Step outputs ({len(result.step_outputs)} steps):")
                for step_id, output in result.step_outputs.items():
                    print(f"\n  🔸 {step_id}:")
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if isinstance(value, str) and len(value) > 200:
                                print(f"    {key}: {value[:200]}... (truncated)")
                            else:
                                print(f"    {key}: {value}")
                    else:
                        output_str = str(output)
                        if len(output_str) > 300:
                            print(f"    {output_str[:300]}... (truncated)")
                        else:
                            print(f"    {output_str}")
            
            print("\n📄 Final Output:")
            print(f"{json.dumps(result.final_output, indent=2, default=str)}")
            
            # Save debug log if requested
            if args.save_debug:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_file = f"output/{agent_name}_debug_{timestamp}.json"
                os.makedirs("output", exist_ok=True)
                
                debug_data = {
                    "agent": agent_name,
                    "timestamp": timestamp,
                    "debug_level": args.debug_level,
                    "inputs": inputs,
                    "total_execution_time": total_time,
                    "step_outputs": result.step_outputs if hasattr(result, 'step_outputs') else {},
                    "final_output": result.final_output,
                    "success": result.success
                }
                
                with open(debug_file, 'w') as f:
                    json.dump(debug_data, f, indent=2, default=str)
                
                print(f"💾 Debug log saved to: {debug_file}")
        else:
            print("\n❌ WORKFLOW FAILED!")
            print(f"💥 Error: {result.error}")
            print(f"⏱️  Time before failure: {total_time:.3f}s")
            
            if hasattr(result, 'step_outputs'):
                print(f"📊 Completed steps: {len(result.step_outputs)}")
                for step_id, output in result.step_outputs.items():
                    print(f"  ✅ {step_id}")
            
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        if args.debug_level >= 2:
            print("📚 Full traceback:")
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())