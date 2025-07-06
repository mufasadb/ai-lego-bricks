#!/usr/bin/env python3
"""
Example usage script for the Agent Workflow Visualizer

This script demonstrates how to use the visualizer programmatically
to parse and generate Mermaid diagrams from JSON workflows.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import visualizer modules
try:
    from visualizer.workflow_parser import WorkflowParser
    from visualizer.mermaid_generator import MermaidGenerator
except ImportError:
    # Fallback for direct execution
    from workflow_parser import WorkflowParser  
    from mermaid_generator import MermaidGenerator


def main():
    print("🔍 Agent Workflow Visualizer - Example Usage")
    print("=" * 50)
    
    # Initialize the services
    parser = WorkflowParser()
    generator = MermaidGenerator()
    
    # Find example JSON files
    examples_dir = Path(__file__).parent.parent.parent / "agent_orchestration" / "examples"
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        print("Please ensure you're running this from the project root directory")
        return
    
    # Get all JSON files
    json_files = list(examples_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in: {examples_dir}")
        return
    
    print(f"📂 Found {len(json_files)} JSON workflow files")
    print()
    
    # Process each file
    for i, json_file in enumerate(json_files[:3], 1):  # Limit to first 3 for demo
        print(f"📄 Processing {i}/{min(3, len(json_files))}: {json_file.name}")
        
        try:
            # Parse the workflow
            workflow = parser.parse_workflow_file(str(json_file))
            
            # Generate visualization
            result = generator.generate_with_statistics(workflow)
            
            # Display results
            print(f"   📊 Workflow: {workflow.name}")
            print(f"   📝 Description: {workflow.description or 'No description'}")
            print(f"   🔢 Steps: {len(workflow.steps)}")
            print(f"   🔗 Connections: {len(workflow.connections)}")
            print(f"   📈 Complexity Score: {result['complexity_score']}")
            
            # Show step breakdown
            stats = result['statistics']
            print(f"   📋 Step Types:")
            for step_type, count in stats['step_types'].items():
                print(f"      • {step_type}: {count}")
            
            # Optionally save diagram to file
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            diagram_file = output_dir / f"{json_file.stem}_diagram.md"
            with open(diagram_file, 'w') as f:
                f.write(f"# {workflow.name}\n\n")
                f.write(f"{workflow.description}\n\n")
                f.write("## Workflow Diagram\n\n")
                f.write("```mermaid\n")
                f.write(result['diagram'])
                f.write("\n```\n\n")
                f.write("## Legend\n\n")
                f.write("```mermaid\n")
                f.write(result['legend'])
                f.write("\n```\n\n")
                f.write("## Statistics\n\n")
                f.write(f"- **Total Steps:** {stats['total_steps']}\n")
                f.write(f"- **Connections:** {stats['connections']}\n")
                f.write(f"- **Entry Points:** {stats['entry_points']}\n")
                f.write(f"- **Exit Points:** {stats['exit_points']}\n")
                f.write(f"- **Complexity Score:** {result['complexity_score']}\n")
                f.write(f"- **Has Loops:** {stats['has_loops']}\n")
                f.write(f"- **Has Conditions:** {stats['has_conditions']}\n")
            
            print(f"   💾 Saved diagram to: {diagram_file}")
            
        except Exception as e:
            print(f"   ❌ Error processing {json_file.name}: {e}")
        
        print()
    
    print("✅ Processing complete!")
    print()
    print("🚀 To start the web interface, run:")
    print("   cd visualizer/web && python app.py")
    print()
    print("Then visit: http://localhost:5000")


def demonstrate_parser_features():
    """Demonstrate specific parser features"""
    print("\n🛠️  Parser Features Demonstration")
    print("=" * 40)
    
    # Create a simple test workflow
    test_workflow = {
        "name": "Demo Workflow",
        "description": "A simple workflow to demonstrate parsing",
        "config": {
            "default_llm_provider": "gemini"
        },
        "steps": [
            {
                "id": "input_step",
                "type": "input",
                "description": "Get user input",
                "outputs": ["user_message"]
            },
            {
                "id": "process_step",
                "type": "llm_chat",
                "description": "Process with LLM",
                "inputs": {
                    "message": {
                        "from_step": "input_step",
                        "field": "user_message"
                    }
                },
                "outputs": ["response"]
            },
            {
                "id": "condition_step",
                "type": "condition",
                "description": "Check response quality",
                "inputs": {
                    "response": {
                        "from_step": "process_step",
                        "field": "response"
                    }
                },
                "routes": {
                    "good": "output_step",
                    "bad": "process_step"
                }
            },
            {
                "id": "output_step",
                "type": "output",
                "description": "Output final result",
                "inputs": {
                    "result": {
                        "from_step": "process_step",
                        "field": "response"
                    }
                }
            }
        ]
    }
    
    # Parse it
    parser = WorkflowParser()
    workflow = parser.parse_workflow_dict(test_workflow)
    
    print(f"📊 Parsed workflow: {workflow.name}")
    print(f"📝 Steps: {len(workflow.steps)}")
    print(f"🔗 Connections: {len(workflow.connections)}")
    print(f"🚪 Entry points: {workflow.entry_points}")
    print(f"🏁 Exit points: {workflow.exit_points}")
    
    print("\n🔄 Connections:")
    for conn in workflow.connections:
        print(f"   {conn['from']} --{conn['type']}--> {conn['to']} ({conn['label']})")
    
    # Generate diagram
    generator = MermaidGenerator()
    result = generator.generate_with_statistics(workflow)
    
    print(f"\n📈 Complexity Score: {result['complexity_score']}")
    print("\n📊 Mermaid Diagram:")
    print("```mermaid")
    print(result['diagram'])
    print("```")


if __name__ == "__main__":
    main()
    demonstrate_parser_features()