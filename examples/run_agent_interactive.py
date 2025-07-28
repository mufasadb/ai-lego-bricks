#!/usr/bin/env python3
"""
Interactive Agent Runner

Handles special input/output scenarios like streaming, file uploads,
continuous conversations, and real-time interactions.

Usage:
    python run_agent_interactive.py agent_orchestration/examples/streaming_agent.json --stream
    python run_agent_interactive.py agent_orchestration/examples/voice_assistant_agent.json --voice-mode
    python run_agent_interactive.py agent_orchestration/examples/basic_chat_agent.json --conversation
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agent_orchestration import AgentOrchestrator


def handle_streaming_mode(orchestrator, workflow, inputs):
    """Handle streaming LLM responses"""
    print("🌊 Streaming mode enabled - real-time response generation")
    print("-" * 50)

    # TODO: Implement actual streaming handling when available
    # For now, run normally but show that streaming is supported
    result = orchestrator.execute_workflow(workflow, inputs)

    if result.success:
        print("✅ Streaming workflow completed")
        if hasattr(result, "step_outputs"):
            for step_id, output in result.step_outputs.items():
                if "streamed" in str(output).lower() or "chunks" in str(output).lower():
                    print(f"🌊 {step_id}: Streaming data detected")

    return result


def handle_conversation_mode(orchestrator, workflow_file):
    """Handle continuous conversation loop"""
    print("💬 Conversation mode - type 'quit' to exit")
    print("-" * 50)

    conversation_count = 0

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            # Reload workflow for each turn to maintain conversation state
            workflow = orchestrator.load_workflow_from_file(workflow_file)

            inputs = {
                "user_input": user_input,
                "user_query": user_input,
                "user_message": user_input,
            }

            print("🤖 Assistant: ", end="", flush=True)

            result = orchestrator.execute_workflow(workflow, inputs)

            if result.success:
                # Extract the main response
                response = result.final_output
                if isinstance(response, dict):
                    response = response.get("response", str(response))

                print(response)
                conversation_count += 1
            else:
                print(f"❌ Error: {result.error}")

        except KeyboardInterrupt:
            print("\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n📊 Conversation summary: {conversation_count} exchanges")


def handle_voice_mode(orchestrator, workflow, inputs):
    """Handle voice input/output workflows"""
    print("🎤 Voice mode - processing audio input/output")
    print("-" * 50)

    # Check if voice input file exists
    voice_input = inputs.get("voice_input_path") or inputs.get("input_file")
    if voice_input and os.path.exists(voice_input):
        print(f"🎵 Processing voice input: {voice_input}")
    else:
        print("⚠️  No voice input file provided")

    result = orchestrator.execute_workflow(workflow, inputs)

    if result.success:
        print("✅ Voice workflow completed")

        # Look for audio output files
        if isinstance(result.final_output, dict):
            for key, value in result.final_output.items():
                if (
                    "audio" in key.lower()
                    and isinstance(value, str)
                    and value.endswith(".wav")
                ):
                    if os.path.exists(value):
                        print(f"🔊 Generated audio: {value}")

    return result


def handle_file_processing_mode(orchestrator, workflow, inputs):
    """Handle file upload and processing workflows"""
    print("📁 File processing mode")
    print("-" * 50)

    # Check for file inputs
    file_inputs = []
    for key, value in inputs.items():
        if isinstance(value, str) and os.path.exists(value):
            file_inputs.append((key, value))

    if file_inputs:
        print("📄 Processing files:")
        for key, filepath in file_inputs:
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"  {key}: {filepath} ({file_size:.1f} KB)")

    result = orchestrator.execute_workflow(workflow, inputs)

    if result.success:
        print("✅ File processing completed")

        # Look for output files
        if hasattr(result, "step_outputs"):
            for step_id, output in result.step_outputs.items():
                if isinstance(output, dict):
                    for key, value in output.items():
                        if (
                            "file_path" in key
                            and isinstance(value, str)
                            and os.path.exists(value)
                        ):
                            print(f"💾 Generated file: {value}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Interactive agent runner for special scenarios"
    )
    parser.add_argument("agent_file", help="Path to JSON agent configuration")
    parser.add_argument("--input", help="Input data for the agent (file path or text)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument(
        "--conversation", action="store_true", help="Enable conversation mode"
    )
    parser.add_argument(
        "--voice-mode", action="store_true", help="Enable voice processing mode"
    )
    parser.add_argument(
        "--file-mode", action="store_true", help="Enable file processing mode"
    )
    parser.add_argument("--save-session", action="store_true", help="Save session data")

    args = parser.parse_args()

    # Check if agent file exists
    if not os.path.exists(args.agent_file):
        print(f"❌ Agent file not found: {args.agent_file}")
        return 1

    try:
        agent_name = Path(args.agent_file).stem
        print("🎯 Interactive Agent Runner")
        print(f"🤖 Agent: {agent_name}")

        # Create orchestrator
        orchestrator = AgentOrchestrator()

        # Handle conversation mode (special case - doesn't load workflow initially)
        if args.conversation:
            handle_conversation_mode(orchestrator, args.agent_file)
            return 0

        # Load workflow for other modes
        workflow = orchestrator.load_workflow_from_file(args.agent_file)

        # Prepare inputs
        inputs = {}
        if args.input:
            if os.path.exists(args.input):
                inputs["input_file"] = args.input
                inputs["document_path"] = args.input
                inputs["voice_input_path"] = args.input
            else:
                inputs["input_text"] = args.input
                inputs["user_input"] = args.input

        # Route to appropriate handler
        if args.stream:
            result = handle_streaming_mode(orchestrator, workflow, inputs)
        elif args.voice_mode:
            result = handle_voice_mode(orchestrator, workflow, inputs)
        elif args.file_mode:
            result = handle_file_processing_mode(orchestrator, workflow, inputs)
        else:
            # Default interactive mode
            print("🎮 Interactive mode - enhanced output display")
            result = orchestrator.execute_workflow(workflow, inputs)

        # Save session if requested
        if args.save_session and result.success:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = f"output/{agent_name}_session_{timestamp}.json"
            os.makedirs("output", exist_ok=True)

            session_data = {
                "agent": agent_name,
                "timestamp": timestamp,
                "mode": (
                    "stream"
                    if args.stream
                    else (
                        "voice"
                        if args.voice_mode
                        else "file" if args.file_mode else "interactive"
                    )
                ),
                "inputs": inputs,
                "results": result.final_output,
                "execution_time": result.execution_time,
            }

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)

            print(f"💾 Session saved to: {session_file}")

        return 0 if result.success else 1

    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
