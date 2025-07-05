"""
Example of using TTS in agent workflows
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_orchestration import AgentOrchestrator


def run_simple_tts_workflow():
    """Run a simple TTS workflow"""
    print("🎤 Running Simple TTS Workflow")
    print("=" * 40)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Load TTS workflow
    workflow = orchestrator.load_workflow_from_file(
        "agent_orchestration/examples/simple_tts_agent.json"
    )
    
    # Execute with sample text
    result = orchestrator.execute_workflow(workflow, {
        "text_input": "Hello! This is a test of the text-to-speech system in AI Lego Bricks."
    })
    
    print(f"Workflow completed: {result.success}")
    if result.success:
        final_output = result.final_output
        print(f"Audio file created: {final_output.get('audio_file')}")
        print(f"TTS Success: {final_output.get('success')}")
        print(f"Duration: {final_output.get('duration_ms')}ms")
        print(f"Voice: {final_output.get('voice_used')}")
        print(f"Provider: {final_output.get('provider')}")
    else:
        print(f"Error: {result.error}")


def run_chat_with_voice_workflow():
    """Run chat with voice workflow"""
    print("\n💬 Running Chat with Voice Workflow")
    print("=" * 40)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Load chat workflow
    workflow = orchestrator.load_workflow_from_file(
        "agent_orchestration/examples/chat_with_voice_agent.json"
    )
    
    # Execute with sample query
    result = orchestrator.execute_workflow(workflow, {
        "user_query": "Tell me an interesting fact about artificial intelligence."
    })
    
    print(f"Workflow completed: {result.success}")
    if result.success:
        final_output = result.final_output
        print(f"Text Response: {final_output.get('text_response')}")
        print(f"Audio File: {final_output.get('audio_file')}")
        print(f"TTS Success: {final_output.get('tts_success')}")
        print(f"Voice Used: {final_output.get('voice_used')}")
        print(f"Provider Used: {final_output.get('provider_used')}")
    else:
        print(f"Error: {result.error}")


def demonstrate_provider_selection():
    """Demonstrate different TTS provider selection"""
    print("\n🔧 Demonstrating TTS Provider Selection")
    print("=" * 40)
    
    from tts import get_available_providers, create_tts_service
    
    # Check available providers
    providers = get_available_providers()
    print("Available TTS providers:")
    for provider, available in providers.items():
        status = "✅" if available else "❌"
        print(f"  {status} {provider}")
    
    # Test each available provider
    for provider_name, available in providers.items():
        if available:
            print(f"\nTesting {provider_name}...")
            try:
                tts = create_tts_service(provider_name)
                response = tts.text_to_speech(
                    f"This is a test using {provider_name}.",
                    output_path=f"output/test_{provider_name}.mp3"
                )
                
                if response.success:
                    print(f"  ✅ Success: {response.audio_file_path}")
                else:
                    print(f"  ❌ Failed: {response.error_message}")
            except Exception as e:
                print(f"  ❌ Error: {e}")


def create_custom_tts_workflow():
    """Create and run a custom TTS workflow"""
    print("\n🛠️ Creating Custom TTS Workflow")
    print("=" * 40)
    
    # Define a custom workflow
    custom_workflow = {
        "name": "multi_voice_demo",
        "description": "Demonstrate multiple voices in sequence",
        "steps": [
            {
                "id": "intro_speech",
                "type": "tts",
                "config": {
                    "provider": "auto",
                    "voice": "default",
                    "output_path": "output/intro.wav"
                },
                "inputs": {
                    "text": "Welcome to the AI Lego Bricks TTS demonstration."
                }
            },
            {
                "id": "feature_speech",
                "type": "tts", 
                "config": {
                    "provider": "auto",
                    "voice": "default",
                    "output_path": "output/features.wav",
                    "speed": 0.9
                },
                "inputs": {
                    "text": "This system supports multiple TTS providers including Coqui-XTTS, OpenAI, and Google."
                }
            },
            {
                "id": "outro_speech",
                "type": "tts",
                "config": {
                    "provider": "auto", 
                    "voice": "default",
                    "output_path": "output/outro.wav",
                    "speed": 1.1
                },
                "inputs": {
                    "text": "Thank you for trying the TTS capabilities!"
                }
            },
            {
                "id": "combine_results",
                "type": "output",
                "inputs": {
                    "intro_file": {"from_step": "intro_speech", "field": "audio_file_path"},
                    "feature_file": {"from_step": "feature_speech", "field": "audio_file_path"},
                    "outro_file": {"from_step": "outro_speech", "field": "audio_file_path"},
                    "all_successful": {
                        "from_step": "outro_speech",
                        "field": "success"
                    }
                }
            }
        ]
    }
    
    # Create orchestrator and run workflow
    orchestrator = AgentOrchestrator()
    workflow = orchestrator.load_workflow_from_dict(custom_workflow)
    
    result = orchestrator.execute_workflow(workflow)
    
    print(f"Custom workflow completed: {result.success}")
    if result.success:
        final_output = result.final_output
        print("Generated audio files:")
        print(f"  Intro: {final_output.get('intro_file')}")
        print(f"  Features: {final_output.get('feature_file')}")
        print(f"  Outro: {final_output.get('outro_file')}")
        print(f"All successful: {final_output.get('all_successful')}")
    else:
        print(f"Error: {result.error}")


def main():
    """Run all TTS workflow examples"""
    print("🎤 AI Lego Bricks TTS Workflow Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    try:
        # Check if any TTS providers are available
        from tts import get_available_providers
        providers = get_available_providers()
        
        if not any(providers.values()):
            print("❌ No TTS providers are available!")
            print("Please configure at least one TTS provider:")
            print("  - For Coqui-XTTS: Set COQUI_XTTS_URL in your .env file")
            print("  - For OpenAI: Set OPENAI_API_KEY environment variable")
            print("  - For Google: Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return
        
        # Run examples
        demonstrate_provider_selection()
        run_simple_tts_workflow()
        run_chat_with_voice_workflow() 
        create_custom_tts_workflow()
        
        print("\n🎉 All TTS workflow examples completed!")
        print("Check the 'output' directory for generated audio files.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()