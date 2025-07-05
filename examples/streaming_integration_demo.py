#!/usr/bin/env python3
"""
Simple working demo of streaming LLM to TTS integration
"""

import os
import sys
import time
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def stream_and_generate_audio():
    """Stream LLM response and generate audio in real-time"""
    print("🌊 Streaming LLM to TTS Integration Demo")
    print("=" * 45)
    
    # Setup
    from llm.generation_service import GenerationService
    from llm.llm_types import LLMProvider
    from tts import create_tts_service, get_available_providers
    
    # Check available providers
    tts_providers = get_available_providers()
    available_tts = [p for p, avail in tts_providers.items() if avail]
    
    if not available_tts:
        print("❌ No TTS providers available!")
        return
    
    print(f"Using TTS provider: {available_tts[0]}")
    
    # Create services
    llm_service = GenerationService(LLMProvider.OLLAMA)
    tts_service = create_tts_service(available_tts[0])
    
    # Create output directory
    output_dir = "output/streaming_integration"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test message
    message = "Explain machine learning in simple terms, including supervised and unsupervised learning."
    
    print(f"\n🤖 Message: {message}")
    print(f"📁 Output directory: {output_dir}")
    print("\n🎵 Starting streaming integration...\n")
    
    # Stream LLM response
    text_buffer = ""
    sentence_count = 0
    audio_files = []
    
    try:
        for chunk in llm_service.generate_stream(message):
            text_buffer += chunk
            print(chunk, end='', flush=True)
            
            # Check for complete sentences
            sentences = re.findall(r'[^.!?]*[.!?]+', text_buffer)
            
            if sentences:
                # Process complete sentences
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 15:  # Only process substantial sentences
                        sentence_count += 1
                        
                        # Generate audio for this sentence
                        audio_file = os.path.join(output_dir, f"sentence_{sentence_count:02d}.wav")
                        
                        print(f"\n🎵 Generating audio for sentence {sentence_count}...")
                        response = tts_service.text_to_speech(
                            text=sentence,
                            output_path=audio_file
                        )
                        
                        if response.success:
                            audio_files.append(response.audio_file_path)
                            print(f"✅ Audio saved: {response.audio_file_path}")
                        else:
                            print(f"❌ Audio generation failed: {response.error_message}")
                        
                        # Remove processed sentence from buffer
                        text_buffer = text_buffer.replace(sentence, "", 1)
        
        # Process any remaining text
        if text_buffer.strip():
            sentence_count += 1
            audio_file = os.path.join(output_dir, f"sentence_{sentence_count:02d}.wav")
            
            print(f"\n🎵 Generating audio for final text...")
            response = tts_service.text_to_speech(
                text=text_buffer.strip(),
                output_path=audio_file
            )
            
            if response.success:
                audio_files.append(response.audio_file_path)
                print(f"✅ Final audio saved: {response.audio_file_path}")
        
        # Create playlist
        playlist_path = os.path.join(output_dir, "playlist.m3u")
        with open(playlist_path, 'w') as f:
            f.write('#EXTM3U\n')
            for i, audio_file in enumerate(audio_files, 1):
                f.write(f'#EXTINF:-1,Sentence {i}\n')
                f.write(f'{os.path.basename(audio_file)}\n')
        
        print(f"\n🎉 Streaming integration completed!")
        print(f"Generated {len(audio_files)} audio files:")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"  {i}. {audio_file}")
        print(f"📝 Playlist created: {playlist_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_conversation_streaming():
    """Demo streaming a conversation with TTS"""
    print("\n💬 Conversation Streaming Demo")
    print("=" * 35)
    
    # Setup
    from llm.generation_service import GenerationService
    from llm.llm_types import LLMProvider
    from tts import create_tts_service, get_available_providers
    
    # Check providers
    tts_providers = get_available_providers()
    available_tts = [p for p, avail in tts_providers.items() if avail]
    
    if not available_tts:
        print("❌ No TTS providers available!")
        return False
    
    # Create services
    llm_service = GenerationService(LLMProvider.OLLAMA)
    tts_service = create_tts_service(available_tts[0])
    
    # Output directory
    output_dir = "output/conversation_streaming"
    os.makedirs(output_dir, exist_ok=True)
    
    # Conversation
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    all_audio_files = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔄 Question {i}: {question}")
        
        # Stream response
        print("Response: ", end='')
        full_response = ""
        
        try:
            for chunk in llm_service.generate_stream(question):
                full_response += chunk
                print(chunk, end='', flush=True)
            
            # Generate audio for complete response
            audio_file = os.path.join(output_dir, f"response_{i:02d}.wav")
            
            print(f"\n🎵 Generating audio...")
            response = tts_service.text_to_speech(
                text=full_response,
                output_path=audio_file
            )
            
            if response.success:
                all_audio_files.append(response.audio_file_path)
                print(f"✅ Audio saved: {response.audio_file_path}")
            else:
                print(f"❌ Audio failed: {response.error_message}")
                
        except Exception as e:
            print(f"❌ Error with question {i}: {e}")
    
    print(f"\n🎉 Conversation completed!")
    print(f"Generated {len(all_audio_files)} audio files:")
    for audio_file in all_audio_files:
        print(f"  - {audio_file}")
    
    return True


def main():
    """Run the streaming integration demos"""
    print("🌊 Streaming LLM to TTS Integration")
    print("=" * 50)
    
    try:
        # Run demos
        demo1 = stream_and_generate_audio()
        demo2 = demo_conversation_streaming()
        
        if demo1 and demo2:
            print("\n🎉 All demos completed successfully!")
            print("Check the output directories for generated audio files.")
        else:
            print("\n⚠️ Some demos had issues")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()