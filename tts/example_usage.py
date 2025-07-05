"""
Example usage of the TTS module
"""

import os
from tts import create_tts_service, get_available_providers, get_provider_info


def test_coqui_xtts():
    """Test Coqui-XTTS local instance"""
    print("Testing Coqui-XTTS...")
    
    try:
        # Create Coqui-XTTS service
        tts = create_tts_service("coqui_xtts")
        
        # Check if available
        if not tts.is_available():
            print("❌ Coqui-XTTS server not available at the configured URL")
            return
        
        print("✅ Coqui-XTTS server is available")
        
        # Get available voices
        voices = tts.get_available_voices()
        print(f"Available voices: {list(voices.keys())}")
        
        # Test synthesis
        test_text = "Hello from Coqui-XTTS! This is a test of the text-to-speech system."
        response = tts.text_to_speech(test_text, output_path="output/coqui_test.wav")
        
        if response.success:
            print(f"✅ Speech generated successfully: {response.audio_file_path}")
            print(f"   Duration: {response.duration_ms}ms")
            print(f"   Voice: {response.voice_used}")
            print(f"   Format: {response.format_used}")
        else:
            print(f"❌ Speech generation failed: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Error testing Coqui-XTTS: {e}")


def test_openai_tts():
    """Test OpenAI TTS"""
    print("\nTesting OpenAI TTS...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping OpenAI TTS test")
        return
    
    try:
        # Create OpenAI TTS service
        tts = create_tts_service("openai")
        
        # Check if available
        if not tts.is_available():
            print("❌ OpenAI TTS API not available")
            return
        
        print("✅ OpenAI TTS is available")
        
        # Get available voices
        voices = tts.get_available_voices()
        print(f"Available voices: {list(voices.keys())}")
        
        # Test synthesis with different voice
        test_text = "Hello from OpenAI TTS! This is a test with the Nova voice."
        response = tts.text_to_speech(
            test_text, 
            voice="nova",
            output_path="output/openai_test.mp3"
        )
        
        if response.success:
            print(f"✅ Speech generated successfully: {response.audio_file_path}")
            print(f"   Duration: {response.duration_ms}ms")
            print(f"   Voice: {response.voice_used}")
            print(f"   Format: {response.format_used}")
        else:
            print(f"❌ Speech generation failed: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Error testing OpenAI TTS: {e}")


def test_google_tts():
    """Test Google TTS"""
    print("\nTesting Google TTS...")
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set, skipping Google TTS test")
        return
    
    try:
        # Create Google TTS service
        tts = create_tts_service("google")
        
        # Check if available
        if not tts.is_available():
            print("❌ Google TTS API not available")
            return
        
        print("✅ Google TTS is available")
        
        # Test synthesis
        test_text = "Hello from Google Text-to-Speech! This is a test of the system."
        response = tts.text_to_speech(
            test_text,
            language_code="en-US",
            output_path="output/google_test.wav"
        )
        
        if response.success:
            print(f"✅ Speech generated successfully: {response.audio_file_path}")
            print(f"   Duration: {response.duration_ms}ms")
            print(f"   Voice: {response.voice_used}")
            print(f"   Format: {response.format_used}")
        else:
            print(f"❌ Speech generation failed: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Error testing Google TTS: {e}")


def test_auto_detection():
    """Test automatic provider detection"""
    print("\nTesting automatic provider detection...")
    
    try:
        # Get available providers
        providers = get_available_providers()
        print(f"Available providers: {providers}")
        
        # Get detailed provider info
        provider_info = get_provider_info()
        for provider, info in provider_info.items():
            status = "✅ Available" if info["available"] else "❌ Not available"
            print(f"{provider}: {status} - {info['description']}")
        
        # Create service with auto-detection
        tts = create_tts_service("auto")
        print(f"\nAuto-selected provider: {tts.config.provider.value}")
        
        # Test with auto-selected provider
        test_text = "This is a test using auto-detected TTS provider."
        response = tts.text_to_speech(test_text, output_path="output/auto_test.mp3")
        
        if response.success:
            print(f"✅ Auto-provider synthesis successful: {response.audio_file_path}")
            print(f"   Provider: {response.provider}")
            print(f"   Voice: {response.voice_used}")
        else:
            print(f"❌ Auto-provider synthesis failed: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Error in auto-detection test: {e}")


def test_voice_switching():
    """Test voice switching functionality"""
    print("\nTesting voice switching...")
    
    try:
        # Create a TTS service
        tts = create_tts_service("auto")
        
        if not tts.is_available():
            print("❌ No TTS provider available for voice switching test")
            return
        
        # Get available voices
        voices = tts.get_available_voices()
        available_voice_names = list(voices.keys())
        
        if len(available_voice_names) < 2:
            print("❌ Not enough voices available for switching test")
            return
        
        print(f"Testing voice switching with provider: {tts.config.provider.value}")
        
        # Test with first voice
        first_voice = available_voice_names[0]
        response1 = tts.text_to_speech(
            f"This is voice {first_voice}.",
            voice=first_voice,
            output_path=f"output/voice_{first_voice}.mp3"
        )
        
        # Test with second voice
        second_voice = available_voice_names[1]
        response2 = tts.text_to_speech(
            f"This is voice {second_voice}.",
            voice=second_voice,
            output_path=f"output/voice_{second_voice}.mp3"
        )
        
        if response1.success and response2.success:
            print(f"✅ Voice switching test successful")
            print(f"   Voice 1: {response1.voice_used} -> {response1.audio_file_path}")
            print(f"   Voice 2: {response2.voice_used} -> {response2.audio_file_path}")
        else:
            print(f"❌ Voice switching test failed")
            if not response1.success:
                print(f"   Voice 1 error: {response1.error_message}")
            if not response2.success:
                print(f"   Voice 2 error: {response2.error_message}")
                
    except Exception as e:
        print(f"❌ Error in voice switching test: {e}")


def main():
    """Run all TTS tests"""
    print("🎤 AI Lego Bricks TTS Module Tests")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run tests
    test_auto_detection()
    test_coqui_xtts()
    test_openai_tts()
    test_google_tts()
    test_voice_switching()
    
    print("\n🎉 TTS testing complete!")
    print("Check the 'output' directory for generated audio files.")


if __name__ == "__main__":
    main()