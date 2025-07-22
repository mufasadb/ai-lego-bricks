#!/usr/bin/env python3
"""
Test script for enhanced streaming buffer functionality
"""

import sys
import os
import time
import json

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from agent_orchestration.models import StreamBufferConfig
    from agent_orchestration.streaming_buffer import StreamBuffer
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to import from local development path...")
    # Try direct import from local files
    import importlib.util
    
    models_path = os.path.join(project_root, "agent_orchestration", "models.py")
    buffer_path = os.path.join(project_root, "agent_orchestration", "streaming_buffer.py")
    
    # Load models module
    spec = importlib.util.spec_from_file_location("models", models_path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    StreamBufferConfig = models.StreamBufferConfig
    
    # Load streaming_buffer module  
    spec = importlib.util.spec_from_file_location("streaming_buffer", buffer_path)
    streaming_buffer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streaming_buffer)
    StreamBuffer = streaming_buffer.StreamBuffer


def simulate_llm_stream():
    """Simulate an LLM streaming response with sentences."""
    text_chunks = [
        "Hello", " there", "! How", " are", " you", " doing", " today?",
        " I", " hope", " you're", " having", " a", " wonderful", " day.",
        " The", " weather", " is", " quite", " nice", " outside.",
        " Would", " you", " like", " to", " know", " more", " about",
        " streaming", " technology?", " It's", " really", " fascinating",
        " how", " we", " can", " process", " text", " in", " real-time."
    ]
    
    for chunk in text_chunks:
        yield chunk
        time.sleep(0.1)  # Simulate streaming delay


def test_sentence_buffering():
    """Test sentence-based buffering strategy."""
    print("🔵 Testing Sentence-Based Buffering")
    print("=" * 50)
    
    config = StreamBufferConfig(
        forward_on="sentence",
        sentence_count=1,
        max_buffer_time=2.0,
        min_chunk_length=5
    )
    
    buffer = StreamBuffer(config)
    forwarded_chunks = []
    
    for chunk in simulate_llm_stream():
        print(f"📥 Received chunk: '{chunk}'")
        
        ready_chunks = buffer.process_chunk(chunk)
        if ready_chunks:
            print(f"✅ Forwarded: {ready_chunks}")
            forwarded_chunks.extend(ready_chunks)
    
    # Flush remaining content
    final_chunks = buffer.flush()
    if final_chunks:
        print(f"🔚 Final flush: {final_chunks}")
        forwarded_chunks.extend(final_chunks)
    
    stats = buffer.get_stats()
    print(f"\n📊 Buffer Stats: {json.dumps(stats, indent=2)}")
    print(f"📦 Total forwarded chunks: {len(forwarded_chunks)}")
    print(f"📝 Forwarded content: {forwarded_chunks}\n")


def test_word_count_buffering():
    """Test word count-based buffering strategy."""
    print("🟡 Testing Word Count-Based Buffering")
    print("=" * 50)
    
    config = StreamBufferConfig(
        forward_on="word_count",
        word_count=5,
        max_buffer_time=2.0,
        min_chunk_length=5
    )
    
    buffer = StreamBuffer(config)
    forwarded_chunks = []
    
    for chunk in simulate_llm_stream():
        print(f"📥 Received chunk: '{chunk}'")
        
        ready_chunks = buffer.process_chunk(chunk)
        if ready_chunks:
            print(f"✅ Forwarded: {ready_chunks}")
            forwarded_chunks.extend(ready_chunks)
    
    # Flush remaining content
    final_chunks = buffer.flush()
    if final_chunks:
        print(f"🔚 Final flush: {final_chunks}")
        forwarded_chunks.extend(final_chunks)
    
    stats = buffer.get_stats()
    print(f"\n📊 Buffer Stats: {json.dumps(stats, indent=2)}")
    print(f"📦 Total forwarded chunks: {len(forwarded_chunks)}")
    print(f"📝 Forwarded content: {forwarded_chunks}\n")


def test_time_based_buffering():
    """Test time-based buffering strategy."""
    print("🟢 Testing Time-Based Buffering")
    print("=" * 50)
    
    config = StreamBufferConfig(
        forward_on="time",
        max_buffer_time=0.5,  # Force forward every 0.5 seconds
        min_chunk_length=5
    )
    
    buffer = StreamBuffer(config)
    forwarded_chunks = []
    
    for chunk in simulate_llm_stream():
        print(f"📥 Received chunk: '{chunk}'")
        
        ready_chunks = buffer.process_chunk(chunk)
        if ready_chunks:
            print(f"✅ Forwarded: {ready_chunks}")
            forwarded_chunks.extend(ready_chunks)
    
    # Flush remaining content
    final_chunks = buffer.flush()
    if final_chunks:
        print(f"🔚 Final flush: {final_chunks}")
        forwarded_chunks.extend(final_chunks)
    
    stats = buffer.get_stats()
    print(f"\n📊 Buffer Stats: {json.dumps(stats, indent=2)}")
    print(f"📦 Total forwarded chunks: {len(forwarded_chunks)}")
    print(f"📝 Forwarded content: {forwarded_chunks}\n")


def test_immediate_buffering():
    """Test immediate (pass-through) buffering strategy."""
    print("🟣 Testing Immediate (Pass-through) Buffering")
    print("=" * 50)
    
    config = StreamBufferConfig(
        forward_on="immediate"
    )
    
    buffer = StreamBuffer(config)
    forwarded_chunks = []
    
    for chunk in simulate_llm_stream():
        print(f"📥 Received chunk: '{chunk}'")
        
        ready_chunks = buffer.process_chunk(chunk)
        if ready_chunks:
            print(f"✅ Forwarded: {ready_chunks}")
            forwarded_chunks.extend(ready_chunks)
    
    stats = buffer.get_stats()
    print(f"\n📊 Buffer Stats: {json.dumps(stats, indent=2)}")
    print(f"📦 Total forwarded chunks: {len(forwarded_chunks)}")
    print(f"📝 Forwarded content: {forwarded_chunks}\n")


def main():
    """Run all streaming buffer tests."""
    print("🚀 Enhanced Streaming Buffer Tests")
    print("=" * 60)
    print("Testing different buffering strategies for intelligent stream forwarding.\n")
    
    try:
        test_sentence_buffering()
        test_word_count_buffering() 
        test_time_based_buffering()
        test_immediate_buffering()
        
        print("✅ All streaming buffer tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())