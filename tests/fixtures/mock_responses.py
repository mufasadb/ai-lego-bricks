"""
Mock responses for external services used in testing.
"""

from typing import Dict, Any, List


class MockLLMResponses:
    """Mock responses for LLM services."""
    
    SIMPLE_CHAT = "Hello! I'm a test AI assistant. How can I help you today?"
    
    JSON_RESPONSE = {
        "result": "success",
        "analysis": {
            "sentiment": "positive",
            "confidence": 0.95,
            "keywords": ["test", "mock", "response"]
        }
    }
    
    ERROR_RESPONSE = "I'm sorry, but I encountered an error processing your request."
    
    STREAMING_CHUNKS = [
        "Hello",
        " there",
        "! I'm",
        " a test",
        " response",
        " that comes",
        " in chunks",
        "."
    ]
    
    DOCUMENT_ANALYSIS = """
    This document appears to be a technical specification for an AI system. 
    Key points identified:
    1. Modular architecture design
    2. Support for multiple AI providers  
    3. JSON-based workflow configuration
    4. Memory and conversation capabilities
    """
    
    VISION_ANALYSIS = """
    Image Analysis:
    - Content: The image shows a technical diagram
    - Colors: Predominantly blue and white
    - Text: Contains labels and arrows indicating workflow
    - Quality: High resolution, clear text
    """


class MockMemoryResponses:
    """Mock responses for memory services."""
    
    SAMPLE_MEMORIES = [
        {
            "memory_id": "mem_001",
            "content": "Machine learning is transforming software development",
            "metadata": {"category": "AI", "importance": "high"},
            "similarity": 0.95,
            "timestamp": "2025-01-01T10:00:00Z"
        },
        {
            "memory_id": "mem_002", 
            "content": "Python is a versatile programming language",
            "metadata": {"category": "Programming", "importance": "medium"},
            "similarity": 0.87,
            "timestamp": "2025-01-01T11:00:00Z"
        },
        {
            "memory_id": "mem_003",
            "content": "Vector databases enable semantic search",
            "metadata": {"category": "Database", "importance": "high"},
            "similarity": 0.82,
            "timestamp": "2025-01-01T12:00:00Z"
        }
    ]
    
    SEARCH_RESULTS = [
        {
            "memory_id": "mem_001",
            "content": "Machine learning is transforming software development",
            "metadata": {"category": "AI", "importance": "high"},
            "similarity": 0.95
        }
    ]


class MockTTSResponses:
    """Mock responses for TTS services."""
    
    AUDIO_DATA = b"fake_audio_data_bytes_here"
    
    STREAMING_AUDIO_CHUNKS = [
        b"audio_chunk_001",
        b"audio_chunk_002", 
        b"audio_chunk_003",
        b"audio_chunk_004"
    ]


class MockWorkflowResponses:
    """Mock responses for workflow execution."""
    
    SIMPLE_WORKFLOW = {
        "name": "SimpleTestWorkflow",
        "description": "A basic workflow for testing",
        "steps": [
            {
                "id": "step_1",
                "type": "llm_prompt",
                "prompt": "Say hello",
                "provider": "auto"
            }
        ]
    }
    
    COMPLEX_WORKFLOW = {
        "name": "ComplexTestWorkflow", 
        "description": "A multi-step workflow for testing",
        "steps": [
            {
                "id": "step_1",
                "type": "llm_prompt",
                "prompt": "Analyze this text: {input}",
                "provider": "auto"
            },
            {
                "id": "step_2",
                "type": "memory_store",
                "content": "{step_1.response}",
                "metadata": {"source": "workflow_test"}
            },
            {
                "id": "step_3",
                "type": "tts_generate",
                "text": "{step_1.response}",
                "provider": "auto"
            }
        ]
    }
    
    EXECUTION_RESULT = {
        "workflow_id": "test_workflow_123",
        "status": "completed",
        "steps": {
            "step_1": {
                "status": "completed",
                "response": "Analysis complete: The text appears to be a test input.",
                "execution_time": 1.2
            },
            "step_2": {
                "status": "completed", 
                "memory_id": "mem_test_123",
                "execution_time": 0.3
            },
            "step_3": {
                "status": "completed",
                "audio_file": "/tmp/test_audio.wav",
                "execution_time": 2.1
            }
        },
        "total_execution_time": 3.6
    }


class MockAPIResponses:
    """Mock API responses for external services."""
    
    ANTHROPIC_RESPONSE = {
        "content": [
            {
                "type": "text",
                "text": "This is a mock response from Anthropic Claude."
            }
        ],
        "id": "msg_test_123",
        "model": "claude-3-sonnet-20240229",
        "role": "assistant",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 12
        }
    }
    
    OPENAI_RESPONSE = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from OpenAI."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }
    
    GOOGLE_RESPONSE = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a mock response from Google Gemini."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        ]
    }
    
    SUPABASE_RESPONSE = {
        "data": [
            {
                "id": "test_id_123",
                "content": "Test memory content",
                "metadata": {"test": True},
                "embedding": [0.1, 0.2, 0.3],
                "created_at": "2025-01-01T10:00:00Z"
            }
        ],
        "error": None
    }
    
    NEO4J_RESPONSE = {
        "records": [
            {
                "data": {
                    "memory_id": "test_id_123",
                    "content": "Test memory content",
                    "metadata": {"test": True}
                }
            }
        ]
    }