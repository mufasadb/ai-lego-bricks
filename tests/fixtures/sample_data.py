"""
Sample data for testing AI Lego Bricks functionality.
"""

from typing import Dict, List, Any
from datetime import datetime


class SampleDocuments:
    """Sample document content for testing."""
    
    TECHNICAL_SPEC = """
    # AI Agent System Technical Specification
    
    ## Overview
    This document outlines the technical architecture for a modular AI agent system.
    The system supports multiple AI providers and uses JSON-based workflow configuration.
    
    ## Architecture
    - Modular design with pluggable components
    - Provider abstraction layer
    - Memory service with vector search
    - Streaming response support
    
    ## Components
    1. Agent Orchestration
    2. LLM Services
    3. Memory Management  
    4. Text-to-Speech
    5. Prompt Management
    """
    
    RESEARCH_PAPER = """
    Abstract: This paper explores the applications of large language models in software development.
    We examine how AI assistants can improve developer productivity through code generation,
    documentation, and debugging assistance.
    
    Introduction: Large language models have shown remarkable capabilities in understanding
    and generating human-like text. Recent advances have extended these capabilities to
    code generation and software engineering tasks.
    
    Methodology: We conducted experiments using various LLM providers including OpenAI,
    Anthropic, and Google to evaluate their performance on common development tasks.
    
    Results: Our findings show significant improvements in developer productivity when
    AI assistants are integrated into the development workflow.
    """
    
    USER_MANUAL = """
    # User Manual - AI Lego Bricks
    
    ## Getting Started
    1. Install the package: pip install ai-lego-bricks
    2. Set up your environment variables
    3. Create your first workflow
    
    ## Basic Usage
    The system uses JSON workflows to define agent behavior.
    Each workflow consists of steps that can call different services.
    
    ## Examples
    - Simple chat agent
    - Document analysis
    - Memory-enhanced conversations
    - Multi-modal processing
    """


class SampleConversations:
    """Sample conversation data for testing."""
    
    SIMPLE_CHAT = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"},
        {"role": "user", "content": "Can you help me understand AI agents?"},
        {"role": "assistant", "content": "Certainly! AI agents are software programs that can perceive their environment and take actions to achieve specific goals. They use artificial intelligence to make decisions and interact with users or other systems."}
    ]
    
    TECHNICAL_DISCUSSION = [
        {"role": "user", "content": "Explain the difference between vector databases and traditional databases."},
        {"role": "assistant", "content": "Vector databases are specialized for storing and querying high-dimensional vectors, typically used for AI embeddings. Traditional databases store structured data in tables with rows and columns. Vector databases excel at similarity search and semantic queries."},
        {"role": "user", "content": "What are some popular vector database options?"},
        {"role": "assistant", "content": "Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and pgvector (PostgreSQL extension). Each has different strengths for various use cases and scale requirements."}
    ]
    
    DEBUGGING_SESSION = [
        {"role": "user", "content": "I'm getting a 'module not found' error in Python. Can you help?"},
        {"role": "assistant", "content": "I'd be happy to help with your Python import error. Can you share the specific error message and the import statement that's failing?"},
        {"role": "user", "content": "The error is 'ModuleNotFoundError: No module named 'requests'' when I try to import requests."},
        {"role": "assistant", "content": "This error means the 'requests' library isn't installed. You can fix it by running 'pip install requests' in your terminal, then try importing again."}
    ]


class SampleMetadata:
    """Sample metadata for testing."""
    
    DOCUMENT_METADATA = [
        {
            "document_id": "doc_001",
            "title": "AI Architecture Guide", 
            "author": "Tech Team",
            "created_date": "2025-01-01",
            "category": "technical",
            "tags": ["ai", "architecture", "guide"],
            "importance": "high"
        },
        {
            "document_id": "doc_002",
            "title": "User Manual v2.0",
            "author": "Documentation Team", 
            "created_date": "2025-01-02",
            "category": "documentation",
            "tags": ["manual", "guide", "user"],
            "importance": "medium"
        }
    ]
    
    MEMORY_METADATA = [
        {
            "type": "conversation",
            "source": "chat_session",
            "timestamp": "2025-01-01T10:00:00Z",
            "user_id": "user_123",
            "session_id": "session_456"
        },
        {
            "type": "document_analysis",
            "source": "document_processor",
            "timestamp": "2025-01-01T11:00:00Z", 
            "document_type": "pdf",
            "confidence": 0.95
        }
    ]


class SampleWorkflowInputs:
    """Sample inputs for workflow testing."""
    
    SIMPLE_INPUTS = {
        "text": "Hello world",
        "number": 42,
        "boolean": True,
        "list": ["item1", "item2", "item3"]
    }
    
    DOCUMENT_INPUTS = {
        "document_path": "/path/to/document.pdf",
        "document_content": SampleDocuments.TECHNICAL_SPEC,
        "analysis_type": "technical_review",
        "output_format": "summary"
    }
    
    CONVERSATION_INPUTS = {
        "user_message": "What is machine learning?",
        "conversation_history": SampleConversations.SIMPLE_CHAT,
        "context": "educational_discussion",
        "response_style": "detailed"
    }
    
    VISION_INPUTS = {
        "image_path": "/path/to/image.jpg",
        "image_url": "https://example.com/image.jpg", 
        "analysis_focus": "technical_diagram",
        "detail_level": "comprehensive"
    }


class SampleConfigurations:
    """Sample configuration data for testing."""
    
    LLM_CONFIGS = {
        "anthropic": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "openai": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "google": {
            "model": "gemini-pro",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    }
    
    MEMORY_CONFIGS = {
        "supabase": {
            "table_name": "memories",
            "embedding_dimension": 384,
            "similarity_threshold": 0.7
        },
        "neo4j": {
            "database": "neo4j",
            "node_label": "Memory",
            "relationship_type": "SIMILAR_TO"
        }
    }
    
    TTS_CONFIGS = {
        "openai": {
            "model": "tts-1",
            "voice": "alloy",
            "response_format": "mp3"
        },
        "google": {
            "voice_name": "en-US-Wavenet-D",
            "language_code": "en-US",
            "audio_encoding": "MP3"
        }
    }


class SampleErrors:
    """Sample error scenarios for testing."""
    
    API_ERRORS = {
        "rate_limit": {
            "error": "RateLimitError",
            "message": "API rate limit exceeded",
            "retry_after": 60
        },
        "auth_error": {
            "error": "AuthenticationError", 
            "message": "Invalid API key",
            "retry_after": None
        },
        "timeout": {
            "error": "TimeoutError",
            "message": "Request timed out",
            "retry_after": 5
        }
    }
    
    WORKFLOW_ERRORS = {
        "invalid_step": {
            "error": "ValidationError",
            "message": "Invalid step type 'invalid_type'",
            "step_id": "invalid_step"
        },
        "missing_variable": {
            "error": "VariableError", 
            "message": "Variable '{undefined_var}' not found",
            "step_id": "step_with_error"
        },
        "provider_error": {
            "error": "ProviderError",
            "message": "Provider 'nonexistent' not available",
            "step_id": "provider_step"
        }
    }


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def generate_test_id(prefix: str = "test") -> str:
    """Generate a test ID with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"