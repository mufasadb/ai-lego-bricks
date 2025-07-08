# LLM Module

Unified abstraction layer for multiple LLM providers with consistent interfaces, credential management, and streaming support.

## Supported Providers

- **Text**: Ollama, Gemini, Anthropic
- **Vision**: Gemini Vision, LLaVA  
- **Embeddings**: SentenceTransformers

## Quick Start

### Basic Text Generation

```python
from llm import quick_generate_gemini, quick_generate_anthropic, quick_generate_ollama

# Quick one-shot generation
response = quick_generate_gemini("Explain quantum computing")
response = quick_generate_anthropic("Write a Python function", model="claude-3-5-sonnet-20241022")
response = quick_generate_ollama("What is machine learning?", model="llama2")
```

### Using Factory Pattern

```python
from llm.llm_factory import LLMClientFactory, LLMProvider

# Create text client
client = LLMClientFactory.create_text_client(
    provider=LLMProvider.GEMINI,
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=1000
)

# Basic chat
response = client.chat("Hello, how are you?")

# Chat with history
from llm.llm_types import ChatMessage
history = [ChatMessage(role="user", content="My name is Alex")]
response = client.chat("What's my name?", chat_history=history)
```

### Streaming Responses

```python
# Stream responses for real-time output
for chunk in client.chat_stream("Write a long story about AI"):
    print(chunk, end="", flush=True)
```

### Vision Processing

```python
from llm.llm_factory import LLMClientFactory, VisionProvider

vision_client = LLMClientFactory.create_vision_client(VisionProvider.GEMINI_VISION)

# Process image with prompt
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    
result = vision_client.process_image(image_data, "What's in this image?")
```

### Structured Responses

```python
from pydantic import BaseModel
from llm.llm_factory import LLMClientFactory, LLMProvider

class TaskAnalysis(BaseModel):
    priority: str
    estimated_hours: int
    complexity: str

# Create structured client
structured_client = LLMClientFactory.create_structured_client(
    provider=LLMProvider.ANTHROPIC,
    schema=TaskAnalysis
)

# Get validated structured response
analysis = structured_client.chat("Analyze this task: Build a web scraper")
print(f"Priority: {analysis.priority}, Hours: {analysis.estimated_hours}")
```

### Embeddings

```python
from llm.llm_factory import LLMClientFactory

embedding_client = LLMClientFactory.create_embedding_client()

# Single embedding
embedding = embedding_client.generate_embedding("Hello world")

# Batch embeddings
embeddings = embedding_client.generate_embeddings(["Text 1", "Text 2", "Text 3"])
print(f"Embedding dimension: {embedding_client.embedding_dimension}")
```

## Generation Service

For stateless one-shot interactions:

```python
from llm.generation_service import GenerationService, LLMProvider

service = GenerationService(
    provider=LLMProvider.GEMINI,
    model="gemini-1.5-flash",
    temperature=0.7
)

response = service.generate("Write a haiku about coding")
```

## Model Switching

```python
from llm.llm_factory import LLMClientFactory, LLMProvider

# Create switchable client
client = LLMClientFactory.create_switchable_text_client(LLMProvider.OLLAMA, "llama2")

# Switch model dynamically
client.switch_model("codellama")
print(f"Current model: {client.get_current_model()}")
```

## Environment Variables

Required credentials:
- `GOOGLE_AI_STUDIO_KEY` - For Gemini
- `ANTHROPIC_API_KEY` - For Anthropic
- `OLLAMA_URL` - For Ollama (defaults to localhost:11434)

Optional configuration:
- `GEMINI_DEFAULT_MODEL`, `ANTHROPIC_DEFAULT_MODEL`, `OLLAMA_DEFAULT_MODEL`
- `EMBEDDING_MODEL` - SentenceTransformer model name

## Key Features

- **Unified Interface**: Same API across all providers
- **Credential Management**: Secure credential handling via CredentialManager
- **Streaming Support**: Real-time response streaming where available
- **Model Switching**: Dynamic model changes without recreating clients
- **Structured Output**: Pydantic model validation for consistent responses
- **Error Handling**: Automatic retries and rate limit management
- **Vision Support**: Image processing with text prompts

## Architecture

- `llm_factory.py` - Main factory for creating clients
- `text_clients.py` - Text generation implementations
- `vision_clients.py` - Vision processing implementations  
- `embedding_client.py` - Embedding generation
- `generation_service.py` - Stateless generation service
- `model_manager.py` - Model discovery and management
- `thinking_tokens_service.py` - Advanced reasoning with thinking tokens
- `llm_types.py` - Core interfaces and types