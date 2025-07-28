"""
AI Lego Bricks - Modular Building Blocks for LLM Agents

A modular library of building blocks for LLM agentic work, designed to be
combined and configured like building blocks. The core philosophy is
JSON-driven configuration over Python code.

## Quick Start

```python
# Basic imports
from ai_lego_bricks.chat import create_chat_service
from ai_lego_bricks.agent_orchestration import AgentOrchestrator

# Create services
chat = create_chat_service("gemini")
orchestrator = AgentOrchestrator()

# Run agent workflows
result = orchestrator.execute_workflow_from_file("workflow.json")
```

## Available Bricks

- **Agent Orchestrator**: JSON-driven workflow orchestration
- **Chat Services**: Multi-provider LLM chat interfaces
- **Memory Services**: Semantic memory with vector search
- **Tools & MCP**: External tool execution and MCP integration
- **Text-to-Speech**: Multi-provider TTS with streaming
- **Speech-to-Text**: Voice input processing
- **PDF Processing**: Document extraction with AI vision
- **Text Chunking**: Intelligent text segmentation
- **Image Generation**: AI-powered image creation
- **Prompt Management**: Versioned prompt templates
- **Credential Management**: Secure API key storage
- **Visualizer**: Workflow diagram generation

Each brick can be used independently or combined through the agent orchestrator.
"""

__version__ = "0.1.0"
__author__ = "Daniel Beach"
__email__ = "callmebeachy@gmail.com"

# Core brick imports for easy access
try:
    from .agent_orchestration import AgentOrchestrator
except ImportError:
    AgentOrchestrator = None

try:
    from .chat import create_chat_service, create_conversation
except ImportError:
    create_chat_service = None
    create_conversation = None

try:
    from .memory import create_memory_service
except ImportError:
    create_memory_service = None

try:
    from .llm import create_text_client, quick_generate_gemini
except ImportError:
    create_text_client = None
    quick_generate_gemini = None

try:
    from .tts import create_tts_service
except ImportError:
    create_tts_service = None

try:
    from .stt import create_stt_service
except ImportError:
    create_stt_service = None

try:
    from .pdf_to_text import extract_text_from_pdf
except ImportError:
    extract_text_from_pdf = None

try:
    from .chunking import create_chunking_service
except ImportError:
    create_chunking_service = None

try:
    from .tools import get_tool_service
except ImportError:
    get_tool_service = None

try:
    from .credentials import CredentialManager
except ImportError:
    CredentialManager = None

# Define what's available when using "from ai_lego_bricks import *"
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "AgentOrchestrator",
    "create_chat_service",
    "create_conversation",
    "create_memory_service",
    "create_text_client",
    "quick_generate_gemini",
    "create_tts_service",
    "create_stt_service",
    "extract_text_from_pdf",
    "create_chunking_service",
    "get_tool_service",
    "CredentialManager",
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]
