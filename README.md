# 🧱 AI Lego Bricks

A modular LLM agent system providing building blocks for intelligent AI workflows with advanced memory capabilities.

## 📦 Installation

### Install as a Library

```bash
# Install from PyPI (when published)
pip install ai-lego-bricks

# Or install from source
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks
pip install -e .
```

### Using the CLI

After installation, you'll have access to the `ailego` command:

```bash
# Initialize a new project
ailego init my-ai-project

# Navigate to your project
cd my-ai-project

# Verify your setup
ailego verify

# Run an agent
ailego run agents/simple_chat.json

# Create new agents
ailego create chat --name "customer-support"
```

### Quick Start with CLI

1. **Create a new project:**
   ```bash
   ailego init my-project --template advanced
   cd my-project
   ```

2. **Configure environment:**
   ```bash
   # Edit .env with your API keys
   cp .env.example .env
   ```

3. **Verify setup:**
   ```bash
   ailego verify --verbose
   ```

4. **Run your first agent:**
   ```bash
   ailego run agents/simple_chat.json
   ```

## 🚀 Developer Setup

**Contributing to this project?** Start with the comprehensive setup guide:

```bash
cd setup
cat README.md
```

Or view the setup files directly:
- **[setup/README.md](setup/README.md)** - Complete setup instructions
- **[setup/SUPABASE_SETUP.md](setup/SUPABASE_SETUP.md)** - Detailed Supabase configuration

## 📁 Project Structure

```
├── setup/                    # 🔧 Setup and configuration
│   ├── README.md            # Complete setup guide
│   ├── SUPABASE_SETUP.md    # Supabase configuration
│   ├── setup_supabase.py    # Supabase verification script
│   └── setup_supabase_pgvector.sql  # Database schema
├── credentials/              # 🔐 Secure credential management
├── memory/                   # 🧠 Memory service implementations
├── llm/                      # 🧠 LLM services (generation + conversation)
├── chat/                     # 💬 Enhanced conversation management
├── prompt/                   # 🎯 Prompt management and evaluation
├── tts/                      # 🎵 Text-to-speech with streaming support
├── pdf_to_text/             # 📄 PDF processing and text extraction
├── chunking/                # ✂️ Text chunking and semantic processing
├── agent_orchestration/      # 🤖 JSON-driven agent workflows
├── examples/                # 📋 Usage examples and demos
├── test/                     # 🧪 Test utilities
├── claude-knowledge/         # 🤖 Claude-specific documentation
├── .env.example             # 📝 Environment template
├── CREDENTIAL_MIGRATION_GUIDE.md  # 📖 Migration guide
└── requirements.txt         # 📦 Python dependencies
```

## 🎯 What This Does

This project provides:

1. **🔐 Secure Credential Management** - Library-safe credential isolation with environment fallback
2. **Clean LLM Architecture** - Separated Generation (one-shot) and Conversation (multi-turn) services
3. **Streaming Support** - Real-time LLM response streaming with native Ollama and Anthropic support
4. **Text-to-Speech Integration** - Convert text to audio with streaming LLM → TTS pipelines
5. **JSON-Driven Agent Orchestration** - Create sophisticated AI workflows through configuration
6. **Intelligent Memory System** - Store and retrieve project knowledge using vector similarity search
7. **Rich Conversation Management** - Full conversation state tracking with search and export
8. **Structured LLM Responses** - Type-safe, validated outputs using Pydantic schemas
9. **Multi-Modal Processing** - Text, vision, and document analysis capabilities
10. **Prompt Management** - Externalized, versioned prompts with evaluation and A/B testing

## 🏃‍♂️ Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Setup Database**
   ```bash
   cd setup
   python setup_supabase.py  # Verify Supabase setup
   ```

4. **Test Memory Service**
   ```bash
   python memory/memory_example.py
   ```

## 🛠️ Configuration

### 🔐 Credential Management

AI Lego Bricks uses a **secure credential management system** that supports both environment variables and explicit credential injection:

```python
from credentials import CredentialManager

# Option 1: Environment variables (default, backward compatible)
from llm import create_text_client
client = create_text_client("gemini")  # Uses .env file

# Option 2: Explicit credentials (library-safe)
creds = CredentialManager({"GOOGLE_AI_STUDIO_KEY": "your-key"}, load_env=False)
client = create_text_client("gemini", credential_manager=creds)
```

**Key Benefits:**
- ✅ **Library Safe**: No unwanted .env loading when used as dependency
- ✅ **Credential Isolation**: Different services can have different credentials
- ✅ **Backward Compatible**: Existing code continues to work
- ✅ **Multi-Tenant Ready**: Support for tenant-specific credentials

For complete migration guide, see **[CREDENTIAL_MIGRATION_GUIDE.md](CREDENTIAL_MIGRATION_GUIDE.md)**

### 📋 Environment Setup

See **[setup/README.md](setup/README.md)** for detailed configuration instructions covering:

- Supabase setup with pgvector for memory storage
- Ollama local LLM configuration (with streaming support)
- Google AI Studio (Gemini) integration
- Text-to-Speech provider setup (OpenAI, Google, Coqui-XTTS)
- Neo4j graph database setup (optional)

## 📚 Usage Examples

### Memory Service
```python
from memory import create_memory_service
from credentials import CredentialManager

# Option 1: Environment variables (default)
memory = create_memory_service("auto")

# Option 2: Explicit credentials (library-safe)
creds = CredentialManager({
    "SUPABASE_URL": "your-url",
    "SUPABASE_ANON_KEY": "your-key"
}, load_env=False)
memory = create_memory_service("supabase", credential_manager=creds)

# Store project knowledge
memory.store_memory(
    "React components should be functional with hooks",
    {"project": "web-app", "category": "frontend"}
)

# Retrieve relevant memories
results = memory.retrieve_memories("React best practices")
```

### LLM Services

**Generation Service (One-Shot)**
```python
from llm.generation_service import quick_generate_gemini

# Fast, stateless generation
response = quick_generate_gemini("Analyze this document")
```

**Streaming Generation**
```python
from llm.generation_service import quick_generate_ollama_stream

# Real-time streaming responses
for chunk in quick_generate_ollama_stream("Tell me about AI"):
    print(chunk, end='', flush=True)
```

**Conversation Service (Multi-Turn)**
```python
from chat.conversation_service import create_gemini_conversation

# Rich conversation with full state management
conv = create_gemini_conversation()
conv.add_system_message("You are a helpful assistant")

response1 = conv.send_message("What is Python?")
response2 = conv.send_message("How do I use it for web development?")

# Rich conversation access
first_prompt = conv.get_first_prompt()
summary = conv.get_conversation_summary()
```

### Text-to-Speech

**Basic TTS**
```python
from tts import create_tts_service
from credentials import CredentialManager

# Option 1: Environment variables (auto-detects providers)
tts = create_tts_service("auto")

# Option 2: Explicit credentials for OpenAI TTS
creds = CredentialManager({"OPENAI_API_KEY": "your-key"}, load_env=False)
tts = create_tts_service("openai", credential_manager=creds)

# Generate speech
response = tts.text_to_speech(
    text="Hello, this is a test of the TTS system!",
    output_path="output/speech.wav"
)
```

**Streaming LLM to TTS**
```python
from tts.streaming_tts_service import create_streaming_pipeline

# Create streaming pipeline
pipeline = create_streaming_pipeline(
    llm_provider="ollama",
    tts_provider="auto"
)

# Stream LLM response directly to audio files
for progress in pipeline.stream_chat_to_audio("Explain quantum computing"):
    print(f"Status: {progress['status']}, Audio files: {progress['audio_files_generated']}")
```

### Prompt Management
```python
from prompt import create_prompt_service, PromptStatus

# Create managed prompts for reusability and evaluation
prompt_service = create_prompt_service("auto")

# Create a template-based prompt
prompt = prompt_service.create_prompt(
    prompt_id="helpful_assistant",
    name="Helpful Assistant",
    content=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": {
            "template": "Answer this question: {{ user_question }}",
            "required_variables": ["user_question"]
        }}
    ],
    version="1.0.0",
    status=PromptStatus.ACTIVE
)

# Use in workflows for easy iteration and A/B testing
```

## 🤖 Creating Agents

This project uses a **JSON-driven agent orchestration system** that lets you create sophisticated AI agents by combining building blocks through configuration rather than code.

### Quick Agent Creation

**1. Basic Chat Agent**
```json
{
  "name": "simple_chat_agent",
  "description": "A basic conversational agent",
  "config": {
    "default_llm_provider": "gemini"
  },
  "steps": [
    {
      "id": "get_input",
      "type": "input",
      "config": {"prompt": "What can I help you with?"},
      "outputs": ["user_query"]
    },
    {
      "id": "generate_response",
      "type": "llm_chat",
      "inputs": {
        "message": {"from_step": "get_input", "field": "user_query"}
      },
      "outputs": ["response"]
    },
    {
      "id": "output",
      "type": "output",
      "inputs": {
        "result": {"from_step": "generate_response", "field": "response"}
      }
    }
  ]
}
```

**2. Execute Your Agent**
```python
from agent_orchestration import AgentOrchestrator

orchestrator = AgentOrchestrator()
workflow = orchestrator.load_workflow_from_file("my_agent.json")
result = orchestrator.execute_workflow(workflow, {"user_query": "Hello!"})
```

### Available Building Blocks

#### Core Operations
- **`input`** - Collect user input or external data
- **`output`** - Format and return results  
- **`llm_chat`** - Generate text using LLM (auto-selects Generation/Conversation service)
- **`llm_vision`** - Analyze images with vision models
- **`llm_structured`** - Generate type-safe, validated JSON responses
- **`tts`** - Convert text to speech with multiple provider support

#### Document Processing
- **`document_processing`** - Extract and enhance text from PDFs
- **`chunk_text`** - Break text into semantic chunks

#### Memory Operations  
- **`memory_store`** - Store content in vector/graph storage
- **`memory_retrieve`** - Search and retrieve relevant memories

#### Control Flow
- **`condition`** - Conditional execution and branching
- **`loop`** - Iterate over collections

### Common Agent Patterns

#### Document Analysis Agent
Processes PDFs, stores in memory, answers questions:
```
Document → Chunking → Memory Storage → 
User Question → Memory Search → LLM Response
```

#### Research Agent  
Multi-document analysis with synthesis:
```
Multiple Documents → Concept Extraction → 
Memory Storage → Research Query → Synthesis → Report
```

#### Multi-Modal Agent
Combines vision and text processing:
```
Image Analysis → Text Generation → 
Conditional Processing → Structured Output
```

### 🚀 Key Architecture Features

**Generation vs Conversation Services**
```json
{
  "id": "document_analysis",
  "type": "llm_chat",
  "config": {
    "provider": "gemini",
    "use_conversation": false,  // Uses fast Generation service
    "system_message": "You are a document analyzer"
  }
}
```

```json
{
  "id": "interactive_chat", 
  "type": "llm_chat",
  "config": {
    "provider": "gemini",
    "use_conversation": true,   // Uses stateful Conversation service
    "conversation_id": "user_session_123"
  }
}
```

**Streaming Support**
```json
{
  "id": "streaming_response",
  "type": "llm_chat",
  "config": {
    "provider": "ollama",      // Best streaming support
    "stream": true,            // Enable streaming
    "use_conversation": false
  }
}
```

**Text-to-Speech Integration**
```json
{
  "id": "convert_to_speech",
  "type": "tts",
  "config": {
    "provider": "auto",        // Auto-detect available TTS
    "voice": "default",
    "output_path": "output/response.wav"
  },
  "inputs": {
    "text": {"from_step": "streaming_response", "field": "response"}
  }
}
```

**Rich Conversation Access**
Agents can reference any part of conversation history:
```json
{
  "id": "conversation_summary",
  "inputs": {
    "message": "Summary: {conversation_summary}",
    "first_question": "{first_prompt}",
    "last_ai_response": "{last_response}"
  }
}
```

### Advanced Features

**Conditional Logic**
```json
{
  "id": "smart_routing",
  "type": "condition", 
  "condition": {
    "field": "document_type",
    "operator": "==", 
    "value": "technical_manual"
  },
  "routes": {
    "true": "technical_processing",
    "false": "general_processing"
  }
}
```

**Structured Responses**
```json
{
  "id": "extract_data",
  "type": "llm_structured",
  "config": {
    "response_schema": {
      "name": "DocumentAnalysis",
      "fields": {
        "summary": {"type": "string"},
        "key_points": {"type": "list"},
        "confidence": {"type": "float"}
      }
    }
  }
}
```

**Multi-Model Workflows**
```json
{
  "id": "vision_analysis", 
  "type": "llm_vision",
  "config": {"provider": "gemini"}
},
{
  "id": "text_summary",
  "type": "llm_chat", 
  "config": {"provider": "anthropic"}
}
```

### Getting Started with Agents

1. **Study Examples**: Check `agent_orchestration/examples/` for templates
2. **Start Simple**: Begin with basic input → LLM → output flows  
3. **Add Memory**: Include storage and retrieval for context
4. **Use Conditions**: Add branching logic for complex decisions
5. **Structure Output**: Define schemas for reliable data extraction

For detailed documentation, see **[agent_orchestration/README.md](agent_orchestration/README.md)**

## 🖥️ CLI Reference

The `ailego` command-line tool provides a complete interface for managing AI agent projects:

### Project Management
```bash
ailego init <project-name>              # Initialize new project
ailego init <name> --template advanced  # Use specific template
ailego verify                           # Verify setup
ailego verify --verbose                 # Detailed verification
ailego status                          # Show system status
```

### Agent Management
```bash
ailego create <type>                    # Create new agent interactively
ailego create chat --name "support"    # Create named chat agent
ailego list-templates                   # Show available templates
```

### Workflow Execution
```bash
ailego run <workflow.json>              # Execute agent workflow
ailego run <workflow.json> --verbose    # Detailed execution logs
ailego run <workflow.json> --output results.json  # Save results
```

### Available Agent Types
- `chat` - Basic conversational agents
- `document-analysis` - PDF and document processing
- `research` - Multi-source research and synthesis
- `vision` - Image analysis and description
- `streaming` - Real-time conversation with TTS

### Available Project Templates
- `basic` - Simple chat and text processing
- `advanced` - Full-featured with memory and multi-modal
- `research` - Document analysis and research workflows

### Examples
```bash
# Complete workflow
ailego init research-project --template research
cd research-project
ailego verify
ailego create document-analysis --name "pdf-analyzer"
ailego run agents/pdf-analyzer.json

# Quick prototyping
ailego init demo --template basic
cd demo
ailego create chat --name "assistant"
ailego run agents/assistant.json
```

## 🤝 Contributing

This project follows a learning-focused development approach. Each implementation should be:

- **Educational** - Help users understand the concepts
- **Well-documented** - Clear explanations and examples
- **Modular** - Easy to extend and modify

## 📄 License

MIT License - see the project files for details.

---

**Ready to get started?** Head to **[setup/README.md](setup/README.md)** for complete setup instructions! 🚀