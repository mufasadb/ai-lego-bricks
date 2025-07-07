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
├── stt/                      # 🎤 Speech-to-text with multiple providers
├── image_generation/         # 🎨 Image generation with multiple providers
├── pdf_to_text/             # 📄 PDF processing and text extraction
├── chunking/                # ✂️ Text chunking and semantic processing
├── agent_orchestration/      # 🤖 JSON-driven agent workflows
├── examples/                # 📋 Usage examples and demos
├── claude-knowledge/         # 🤖 Claude-specific documentation
├── .env.example             # 📝 Environment template
├── CREDENTIAL_MIGRATION_GUIDE.md  # 📖 Migration guide
└── requirements.txt         # 📦 Python dependencies
```

## 🎯 What This Does

This project provides:

1. **🔐 Secure Credential Management** - Library-safe credential isolation with environment fallback
2. **🧠 LLM Services** - Generation (one-shot) and Conversation (multi-turn) with streaming support
3. **🌊 Real-time Streaming** - Native streaming for Ollama, Anthropic; simulated for others
4. **🎵 Audio Processing** - Text-to-Speech and Speech-to-Text with streaming LLM → TTS pipelines
5. **🎨 Image Generation** - Multi-provider support (OpenAI, Stability AI, Google Imagen, local models)
6. **🤖 JSON-Driven Agents** - Create sophisticated AI workflows through configuration
7. **🧠 Intelligent Memory** - Vector similarity search for project knowledge storage
8. **💬 Rich Conversations** - Full conversation state tracking with search and export
9. **📄 Multi-Modal Processing** - Text, vision, and document analysis capabilities
10. **🎯 Prompt Management** - Externalized, versioned prompts with evaluation and A/B testing
11. **📊 Concept Evaluation** - LLM-as-judge framework for testing prompt quality

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
- Speech-to-Text provider setup (Faster Whisper, Google Cloud Speech)
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

### Streaming LLM to TTS Pipeline
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

### Speech-to-Text

**Basic STT**
```python
from stt import create_stt_service
from credentials import CredentialManager

# Option 1: Environment variables (auto-detects providers)
stt = create_stt_service("auto")

# Option 2: Explicit credentials for Google Speech
creds = CredentialManager({"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}, load_env=False)
stt = create_stt_service("google", credential_manager=creds)

# Transcribe audio
response = stt.speech_to_text("path/to/audio.wav")
print(f"Transcript: {response.transcript}")
print(f"Confidence: {response.confidence}")
print(f"Language: {response.language_detected}")
```

**Voice Assistant Workflow (STT → LLM → TTS)**
```python
from stt import create_stt_service
from llm.generation_service import quick_generate_gemini
from tts import create_tts_service

# Complete voice interaction pipeline
stt = create_stt_service("faster_whisper")
tts = create_tts_service("auto")

# 1. Convert voice to text
voice_input = stt.speech_to_text("user_question.wav")
user_text = voice_input.transcript

# 2. Process with LLM
ai_response = quick_generate_gemini(f"Answer this question: {user_text}")

# 3. Convert response to speech
voice_output = tts.text_to_speech(ai_response, output_path="ai_response.wav")
```

### Image Generation

**Basic Image Generation**
```python
from image_generation import create_image_generation_service
from credentials import CredentialManager

# Option 1: Environment variables (auto-detects providers)
image_service = create_image_generation_service("auto")

# Option 2: Explicit credentials for OpenAI DALL-E
creds = CredentialManager({"OPENAI_API_KEY": "your-key"}, load_env=False)
image_service = create_image_generation_service("openai", credential_manager=creds)

# Generate images
response = image_service.generate_image(
    prompt="A serene mountain landscape at sunset with a crystal clear lake",
    size="1024x1024",
    quality="hd",
    num_images=2
)

if response.success:
    print(f"Generated images: {response.images}")
else:
    print(f"Error: {response.error_message}")
```

**Quick Image Generation**
```python
from image_generation import quick_image_generation

# One-line image generation
image_path = quick_image_generation(
    "A futuristic cityscape with flying cars and neon lights",
    provider="auto",
    size="1024x1024"
)
print(f"Image saved to: {image_path}")
```

**Batch Image Generation**
```python
from image_generation import create_image_generation_service

service = create_image_generation_service("auto")

# Generate variations
base_prompt = "A majestic dragon"
variations = ["in a medieval castle", "flying over a modern city", "breathing colorful flames"]

responses = service.generate_variations(base_prompt, variations)
for i, response in enumerate(responses):
    if response.success:
        print(f"Variation {i+1}: {response.images[0]}")
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

# Render with context
rendered = prompt_service.render_prompt(
    "helpful_assistant", 
    context={"user_question": "What is machine learning?"}
)

# A/B test versions
from prompt.evaluation_service import EvaluationService
eval_service = EvaluationService(prompt_service.storage)
comparison = eval_service.compare_prompt_versions(
    "helpful_assistant", "1.0.0", "helpful_assistant", "1.1.0"
)
```

### Concept-Based Evaluation
```python
from prompt.eval_builder import EvaluationBuilder
from prompt.concept_evaluation_service import ConceptEvaluationService

# Create evaluation with concept checks
builder = EvaluationBuilder("Document Summary Eval")
builder.with_prompt_template("Summarize this {{doc_type}}: {{content}}")

# Add concept verification
builder.add_concept_check(
    check_type="must_contain",
    description="Contains key facts",
    concept="specific facts and data from the source",
    check_id="facts"
)

builder.add_concept_check(
    check_type="must_not_contain",
    description="Avoids opinions", 
    concept="personal opinions or subjective judgments",
    check_id="objective"
)

# Add test cases
builder.add_test_case(
    context={"doc_type": "report", "content": "Revenue grew 15% to $2.3B..."},
    concept_check_refs=["facts", "objective"]
)

# Run evaluation
eval_def = builder.build("summary_eval")
service = ConceptEvaluationService(storage)
results = service.run_evaluation(eval_def)
print(f"Score: {results.overall_score:.1%}")
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
- **`stt`** - Convert speech to text with word timestamps and speaker detection

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

**Multi-Modal Voice Assistant**
```json
[
  {
    "id": "transcribe",
    "type": "stt",
    "config": {"provider": "faster_whisper"}
  },
  {
    "id": "process",
    "type": "llm_chat",
    "config": {"stream": true, "provider": "ollama"},
    "inputs": {
      "message": {"from_step": "transcribe", "field": "transcript"}
    }
  },
  {
    "id": "respond",
    "type": "tts",
    "inputs": {
      "text": {"from_step": "process", "field": "response"}
    }
  }
]
```

**Prompt Management Integration**
```json
{
  "id": "managed_prompt",
  "type": "llm_chat",
  "prompt_ref": {
    "prompt_id": "helpful_assistant",
    "version": "1.0.0",
    "context_variables": {"user_question": "$user_input"}
  },
  "config": {"provider": "gemini"}
}
```

### Advanced Features

**Conditional Logic & Structured Responses**
```json
{
  "id": "smart_routing",
  "type": "condition", 
  "condition": {"field": "document_type", "operator": "==", "value": "technical_manual"},
  "routes": {"true": "technical_processing", "false": "general_processing"}
}
```

**Concept-Based Evaluation**
```json
{
  "id": "evaluate_response",
  "type": "concept_evaluation",
  "config": {"eval_id": "my_evaluation", "llm_provider": "gemini"},
  "inputs": {"context_variables": {"doc_type": "report", "content": "$document_text"}}
}
```

### Getting Started with Agents

1. **Study Examples**: Check `agent_orchestration/examples/` for templates
2. **Start Simple**: Begin with basic input → LLM → output flows  
3. **Add Memory**: Include storage and retrieval for context
4. **Use Streaming**: Enable real-time responses with Ollama
5. **Structure Output**: Define schemas for reliable data extraction

For detailed documentation, see **[agent_orchestration/README.md](agent_orchestration/README.md)**

## 🖥️ CLI Reference

The `ailego` command-line tool provides a complete interface for managing AI agent projects:

### Project Management
```bash
ailego init <project-name>              # Initialize new project
ailego init <name> --template advanced  # Use specific template  
ailego verify                           # Verify setup
ailego run <workflow.json>              # Execute agent workflow
```

### Agent Types & Templates
- **Agent Types**: `chat`, `document-analysis`, `research`, `vision`, `streaming`, `voice`
- **Templates**: `basic`, `advanced`, `research`

### Quick Start
```bash
# Complete workflow
ailego init research-project --template research
cd research-project
ailego verify
ailego create document-analysis --name "pdf-analyzer"
ailego run agents/pdf-analyzer.json
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