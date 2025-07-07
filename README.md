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
├── pdf_to_text/             # 📄 Visual content processing (PDFs, images) with bounding boxes
├── chunking/                # ✂️ Text chunking and semantic processing
├── tools/                   # 🔧 Universal tool service + MCP integration
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
4. **🔧 Universal Tool Service** - Register tools once, use with any LLM provider (OpenAI, Anthropic, Gemini, Ollama)
5. **🛠️ MCP Integration** - Full Model Context Protocol support with secure credential management
6. **🎵 Audio Processing** - Text-to-Speech and Speech-to-Text with streaming LLM → TTS pipelines
7. **🎨 Image Generation** - Multi-provider support (OpenAI, Stability AI, Google Imagen, local models)
8. **🤖 JSON-Driven Agents** - Create sophisticated AI workflows through configuration
9. **🧠 Intelligent Memory** - Vector similarity search for project knowledge storage
10. **💬 Rich Conversations** - Full conversation state tracking with search and export
11. **📄 Visual Content Processing** - Extract text from PDFs, images, and base64 data with precise bounding boxes
12. **🎯 Prompt Management** - Externalized, versioned prompts with evaluation and A/B testing
13. **📊 Concept Evaluation** - LLM-as-judge framework for testing prompt quality

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

### Visual Content Processing

**Basic PDF and Image Text Extraction**
```python
from pdf_to_text.visual_to_text_service import VisualToTextService, VisualExtractOptions

# Create service
service = VisualToTextService()

# Basic text extraction from any visual content
result = service.extract_text_from_file("document.pdf")
print(f"Extracted text: {result.text}")
print(f"Source type: {result.source_type}")  # "pdf", "image", or "base64_image"
print(f"Page count: {result.page_count}")
```

**Advanced Processing with Bounding Boxes**
```python
# Configure for precise text location extraction
options = VisualExtractOptions(
    include_bounding_boxes=True,
    vision_prompt="Extract text and provide precise coordinate locations",
    extract_tables=True,
    preserve_layout=True
)

result = service.extract_text_from_file("invoice.pdf", options)

# Access bounding box data
if result.bounding_boxes:
    for bbox in result.bounding_boxes:
        print(f"Text location: {bbox}")
```

**Base64 Image Processing**
```python
# Process base64 encoded images directly
base64_result = service.extract_text_from_base64_image(
    base64_image_string,
    options=VisualExtractOptions(include_bounding_boxes=True)
)
```

**PDF to Images Conversion**
```python
# Convert PDF pages to base64 images for further processing
images = service.convert_pdf_to_base64_images("document.pdf", dpi=150)
print(f"Generated {len(images)} images from PDF")

# Process each page separately
for i, image_b64 in enumerate(images):
    page_result = service.extract_text_from_base64_image(image_b64)
    print(f"Page {i+1} text: {page_result.text}")
```

**Specialized Use Cases**
```python
from pdf_to_text.visual_to_text_service import (
    extract_text_from_visual,
    extract_with_bounding_boxes,
    extract_tables_from_visual,
    convert_pdf_to_images
)

# Quick text extraction
text = extract_text_from_visual("screenshot.png")

# Extract with precise positioning
result = extract_with_bounding_boxes("invoice.pdf")

# Focus on table data
table_result = extract_tables_from_visual("financial_report.pdf")

# Convert for batch processing
pdf_images = convert_pdf_to_images("presentation.pdf", dpi=200)
```

**Zoom Invoice Subscription Date Extraction Example**
```python
# Specialized configuration for invoice processing
invoice_options = VisualExtractOptions(
    include_bounding_boxes=True,
    vision_prompt="""
    Extract subscription dates from this invoice. 
    Look for subscription periods, billing cycles, or service dates.
    Provide bounding box coordinates for each date found.
    """,
    extract_tables=True
)

# Process Zoom invoice
result = service.extract_text_from_file("zoom_invoice.pdf", invoice_options)

# Parse subscription dates with regex patterns
import re
subscription_pattern = r'(\w{3}\s+\d{1,2},?\s+\d{4})\s*-\s*(\w{3}\s+\d{1,2},?\s+\d{4})'
matches = re.findall(subscription_pattern, result.text)

for start_date, end_date in matches:
    print(f"Subscription period: {start_date} to {end_date}")
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

## 🔧 Universal Tool Service

AI Lego Bricks includes a powerful **Universal Tool Service** that allows you to register tools once and use them with any LLM provider. Tools are defined with a universal schema and automatically adapted for OpenAI, Anthropic, Google Gemini, and Ollama.

### Key Features

- **🔄 Provider Abstraction**: Define tools once, use with any LLM provider
- **🔐 Secure Credential Management**: Integrated with CredentialManager for API keys
- **⚡ Async Execution**: Concurrent tool execution for performance
- **🏗️ Workflow Integration**: Seamless integration with agent orchestration
- **🛡️ Early Validation**: Credential validation at registration time

### Quick Start

**1. Register a Tool**
```python
from tools import ToolSchema, ToolParameter, ParameterType, Tool, ToolExecutor
from tools import register_tool_globally

class WeatherExecutor(ToolExecutor):
    async def execute(self, tool_call):
        location = tool_call.parameters.get("location")
        # Your weather API logic here
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result={"weather": "sunny", "temp": "22°C", "location": location}
        )

# Define tool schema
weather_schema = ToolSchema(
    name="get_weather",
    description="Get current weather for a location",
    parameters=ToolParameter(
        type=ParameterType.OBJECT,
        properties={
            "location": ToolParameter(
                type=ParameterType.STRING,
                description="City name"
            )
        },
        required=["location"]
    )
)

# Create and register tool
weather_tool = Tool(schema=weather_schema, executor=WeatherExecutor())
await register_tool_globally(weather_tool, category="utilities")
```

**2. Use in Workflows**
```json
{
  "id": "weather_assistant",
  "type": "tool_call",
  "config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "tools": ["get_weather"],
    "tool_choice": "auto",
    "auto_execute": true,
    "prompt": "You are a weather assistant with access to weather tools."
  },
  "inputs": {
    "message": "What's the weather in London?"
  }
}
```

### Secure Tools with API Keys

**Create Secure API Tool**
```python
from tools import APIToolExecutor, SecureToolExecutor
from credentials import CredentialManager

class GitHubTool(APIToolExecutor):
    def __init__(self, credential_manager=None):
        super().__init__(
            base_url="https://api.github.com",
            api_key_name="GITHUB_TOKEN",
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call):
        repo = tool_call.parameters.get("repo")
        response = await self.make_api_request(f"repos/{repo}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result={
                "repo": repo,
                "stars": response["stargazers_count"],
                "description": response["description"]
            }
        )
```

**Use with Credential Management**
```python
# Environment-based (production)
creds = CredentialManager(load_env=True)
tool_service = ToolService(credential_manager=creds)

# Explicit credentials (library/multi-tenant)
creds = CredentialManager({
    "GITHUB_TOKEN": user_provided_token,
    "OPENAI_API_KEY": user_api_key
}, load_env=False)

# Validate credentials
validation = await tool_service.validate_tool_credentials()
print(f"Available tools: {validation['available_tools']}")
```

### Provider Support

| Provider | Tool Format | Choice Options | Status |
|----------|-------------|----------------|---------|
| **OpenAI** | `tools` parameter | auto, none, specific | ✅ Full Support |
| **Anthropic** | `tools` parameter | auto, any, none, specific | ✅ Full Support |
| **Google Gemini** | `functionDeclarations` | AUTO, ANY, NONE | ✅ Full Support |
| **Ollama** | `tools` parameter | auto, none, specific | ✅ Full Support |

### Complete Documentation

See **[TOOLS_README.md](TOOLS_README.md)** for comprehensive documentation including:
- Secure credential patterns
- Provider-specific behavior
- Error handling best practices
- Performance considerations
- Testing with mock credentials

## 🛠️ MCP (Model Context Protocol) Integration

AI Lego Bricks includes full **Model Context Protocol** support, allowing you to integrate external MCP servers and use their tools seamlessly in agent workflows.

### Key Features

- **🔄 Server Management**: Automatic MCP server lifecycle management with process monitoring
- **🔍 Tool Discovery**: Automatic discovery and conversion of MCP tools to universal format
- **🔐 Secure Credentials**: Integration with CredentialManager for secure API key handling
- **🌐 Multi-Provider**: MCP tools work with all LLM providers (OpenAI, Anthropic, Gemini, Ollama)
- **⚡ Agent Integration**: MCP tools automatically available in JSON-driven workflows

### Quick Start

**1. Install MCP Servers**
```bash
# Install common MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-github
```

**2. Configure with Credentials**
```json
{
  "servers": {
    "github": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env_credentials": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"
      },
      "required_credentials": ["GITHUB_TOKEN"]
    },
    "filesystem": {
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
      "args": ["/allowed/directory"]
    }
  }
}
```

**3. Initialize and Use**
```python
from tools import initialize_mcp_servers_from_config, register_mcp_tools_globally
from credentials import CredentialManager

# Setup credentials
creds = CredentialManager()  # Loads from .env

# Initialize MCP servers with secure credential handling
await initialize_mcp_servers_from_config(credential_manager=creds)
await register_mcp_tools_globally()

# Tools now available in agent workflows
```

**4. Use in Agent Workflows**
```json
{
  "id": "github_agent",
  "type": "tool_call",
  "config": {
    "provider": "ollama",
    "tools": ["mcp_github_get_repository"],
    "tool_choice": "auto"
  },
  "inputs": {
    "message": "Get information about the pytorch/pytorch repository"
  }
}
```

### Available MCP Servers

- **@modelcontextprotocol/server-filesystem**: File system operations
- **@modelcontextprotocol/server-git**: Git repository operations  
- **@modelcontextprotocol/server-brave-search**: Web search via Brave API
- **@modelcontextprotocol/server-github**: GitHub API operations
- **@modelcontextprotocol/server-postgres**: PostgreSQL database operations
- **@modelcontextprotocol/server-sqlite**: SQLite database operations
- **@modelcontextprotocol/server-puppeteer**: Web browser automation
- **@modelcontextprotocol/server-memory**: Persistent memory/knowledge

### Security & Credentials

MCP integration follows the same secure credential patterns:

- **No hardcoded secrets** in configuration files
- **Credential validation** before server startup  
- **Environment injection** at runtime
- **CredentialManager integration** for multi-tenant support

### Complete Documentation

See **[tools/examples/README_MCP.md](tools/examples/README_MCP.md)** for comprehensive MCP documentation including:
- Configuration examples
- Credential management patterns
- Custom server integration
- Troubleshooting guide

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
- **`tool_call`** - 🆕 Call external tools/APIs with automatic credential management
- **`tts`** - Convert text to speech with multiple provider support
- **`stt`** - Convert speech to text with word timestamps and speaker detection

#### Visual Content Processing
- **`document_processing`** - Extract text from PDFs, images, and base64 data with bounding boxes
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

#### Invoice Processing Agent
Extracts structured data from invoices:
```
PDF/Image Input → Visual Text Extraction → 
Date/Amount Parsing → Bounding Box Mapping → Structured Output
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

**Visual Content Processing**
```json
{
  "id": "extract_invoice_data",
  "type": "document_processing",
  "config": {
    "include_bounding_boxes": true,
    "vision_prompt": "Extract subscription dates and billing amounts with coordinates",
    "extract_tables": true
  },
  "inputs": {
    "file_path": "invoice.pdf"
  }
}
```

**Base64 Image Processing**
```json
{
  "id": "process_screenshot",
  "type": "document_processing",
  "config": {
    "include_bounding_boxes": true,
    "vision_prompt": "Extract text from this screenshot"
  },
  "inputs": {
    "base64_image": "$image_data"
  }
}
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