# 🧱 AI Lego Bricks

A modular LLM agent system providing building blocks for intelligent AI workflows with advanced memory capabilities.

## 🚀 Quick Start

**New to this project?** Start with the comprehensive setup guide:

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
├── memory/                   # 🧠 Memory service implementations
├── llm/                      # 🧠 LLM services (generation + conversation)
├── chat/                     # 💬 Enhanced conversation management
├── agent_orchestration/      # 🤖 JSON-driven agent workflows
├── test/                     # 🧪 Test utilities
├── claude-knowledge/         # 🤖 Claude-specific documentation
├── .env.example             # 📝 Environment template
└── requirements.txt         # 📦 Python dependencies
```

## 🎯 What This Does

This project provides:

1. **Clean LLM Architecture** - Separated Generation (one-shot) and Conversation (multi-turn) services
2. **JSON-Driven Agent Orchestration** - Create sophisticated AI workflows through configuration
3. **Intelligent Memory System** - Store and retrieve project knowledge using vector similarity search
4. **Rich Conversation Management** - Full conversation state tracking with search and export
5. **Structured LLM Responses** - Type-safe, validated outputs using Pydantic schemas
6. **Multi-Modal Processing** - Text, vision, and document analysis capabilities

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

See **[setup/README.md](setup/README.md)** for detailed configuration instructions covering:

- Supabase setup with pgvector for memory storage
- Ollama local LLM configuration  
- Google AI Studio (Gemini) integration
- Neo4j graph database setup (optional)

## 📚 Usage Examples

### Memory Service
```python
from memory import create_memory_service

# Auto-detects available services (Supabase/Neo4j)
memory = create_memory_service("auto")

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

## 🤝 Contributing

This project follows a learning-focused development approach. Each implementation should be:

- **Educational** - Help users understand the concepts
- **Well-documented** - Clear explanations and examples
- **Modular** - Easy to extend and modify

## 📄 License

MIT License - see the project files for details.

---

**Ready to get started?** Head to **[setup/README.md](setup/README.md)** for complete setup instructions! 🚀