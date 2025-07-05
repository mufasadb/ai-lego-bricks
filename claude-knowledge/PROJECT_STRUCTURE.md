# Project Structure

## 📁 Organized Code Structure

The codebase has been organized into logical folders for better maintainability and easy setup:

```
/Beachys-project-assistant/
├── setup/                         # 🔧 Setup and configuration
│   ├── README.md                 # Complete setup guide
│   ├── SUPABASE_SETUP.md         # Detailed Supabase instructions
│   ├── setup_supabase.py         # Supabase verification script
│   └── setup_supabase_pgvector.sql # Database schema setup
├── llm/                           # 🤖 LLM abstraction layer
│   ├── __init__.py               # Package exports
│   ├── llm_client_factory.py     # Factory for creating LLM clients
│   ├── text_clients.py           # Text generation clients (Gemini, Ollama)
│   ├── vision_clients.py         # Vision analysis clients
│   ├── embedding_client.py       # Embedding generation
│   ├── model_manager.py          # Model switching and management
│   ├── llm_types.py              # Type definitions and enums
│   └── providers/                # Provider-specific implementations
├── memory/                        # 🧠 Memory service package
│   ├── __init__.py               # Package exports
│   ├── memory_service.py         # Abstract base class & Memory dataclass
│   ├── memory_factory.py         # Factory for creating memory services
│   ├── neo4j_memory_service.py   # Neo4j implementation
│   ├── supabase_memory_service.py # Supabase + pgvector implementation  
│   └── memory_example.py         # Usage examples and demos
├── pdf_to_text/                   # 📄 Document processing package
│   ├── __init__.py               # Package exports
│   ├── pdf_to_text_service.py    # PDF extraction with LLM enhancement
│   ├── pdf_extract_options.py    # Configuration for extraction options
│   └── example_usage.py          # PDF processing examples
├── chunking/                      # ✂️ Text chunking service
│   ├── __init__.py               # Package exports
│   ├── chunking_service.py       # Intelligent text segmentation
│   ├── chunking_config.py        # Configuration for chunking strategies
│   └── example_usage.py          # Chunking usage examples
├── chat/                          # 💬 Chat service package
│   ├── __init__.py               # Package exports
│   ├── chat_service.py           # LLM integrations (Ollama, Gemini)
│   └── example_usage.py          # Chat usage examples
├── agent_orchestration/           # 🎭 Agent orchestration system
│   ├── __init__.py               # Package exports
│   ├── orchestrator.py           # Main orchestration classes
│   ├── models.py                 # Pydantic models for configuration
│   ├── step_handlers.py          # Step-specific execution logic
│   ├── schemas/                  # JSON schemas for validation
│   │   └── workflow_schema.json  # Workflow configuration schema
│   └── examples/                 # Example workflows and usage
│       ├── simple_chat_agent.json
│       ├── document_analysis_agent.json
│       ├── research_agent.json
│       └── usage_example.py
├── claude-knowledge/              # 🤖 Claude-specific documentation
│   ├── CLAUDE.md                 # Claude guidelines and project overview
│   ├── PROJECT_STRUCTURE.md      # This file - project structure docs
│   └── README.md                 # Knowledge base overview
├── test/                          # 🧪 Test suite
│   ├── __init__.py               # Test package
│   ├── test_delete_operations.py # Delete functionality tests
│   ├── test_bulk_delete.py       # Bulk delete tests
│   └── test_chunking_service.py  # Chunking service tests
├── README.md                      # 📖 Main project documentation
├── requirements.txt               # 📦 Python dependencies
├── .env.example                   # 📝 Environment template
├── .env                          # 🔐 Environment configuration (user creates)
└── CLAUDE.md                     # 🤖 Claude-specific instructions
```

## 🚀 How to Use

### First Time Setup
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Follow setup guide
cat setup/README.md

# 3. Verify Supabase setup (if using Supabase)
python setup/setup_supabase.py
```

### Agent Orchestration System (Primary Interface)
```bash
# Run example orchestration workflows
python agent_orchestration/examples/usage_example.py

# Create custom workflows with JSON configuration
```
```python
from agent_orchestration import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator()

# Load workflow from JSON
workflow = orchestrator.load_workflow_from_file("my_agent.json")

# Execute with inputs
result = orchestrator.execute_workflow(workflow, {
    "document_path": "/path/to/document.pdf",
    "user_query": "What are the main findings?"
})
```

### LLM Services
```bash
# Run LLM examples
python llm/model_switching_example.py

# Test vision capabilities
python llm/vision_example.py
```
```python
from llm import LLMClientFactory, LLMProvider, VisionProvider

# Create text client
factory = LLMClientFactory()
text_client = factory.create_text_client(LLMProvider.GEMINI)
response = text_client.chat("Hello, world!")

# Create vision client
vision_client = factory.create_vision_client(VisionProvider.GEMINI_VISION)
analysis = vision_client.analyze_image("image.jpg", "Describe this image")
```

### Memory Service
```bash
# Run memory examples
python memory/memory_example.py
```
```python
from memory import create_memory_service
memory_service = create_memory_service("auto")  # Auto-detects available services
```

### Document Processing
```bash
# Run PDF processing examples
python pdf_to_text/example_usage.py
```
```python
from pdf_to_text import PDFToTextService
pdf_service = PDFToTextService()
result = pdf_service.extract_text_with_llm_enhancement("document.pdf")
```

### Text Chunking
```bash
# Run chunking examples
python chunking/example_usage.py
```
```python
from chunking import ChunkingService, ChunkingConfig
chunking_service = ChunkingService(ChunkingConfig(target_size=1000))
chunks = chunking_service.chunk_text("Your long text here...")
```

### Chat Interface
```bash
# Run chat examples
python chat/example_usage.py
```
```python
from chat import ChatService
chat_service = ChatService("gemini")
response = chat_service.chat("Hello, how are you?")
```

### Running Tests
```bash
# Test delete operations
python test/test_delete_operations.py

# Test bulk delete functionality  
python test/test_bulk_delete.py

# Test chunking service
python test/test_chunking_service.py
```

## ✅ What Works

### Agent Orchestration System
- ✅ **JSON-driven workflows:** Define complex agent behavior through configuration
- ✅ **Step-by-step execution:** Clear data flow between operations
- ✅ **Service integration:** Combines all building blocks seamlessly
- ✅ **Error handling:** Comprehensive error reporting and graceful fallbacks
- ✅ **Input/output mapping:** Flexible data passing between steps
- ✅ **Example workflows:** Chat agent, document analysis, research agent
- ✅ **Extensible:** Easy to add new step types and handlers

### LLM Abstraction Layer
- ✅ **Multi-provider support:** Gemini and Ollama integration
- ✅ **Text generation:** Chat and completion capabilities
- ✅ **Vision analysis:** Image analysis with Gemini Vision
- ✅ **Embedding generation:** Text vectorization for similarity search
- ✅ **Model switching:** Runtime model changes without client recreation
- ✅ **Factory pattern:** Consistent interface across providers
- ✅ **Configuration management:** Flexible model and parameter configuration

### Document Processing
- ✅ **PDF extraction:** Traditional and LLM-enhanced text extraction
- ✅ **Vision fallback:** LLM vision for challenging PDFs
- ✅ **Content enhancement:** LLM-powered formatting improvement
- ✅ **Semantic analysis:** Document classification and key point extraction
- ✅ **RAG integration:** Automatic chunking for vector databases
- ✅ **Configurable options:** Flexible extraction strategies

### Text Chunking Service
- ✅ **Boundary preservation:** Respects paragraphs, sentences, and words
- ✅ **Configurable strategies:** Target size with tolerance ranges
- ✅ **Fallback hierarchy:** Graceful degradation from paragraph → sentence → word → character
- ✅ **Integration ready:** Used by PDF service for RAG applications
- ✅ **Semantic awareness:** Preserves meaning across chunk boundaries

### Memory Service Features
- ✅ **Auto-detection:** Finds available services (Neo4j/Supabase)
- ✅ **Neo4j support:** Graph database with relationships
- ✅ **Supabase + pgvector:** PostgreSQL with vector similarity search
- ✅ **Vector search:** Semantic similarity with embeddings
- ✅ **Hybrid search:** Combined vector and text search (Supabase)
- ✅ **CRUD operations:** Create, read, update, delete memories
- ✅ **Bulk operations:** Efficient multi-delete with single query
- ✅ **Search-based delete:** Delete by query criteria
- ✅ **Metadata support:** Structured data with JSON serialization
- ✅ **Pydantic models:** Type-safe data validation and serialization

### Chat Interface
- ✅ **Multi-provider support:** Ollama and Gemini integration
- ✅ **History management:** Automatic conversation tracking
- ✅ **Message format:** Standardized ChatMessage structure
- ✅ **Backward compatibility:** Legacy adapter for existing code

### Testing & Quality
- ✅ **Comprehensive tests:** All core operations validated
- ✅ **Error handling:** Invalid inputs, missing services, network issues
- ✅ **Data integrity:** Relationships cleaned up, search consistency
- ✅ **Performance:** Bulk operations use optimized queries
- ✅ **Type safety:** Pydantic models throughout for validation

## 🔧 Configuration

### Environment Variables
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687      # or your Neo4j server
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Supabase Configuration (with pgvector)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key      # NOT service role key

# AI/ML Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2     # Sentence transformer model
```

### Setup Instructions
For detailed setup instructions, see the setup folder:

```bash
# Complete setup guide
cat setup/README.md

# Supabase-specific setup
cat setup/SUPABASE_SETUP.md

# Verify Supabase configuration
python setup/setup_supabase.py
```

The setup process includes:
1. **Environment Configuration** - Copy .env.example to .env
2. **Database Setup** - Choose between Supabase (recommended) or Neo4j
3. **API Keys** - Configure Ollama, Gemini, or other LLM services
4. **Verification** - Test all connections and functionality

## 📦 Benefits of Current Structure

1. **Easy Setup:** Dedicated setup folder with comprehensive guides
2. **Modularity:** Clear separation of concerns (setup, memory, chat, tests)
3. **Maintainability:** Easier to find and modify code
4. **User-Friendly:** New users can get started quickly with setup/README.md
5. **Testing:** Isolated test suite with comprehensive coverage
6. **Reusability:** Memory package can be imported elsewhere
7. **Scalability:** Easy to add new services (Redis, Elasticsearch, etc.)
8. **Clean imports:** Proper Python package structure
9. **Documentation:** Multiple levels of documentation for different needs
10. **Production-Ready:** Includes environment templates, verification scripts

## 🎯 Project Philosophy

This project follows a **learning-focused development approach**:

- **Educational:** Each component helps users understand LLM project patterns
- **Well-documented:** Clear explanations and examples throughout
- **Modular:** Easy to extend, modify, and understand
- **Partnership-focused:** Claude acts as a thoughtful development partner

The codebase is now production-ready with a clean, organized structure that prioritizes both functionality and user experience! 🎉