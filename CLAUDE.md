# AI Lego Bricks - Modular Building Blocks for LLM Agents

## Project Overview
AI Lego Bricks is a modular library of building blocks for LLM agentic work, designed to be combined and configured like building blocks. The core philosophy is **JSON-driven configuration over Python code** - almost all behavior should be driven through agent orchestrator JSON files rather than writing Python to configure them.

## The Lego Bricks

### 🎯 Agent Orchestrator (Core Brick)
**The central nervous system** - A JSON-driven workflow orchestration system that combines all other bricks into sophisticated AI agents. Define complex multi-step workflows, conditional logic, loops, and human-in-the-loop interactions entirely through configuration files.

**Key Features:**
- JSON-driven workflows (no Python required for most use cases)
- Conditional routing and loops
- Multi-modal processing pipelines
- Streaming support with real-time responses
- Human approval workflows

### 💬 Chat Services
**Conversation engine** - Unified chat interfaces supporting multiple LLM providers (Gemini, Ollama, OpenAI, Anthropic, OpenRouter). Choose between stateless quick interactions or stateful multi-turn conversations with full history management. OpenRouter provides access to 100+ models through a single API key with unified billing.

### 🧠 Memory Services  
**Long-term knowledge** - Semantic memory system with vector embeddings and graph storage. Automatically store and retrieve relevant context using similarity search across Neo4j and Supabase backends.

### 🛠️ Tools & MCP Integration
**External capabilities** - Universal tool execution framework with full MCP (Model Context Protocol) integration. Automatically discover and use external tools with secure credential management.

### 🗣️ Text-to-Speech (TTS)
**Voice output** - Multi-provider TTS with streaming support. Convert LLM responses to natural speech using OpenAI, Google, or local Coqui-XTTS models.

### 🎤 Speech-to-Text (STT)
**Voice input** - Speech recognition services for audio-based interactions. Transform voice input into text for processing by other bricks.

### 📄 PDF & Visual Processing
**Document intelligence** - Extract text from PDFs, images, and visual content. Supports OCR, bounding box extraction, table parsing, and AI-powered visual analysis.

### 🔤 Text Chunking
**Content preparation** - Intelligent text segmentation for large documents. Optimize content for vector storage and LLM context windows.

### 🎨 Image Generation
**Visual creation** - AI-powered image generation using multiple providers. Create visual content programmatically within agent workflows.

### 📝 Prompt Management
**Prompt engineering** - Versioned prompt templates with A/B testing, execution tracking, and dynamic variable substitution using Jinja2.

### 🔐 Credential Management
**Secure configuration** - Centralized, secure storage and retrieval of API keys and sensitive configuration across all services.

### 📊 Visualizer
**Workflow debugging** - Generate visual diagrams of agent workflows for debugging and documentation purposes.

### 🧪 Testing Framework
**VCR-based testing** - Comprehensive testing infrastructure with three-tier approach: integration tests that record real API calls to VCR cassettes, unit tests that use recorded cassettes for fast CI execution, and CLI commands for easy tape management. Includes automatic security filtering to prevent API key leakage and GitHub Actions integration.

## Core Design Principle: JSON-Driven Configuration

The fundamental principle of AI Lego Bricks is that **behavior should be driven through agent orchestrator JSON files**, not Python code. This means:

✅ **Preferred Approach**: Define agents in JSON configuration files
```json
{
  "name": "DocumentAnalysisAgent",
  "steps": [
    {"type": "pdf_to_text", "input": "document_path"},
    {"type": "chunking", "chunk_size": 1000},
    {"type": "memory_store", "metadata": {"type": "document"}},
    {"type": "llm_analysis", "provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"},
    {"type": "tts", "voice": "alloy"}
  ]
}
```

❌ **Avoid**: Writing Python code to configure each agent manually

This approach provides:
- **Rapid prototyping** - Create new agents in minutes
- **Non-technical accessibility** - Business users can modify agent behavior  
- **Version control** - Track agent evolution through JSON diffs
- **Reusability** - Share and compose agent templates
- **Debugging** - Clear workflow visualization and step-by-step execution

## Getting Started

1. **Explore The Brick Books** - Each brick has detailed documentation in `/The Brick Books/` folder
2. **Check examples** - See `/agent_orchestration/examples/` for JSON workflow templates
3. **Use the CLI** - `ailego` command provides scaffolding and templates
4. **Start with JSON** - Build agents through configuration, not code
5. **Run tests** - Use `ailego test record` to record API interactions, then `ailego test unit` for fast testing

## Testing Framework

AI Lego Bricks includes a comprehensive VCR-based testing framework with three main components:

### 🎬 Integration Tests
- **Real API calls** that record to VCR cassettes
- **Full environment setup** with actual API keys
- **Comprehensive coverage** of chat, HTTP, memory, and orchestration services
- **Command**: `ailego test record [service]`

### ⚡ Unit Tests  
- **Fast execution** using recorded cassettes (no network calls)
- **CI/CD friendly** - runs in GitHub Actions without API keys
- **Identical test logic** to integration tests for consistency
- **Command**: `ailego test unit [service]`

### 🛠️ Tape Management
- **Easy cassette management** with CLI commands
- **Security validation** to prevent API key leakage
- **Update workflows** to refresh recordings when APIs change
- **Commands**: `ailego test info`, `ailego test validate`, `ailego test clean-cassettes`

### Key Benefits
- **No API keys in CI** - Unit tests run entirely offline using cassettes
- **Automatic security filtering** - Sensitive data automatically removed from recordings
- **Easy updates** - Simple commands to refresh when APIs change
- **Full coverage** - Both real API validation and fast CI execution

See `TESTING.md` for complete documentation and usage examples.

## Documentation Structure

- **CLAUDE.md** files in each brick folder contain detailed technical guides for Claude
- **The Brick Books/** contains user-facing documentation for each brick
- **claude-knowledge/** contains general guides and architectural overviews
- **TESTING.md** contains comprehensive testing framework documentation

## Security & Credentials

All bricks follow secure credential patterns - no hardcoded API keys in configuration files. Use the credential management brick for secure, centralized credential storage.

## Philosophy

Think of this as "Infrastructure as Configuration" for AI agents - the power of complex agentic workflows through simple, declarative JSON files that combine pre-built, tested building blocks.