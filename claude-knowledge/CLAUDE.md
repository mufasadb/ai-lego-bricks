# Claude Code Guidelines for LLM Project Tool, for crafting files for programming agent.

## Project Overview
This is a comprehensive collection of building blocks for creating sophisticated LLM-powered agents. The project has evolved into a modular system where different components can be orchestrated together to build complex AI workflows:

### Core Building Blocks
- **LLM Abstraction Layer**: Unified interface for multiple LLM providers (Gemini, Ollama) with text, vision, and embedding clients
- **Universal Tool Service**: Register tools once, use with any LLM provider (OpenAI, Anthropic, Gemini, Ollama)
- **Memory Management**: Persistent storage with semantic search (Supabase pgvector, Neo4j)
- **Document Processing**: PDF extraction with LLM enhancement and semantic analysis
- **Text Chunking**: Intelligent text segmentation preserving semantic boundaries
- **Chat Interface**: Conversational interface with history management
- **Agent Orchestration**: JSON-driven system for combining all components into sophisticated workflows

### Agent Orchestration System ⭐ **LATEST ENHANCEMENTS**
The JSON-driven agent orchestration system has been significantly enhanced with three major capabilities:

**1. Structured LLM Responses (NEW)**
- **Guaranteed data types**: All LLM responses return validated Pydantic models
- **Provider optimization**: Gemini uses function calling, others use enhanced JSON prompting
- **Multiple schema methods**: Predefined schemas, dictionary definitions, or custom Pydantic classes
- **Built-in schemas**: classification, decision, summary, extraction, simple_response
- **Error handling**: Retry logic, fallback mechanisms, and validation

**2. Conditional Workflows (NEW)**
- **Intelligent routing**: LLM makes context-aware decisions about workflow paths
- **Simple comparisons**: Numeric/string comparisons with full operator support
- **Multi-level decisions**: Complex decision trees and branching logic
- **Route mapping**: Maps condition results to target step IDs for dynamic flow control
- **Loop protection**: Prevents infinite loops with execution tracking

**3. Human-in-the-Loop Workflows (NEW)**
- **Human approval nodes**: Interactive approval and feedback collection during workflow execution
- **Multiple interaction types**: approve/reject, multiple choice, custom text input
- **Context presentation**: Shows relevant workflow data to help human decision-making
- **Timeout handling**: Configurable timeouts with default fallback actions
- **Data flow integration**: Human responses become available to subsequent AI steps
- **Conditional routing**: Different workflow paths based on human decisions

**Core Capabilities:**
- Execute multi-step workflows defined in JSON
- Chain together different services (LLM, memory, document processing, etc.)
- Handle data flow between steps with structured validation
- Support conditional logic, branching, and dynamic routing
- Provide comprehensive error handling and execution tracking 

### Universal Tool Service ⭐ **LATEST ADDITION**
A comprehensive tool calling system that provides unified interface across all LLM providers:

**Key Features:**
- **Provider Abstraction**: Define tools once, use with OpenAI, Anthropic, Gemini, Ollama
- **Secure Credential Management**: Integrated with CredentialManager for safe API key handling
- **Workflow Integration**: New `tool_call` step type in agent orchestration
- **Performance**: Async execution, concurrent tool calls, registry caching
- **Security**: Early validation, credential isolation, safe error handling

**New Step Type: `tool_call`**
- Execute tools/APIs with LLM guidance
- Support for all provider tool calling formats
- Automatic credential management and validation
- Rich conversation tracking and error handling

**Examples Provided:**
- Weather, calculator, file management tools
- Secure API tools (GitHub, OpenAI, Slack, Supabase)
- Comprehensive testing and demonstration scripts

## Development Style
The goal here is to have Claude assist the user in learning how to utilise the tools in this project, so each task should be broken down and steppe through so that the user has an opportunity to learn whats going on and could feasibly continue development themselves without Claude afterward

## Partnership Approach
Claude Code should act as a thoughtful development partner, not just executing instructions blindly. Always:
- Critically evaluate each task for context and potential unintended impacts
- Ask clarifying questions when requirements are unclear or seem problematic
- Consider how changes fit within the broader application architecture
- Start every task by crafting a todo list to clarify intentions and next steps

## Implementation Standards
- Consider upcoming work during implementation but avoid leaving TODO comments in code
- Avoid mocking data unless explicitly instructed
- Write production-ready code, not placeholders
- Follow existing code patterns and conventions

## Completing Tasks
- Finish work during a task, if there is an error or compatibility issue, seek to fix it now rather than put it off or fake the passing of a test

## Project dev opps
the user runs neo4j and ollama on an unraid server, cloudflare is used to route to the unraid server from anywhere, if the service is local it will be on 192.168.0.15 (normal port) and then otherwise the services are custom urls (toolname.beachysapp.com usually)