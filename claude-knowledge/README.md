# 🤖 Claude Knowledge Base

This folder contains documentation and instructions specifically for Claude Code when working on this project.

## 📁 Contents

- **`CLAUDE.md`** - Project instructions and guidelines for Claude Code
- **`PROJECT_STRUCTURE.md`** - Detailed project structure documentation for Claude's reference
- **`CREDENTIAL_MANAGEMENT.md`** - Secure credential management system documentation ⭐ **NEW**
- **`AGENT_ORCHESTRATION_OVERVIEW.md`** - Complete guide to structured responses and conditional workflows
- **`STRUCTURED_RESPONSES.md`** - Detailed implementation of structured LLM responses
- **`CONDITIONAL_WORKFLOWS.md`** - Detailed implementation of conditional flow control
- **`PROMPT_MANAGEMENT.md`** - Comprehensive guide to prompt management system ⭐ **NEW**
- **`VISUAL_CONTENT_PROCESSING.md`** - Visual to text service with bounding box extraction ⭐ **NEW**
- **`UNIVERSAL_TOOL_SERVICE.md`** - Complete tool service implementation with secure credential management ⭐ **NEW**
- **`AGENT_ORCHESTRATION_GUIDE.md`** - Original orchestration system documentation

## 🎯 Purpose

This folder is designed to provide Claude with comprehensive project context and instructions while keeping these files separate from user-facing documentation.

### Core Documentation

**CLAUDE.md**
Contains specific instructions for how Claude should approach this project, including:
- Development style and philosophy
- Partnership approach guidelines
- Implementation standards
- Project development operations context

**PROJECT_STRUCTURE.md**
Provides detailed technical documentation about:
- Complete project organization
- How different components work together
- Available features and capabilities
- Usage patterns and best practices

### Latest Features

**CREDENTIAL_MANAGEMENT.md** ⭐ **NEW**
Complete guide to the secure credential management system:
- **Library-Safe Credentials**: Optional .env loading to prevent interference when used as dependency
- **Credential Isolation**: Different services can have different credentials for multi-tenant applications
- **Explicit Injection**: Direct credential passing for library integration scenarios
- **Backward Compatibility**: Existing environment variable patterns continue to work
- **Migration Guide**: Step-by-step guide for updating existing code

**AGENT_ORCHESTRATION_OVERVIEW.md**
Comprehensive guide covering major orchestration enhancements:
- **Structured LLM Responses**: Guaranteed, validated responses using Pydantic schemas
- **Conditional Workflows**: Intelligent branching and routing based on LLM decisions
- **Combined Usage Patterns**: How to use both features together
- **Real-world Examples**: Production-ready workflow examples

**PROMPT_MANAGEMENT.md** ⭐ **NEW**
Complete guide to the prompt management system:
- **Externalized Prompts**: Manage prompts outside of code with versioning
- **Template System**: Dynamic prompts with Jinja2 variable substitution
- **Evaluation & A/B Testing**: Performance tracking and optimization
- **Storage & Caching**: File and Supabase backends with registry caching
- **Workflow Integration**: Seamless integration with agent orchestration

**VISUAL_CONTENT_PROCESSING.md** ⭐ **NEW**
Comprehensive guide to visual content processing capabilities:
- **Multi-Format Support**: PDFs, images (.jpg, .png, .bmp, .tiff, .webp), and base64 data
- **Bounding Box Extraction**: Precise text coordinate positioning for layout analysis
- **Vision AI Integration**: Custom prompts for specialized extraction tasks
- **PDF to Images**: Convert PDF pages to base64 images for batch processing
- **Agent Integration**: Document processing steps with visual content support
- **Real-World Examples**: Invoice processing, form analysis, and document extraction

**UNIVERSAL_TOOL_SERVICE.md** ⭐ **NEW**
Complete guide to the universal tool service implementation:
- **Provider Abstraction**: Define tools once, use with OpenAI, Anthropic, Gemini, Ollama
- **Secure Credential Management**: Integrated with CredentialManager for safe API key handling
- **Workflow Integration**: New `tool_call` step type in agent orchestration
- **Performance**: Async execution, concurrent tool calls, registry caching
- **Security Patterns**: Early validation, credential isolation, safe error handling
- **Examples**: Weather, calculator, GitHub API, Slack webhooks, database tools

**STRUCTURED_RESPONSES.md**
Deep dive into the structured response implementation:
- Provider-specific optimizations (Gemini function calling, JSON prompting)
- Multiple schema definition methods (predefined, dictionary, custom classes)
- Error handling and validation strategies
- Integration with agent orchestration

**CONDITIONAL_WORKFLOWS.md**
Complete guide to conditional flow control:
- LLM-based intelligent decision making
- Simple comparison operations
- Multi-level decision trees
- Route mapping and step targeting

## 📝 For Users

Users don't typically need to read these files directly. For getting started with the project, see:
- **Main project documentation:** `README.md` 
- **Setup instructions:** `setup/README.md`
- **Supabase setup:** `setup/SUPABASE_SETUP.md`

## 🔄 Maintenance

These files should be updated whenever:
- Project structure changes significantly
- New features or services are added
- Development guidelines or philosophy evolves
- Setup procedures are modified
- Agent orchestration capabilities are enhanced

This helps ensure Claude has current and accurate information about the project context.

## 🎯 Recent Updates

- **Universal Tool Service**: Complete tool calling implementation with provider abstraction and secure credential management
- **Tool Integration**: New `tool_call` step type added to agent orchestration
- **Security Enhancements**: Comprehensive credential validation and safe error handling
- **Visual Content Processing**: Added comprehensive visual to text service with bounding box extraction
- **Multi-Format Support**: Now supports PDFs, images, and base64 data processing
- **Invoice Processing**: Successfully tested with real Zoom invoices for subscription date extraction
- **Enhanced agent orchestration**: Added structured responses, conditional workflows, and human-in-the-loop capabilities
- **Improved documentation**: Updated all guides to reflect current project state