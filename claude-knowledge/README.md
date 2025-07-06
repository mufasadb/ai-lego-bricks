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

This helps ensure Claude has current and accurate information about the project context.