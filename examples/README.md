# Examples Directory

This directory contains curated examples for **AI Lego Bricks** - a modular LLM agent system. These examples demonstrate core functionality including streaming, TTS integration, and agent workflows.

## 🎯 Quick Start

```bash
# Basic streaming example
python streaming_example.py

# TTS workflow example  
python tts_workflow_example.py

# Streaming LLM to TTS integration
python streaming_integration_demo.py

# Agent integration patterns
python agent_integration_example.py
```

## 📚 What's Here

### Core Examples
- **`streaming_example.py`** - LLM streaming demonstrations
- **`streaming_integration_demo.py`** - Working streaming LLM → TTS pipeline
- **`tts_workflow_example.py`** - Text-to-speech workflows and agent integration
- **`agent_integration_example.py`** - How to embed agents in your applications
- **`concept_evaluation_example.py`** - Prompt management and evaluation
- **`prompt_management_example.py`** - Advanced prompt handling
- **`run_agent.py`** - Simple agent execution runner

### Featured Workflows

**Streaming & Audio:**
- **Streaming LLM responses** with real-time output
- **TTS integration** with multiple providers  
- **Streaming → TTS pipeline** for audio generation

**Agent Orchestration:**
- **Basic chat agents** with JSON configuration
- **Document processing** with PDF analysis
- **Memory integration** with semantic search
- **Prompt management** with versioning and evaluation

## 🔧 Prerequisites

Ensure you have:
1. Completed setup from `/setup/README.md`
2. Configured your `.env` file with necessary API keys
3. Installed requirements: `pip install -r requirements.txt`

### 🔐 Credential Management

All examples support both traditional environment variables and explicit credential injection:

**Traditional (uses .env file):**
```python
# Examples work as-is with .env configuration
python streaming_example.py
```

**Library-safe (explicit credentials):**
```python
from credentials import CredentialManager

# Pass credentials directly
creds = CredentialManager({
    "GOOGLE_AI_STUDIO_KEY": "your-key"
}, load_env=False)

# Use in examples that support credential_manager parameter
```

## 🚀 Running Examples

### Core Examples
```bash
# Navigate to examples directory
cd examples

# Test streaming functionality
python streaming_example.py

# Test TTS integration
python tts_workflow_example.py

# Test streaming LLM → TTS pipeline
python streaming_integration_demo.py
```

### Agent Examples
```bash
# Run agent through orchestration system
python -c "
from agent_orchestration import AgentOrchestrator
orchestrator = AgentOrchestrator()
workflow = orchestrator.load_workflow_from_file('../agent_orchestration/examples/simple_chat_agent.json')
result = orchestrator.execute_workflow(workflow, {'user_query': 'Hello!'})
print(result.final_output)
"
```

## 📖 Learning Path

**New to AI Lego Bricks?** Follow this progression:

1. **Basic Streaming**: `streaming_example.py`
2. **TTS Integration**: `tts_workflow_example.py`  
3. **Streaming Pipeline**: `streaming_integration_demo.py`
4. **Agent Basics**: `agent_integration_example.py`
5. **Prompt Management**: `prompt_management_example.py`
6. **Advanced Concepts**: `concept_evaluation_example.py`

## 🎨 Creating Your Own Examples

Want to add new examples? Follow these guidelines:

1. **Choose the right directory** based on complexity
2. **Include docstrings** explaining what the example demonstrates
3. **Add error handling** for missing dependencies/configuration
4. **Provide sample inputs** or use reasonable defaults
5. **Document expected outputs** in comments
6. **Test with `run_examples.py`** before submitting

## 🔍 Troubleshooting

**Example not working?**
1. Check your `.env` configuration
2. Verify required services are running (Ollama, Supabase, etc.)
3. Check the example's specific requirements in its docstring
4. Run the setup verification: `python setup/setup_supabase.py`

**Need help?**
- Check the main project README.md
- Review setup documentation in `/setup/`
- Examine the `claude-knowledge/` directory for detailed guides

## 🎯 What Each Example Teaches

| Example | Core Concepts | Skills Learned |
|---------|--------------|----------------|
| `streaming_example.py` | LLM streaming, real-time responses | Streaming APIs, progressive output |
| `streaming_integration_demo.py` | LLM → TTS pipeline | Audio generation, sentence chunking |
| `tts_workflow_example.py` | TTS providers, agent orchestration | Audio workflows, multi-provider TTS |
| `agent_integration_example.py` | JSON agents, workflow execution | Declarative configuration, modularity |
| `prompt_management_example.py` | Versioned prompts, templates | Prompt lifecycle, template systems |
| `concept_evaluation_example.py` | A/B testing, prompt optimization | Evaluation metrics, continuous improvement |

---

Ready to start learning? Begin with `streaming_example.py` and work your way up! 🚀