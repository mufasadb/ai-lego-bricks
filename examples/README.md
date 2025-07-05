# Examples Directory

This directory contains curated examples for **Beachy's Project Assistant** - an LLM agentic toolkit. The focus is on JSON agent configurations with simple Python runners to execute them.

## 🎯 Quick Start

```bash
# Run any agent configuration
python run_agent.py agents/simple_chat_agent.json --user-input "Hello!"

# Debug an agent with detailed logging
python run_agent_debug.py agents/document_analysis_agent.json --input document.pdf

# See how to integrate agents into your own applications
python agent_integration_example.py
```

## 📚 What's Here

### 🐍 Python Runners (3 files)
Simple scripts to execute JSON agent configurations:

- **`run_agent.py`** - Clean, minimal agent execution
- **`run_agent_debug.py`** - Detailed logging and debugging  
- **`agent_integration_example.py`** - How to embed agents in your applications

### 🤖 JSON Agent Configurations (`/agents/`)
The real examples! Different agent workflows defined in JSON:

**Basic Agents:**
- **`simple_chat_agent.json`** - Basic conversational agent
- **`anthropic_chat_agent.json`** - Claude-specific chat agent
- **`conversation_chat_agent.json`** - Conversation with memory

**Document Processing:**
- **`document_analysis_agent.json`** - PDF processing and analysis
- **`multi_vision_pdf_comparison.json`** - Advanced PDF analysis with vision
- **`ollama_vision_pdf_comparison.json`** - Ollama-specific vision processing

**Advanced Workflows:**
- **`research_agent.json`** - Multi-source research synthesis
- **`human_approval_workflow.json`** - Interactive approval workflow
- **`multi_turn_conversation.json`** - Complex conversation management
- **`file_output_example.json`** - File output and formatting

## 🔧 Prerequisites

Ensure you have:
1. Completed setup from `/setup/README.md`
2. Configured your `.env` file with necessary API keys
3. Installed requirements: `pip install -r requirements.txt`

## 🚀 Running Examples

### Python Examples
```bash
# Navigate to examples directory
cd examples

# Run a specific example
python basic/simple_chat_example.py

# Most examples accept command line arguments
python intermediate/document_processing_example.py --help
```

### JSON Agent Examples
```bash
# Run JSON agents through the orchestration system
python -c "
from agent_orchestration import AgentOrchestrator
orchestrator = AgentOrchestrator()
workflow = orchestrator.load_workflow_from_file('agents/simple_chat_agent.json')
result = orchestrator.execute_workflow(workflow, {'user_input': 'Hello!'})
print(result)
"
```

## 📖 Learning Path

**New to LLM agents?** Follow this progression:

1. **Start Here**: `basic/simple_chat_example.py`
2. **Add Memory**: `basic/basic_memory_example.py`
3. **Process Documents**: `intermediate/document_processing_example.py`
4. **Structure Responses**: `intermediate/structured_responses_example.py`
5. **Add Logic**: `advanced/conditional_workflows_example.py`
6. **Human Interaction**: `advanced/human_approval_example.py`
7. **JSON Agents**: `agents/simple_chat_agent.json`
8. **Complex Workflows**: `agents/document_analysis_agent.json`

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
| Simple Chat | LLM providers, basic messaging | Foundation setup, API usage |
| Basic Memory | Vector search, CRUD operations | Data persistence, similarity search |
| Document Processing | PDF extraction, content enhancement | File handling, multi-modal processing |
| Model Switching | Dynamic provider selection | Flexibility, model management |
| Structured Responses | Type safety, data validation | Reliable data extraction |
| Conditional Workflows | Decision trees, branching | Complex logic, workflow control |
| Human Approval | Interactive workflows | User interaction, approval patterns |
| JSON Agents | Declarative configuration | No-code agent creation |

---

Ready to start learning? Begin with `basic/simple_chat_example.py` and work your way up! 🚀