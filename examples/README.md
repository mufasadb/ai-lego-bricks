# AI Lego Bricks - Examples & Agent Runners

This directory contains the main tools for running and testing JSON-defined agent workflows.

## 🚀 Quick Start

```bash
# Interactive agent testing
./test_agents.sh

# Or run agents directly
python3 examples/run_agent_simple.py agent_orchestration/examples/basic_chat_agent.json --user-input "Hello!"
```

## 📋 Agent Examples

Located in `agent_orchestration/examples/`, these JSON files demonstrate different capabilities:

### ✅ Fully Working Examples

| Agent | Description | Key Features | Status |
|-------|-------------|--------------|--------|
| **basic_chat_agent.json** | Simple conversational AI | Multi-provider support, conversation memory | ✅ Working |
| **ai_coordinator_agent.json** | Intelligent request routing | Routes to Japanese, Home Assistant, or General experts | ✅ Working |
| **streaming_agent.json** | Real-time response streaming | Streaming responses, TTS output | ✅ Working |
| **voice_assistant_agent.json** | Complete voice pipeline | STT → LLM → TTS, conversation logging | ✅ Working |
| **gemini_ollama_parallel_agent.json** | Multi-provider parallel execution | Concurrent analysis, memory storage, synthesis | ✅ Working |
| **json_props_demo_agent.json** | Structured response generation | Natural JSON schema definitions, data validation | ✅ Working |
| **graph_memory_agent.json** | Graph-based memory processing | Entity/relationship extraction, thinking tokens | ✅ Working |
| **parallel_workflow_example.json** | Advanced parallel processing | Technical/business/UX analysis, memory integration | ✅ Working |
| **dollar_amount_extraction_agent.json** | PDF vision analysis | Extract dollar amounts, image cropping, vision AI | ✅ Working |

### 🔧 Partially Working Examples

| Agent | Description | Issues | Status |
|-------|-------------|--------|--------|
| **complex_workflow_agent.json** | Advanced workflow patterns | Template variable substitution issues | 🔧 Needs fixes |

### 🛠️ Recent Infrastructure Improvements

- **Fixed parallel processing**: Dictionary handling in loops and chunking
- **Enhanced vision analysis**: Support for different input formats (PDF→images)
- **Improved output mapping**: Better field mapping for python functions  
- **Better error handling**: Enhanced error propagation and validation

## 🛠️ Agent Runners

### 1. Simple Runner (`run_agent_simple.py`)
Clean, minimal execution with results output.

```bash
python3 examples/run_agent_simple.py agent_orchestration/examples/basic_chat_agent.json \
  --user-input "What is machine learning?" \
  --save-output
```

**Options:**
- `--input <file_or_text>` - Input data (file path or text)
- `--user-input <text>` - User query/input
- `--save-output` - Save results to output folder

### 2. Debug Runner (`run_agent_debug.py`)
Verbose debugging with step-by-step execution details.

```bash
python3 examples/run_agent_debug.py agent_orchestration/examples/document_analysis_agent.json \
  --input document.pdf \
  --debug-level 2 \
  --save-debug
```

**Options:**
- `--debug-level {1,2,3}` - Verbosity level (1=outputs, 2=configs, 3=full)
- `--save-debug` - Save debug log to file
- `--step-pause` - Pause after each step

### 3. Interactive Runner (`run_agent_interactive.py`)
Advanced interactions: streaming, voice, conversations.

```bash
# Conversation mode
python3 examples/run_agent_interactive.py agent_orchestration/examples/basic_chat_agent.json --conversation

# Voice mode
python3 examples/run_agent_interactive.py agent_orchestration/examples/voice_assistant_agent.json \
  --input voice.wav \
  --voice-mode

# Streaming mode
python3 examples/run_agent_interactive.py agent_orchestration/examples/streaming_agent.json --stream
```

**Options:**
- `--stream` - Enable streaming response mode
- `--conversation` - Continuous conversation loop
- `--voice-mode` - Voice input/output processing
- `--file-mode` - Enhanced file processing display
- `--save-session` - Save session data

## 🎯 Testing Navigator (`test_agents.sh`)

Interactive script for exploring and testing agents without memorizing commands.

```bash
# Interactive mode
./test_agents.sh

# Quick commands
./test_agents.sh list      # Show available agents
./test_agents.sh output    # Show recent output files
./test_agents.sh clear     # Clear output directory
```

Features:
- 🔍 Browse available agents with descriptions
- ⚙️ Configure inputs based on agent requirements
- 🚀 Select appropriate runner for your needs
- 📊 View execution results and output files

## 💡 Usage Patterns

### Basic Chat Testing
```bash
python3 examples/run_agent_simple.py agent_orchestration/examples/basic_chat_agent.json \
  --user-input "Explain quantum computing briefly"
```

### Document Analysis
```bash
python3 examples/run_agent_simple.py agent_orchestration/examples/document_analysis_agent.json \
  --input report.pdf \
  --user-input "What are the key findings?"
```

### Voice Interaction
```bash
python3 examples/run_agent_interactive.py agent_orchestration/examples/voice_assistant_agent.json \
  --input question.wav \
  --voice-mode
```

### Continuous Conversation
```bash
python3 examples/run_agent_interactive.py agent_orchestration/examples/basic_chat_agent.json \
  --conversation
```

### Debugging Issues
```bash
python3 examples/run_agent_debug.py agent_orchestration/examples/complex_workflow_agent.json \
  --debug-level 3 \
  --save-debug
```

## 📁 Output Management

All runners save outputs to the `output/` directory:
- `*_results_*.json` - Simple runner results
- `*_debug_*.json` - Debug logs with execution details
- `*_session_*.json` - Interactive session data
- Generated files from agents (audio, analysis, etc.)

## 🔧 Development Tips

1. **Start with Simple Runner** for basic testing
2. **Use Debug Runner** when workflows fail or behave unexpectedly
3. **Try Interactive Runner** for streaming, voice, or conversation agents
4. **Use the Navigator** (`test_agents.sh`) to explore capabilities without CLI complexity

## 🤝 Creating New Agents

1. Study existing JSON examples in `agent_orchestration/examples/`
2. Test with Debug Runner to see step-by-step execution
3. Use the Navigator to rapidly iterate and test
4. Check the [Agent Orchestration Guide](../claude-knowledge/AGENT_ORCHESTRATION_GUIDE.md) for detailed step documentation

## 🔐 Credential Management

All examples support both traditional environment variables and explicit credential injection:

**Traditional (uses .env file):**
```python
# Examples work as-is with .env configuration
python3 examples/run_agent_simple.py agent_orchestration/examples/basic_chat_agent.json
```

**Library-safe (explicit credentials):**
```python
from credentials import CredentialManager

# Pass credentials directly
creds = CredentialManager({
    "GOOGLE_AI_STUDIO_KEY": "your-key"
}, load_env=False)
```

## 🔍 Troubleshooting

**Example not working?**
1. Check your `.env` configuration
2. Verify required services are running (Ollama, Supabase, etc.)
3. Check the example's specific requirements in its docstring
4. Run the setup verification: `python3 setup/setup_supabase.py`

**Need help?**
- Check the main project README.md
- Review setup documentation in `/setup/`
- Examine the `claude-knowledge/` directory for detailed guides