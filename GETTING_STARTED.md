# 🚀 Getting Started with AI Lego Bricks

A quick guide to help you get up and running with AI Lego Bricks based on user feedback about documentation gaps and environment setup.

## 🎯 Quick Start (5 Minutes)

### 1. Installation
```bash
# Install from source
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks
pip install -e .
```

### 2. Basic Environment Setup
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (minimum required)
# Add at least one of these:
# GOOGLE_AI_STUDIO_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### 3. Test Your Setup
```python
# Create test_setup.py
from llm.generation_service import quick_generate_gemini
from dotenv import load_dotenv

# This loads .env automatically
load_dotenv()

try:
    response = quick_generate_gemini("Say hello!")
    print(f"✅ Success: {response}")
except Exception as e:
    print(f"❌ Error: {e}")
```

```bash
python test_setup.py
```

## 📚 Core Import Patterns

Based on user feedback, here are the **correct import paths** for common functions:

### Memory Service
```python
from memory import create_memory_service
from credentials import CredentialManager

# Environment variables (default)
memory = create_memory_service("auto")

# Explicit credentials (library-safe)
creds = CredentialManager({
    "SUPABASE_URL": "your-url",
    "SUPABASE_ANON_KEY": "your-key"
}, load_env=False)
memory = create_memory_service("supabase", credential_manager=creds)
```

### LLM Services
```python
# Quick generation (one-shot)
from llm.generation_service import quick_generate_gemini, quick_generate_openai

# Conversation service (multi-turn)
from chat.conversation_service import create_gemini_conversation

# Text client (advanced)
from llm import create_text_client
```

### Tool Service
```python
# Get available tools
from tools import get_tool_service

tool_service = get_tool_service()
available_tools = tool_service.get_available_tools()  # NOT tool_service.get_available_tools()
```

### Agent Orchestration
```python
from agent_orchestration import AgentOrchestrator

orchestrator = AgentOrchestrator()
workflow = orchestrator.load_workflow_from_file("agent.json")
result = orchestrator.execute_workflow(workflow, inputs)
```

## 🔧 Environment Loading Best Practices

### Issue: `.env` loading wasn't automatic
**Solution**: Always use explicit `load_dotenv()` calls:

```python
from dotenv import load_dotenv
from credentials import CredentialManager

# Option 1: Load environment globally (simple)
load_dotenv()
from llm.generation_service import quick_generate_gemini
response = quick_generate_gemini("Hello")

# Option 2: Explicit credential management (library-safe)
creds = CredentialManager(load_env=True)  # Loads .env explicitly
from llm import create_text_client
client = create_text_client("gemini", credential_manager=creds)
```

### Environment File Template
```bash
# Copy and edit .env.example
cp .env.example .env
```

**Minimum required** (choose one):
```bash
# Google AI Studio (Gemini)
GOOGLE_AI_STUDIO_KEY=your_key_here

# OpenAI
OPENAI_API_KEY=your_key_here

# Anthropic
ANTHROPIC_API_KEY=your_key_here
```

**Optional but recommended**:
```bash
# Supabase (for memory storage)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key

# Ollama (for local LLM)
OLLAMA_BASE_URL=http://localhost:11434
```

## 🏗️ First Working Example

Here's a complete example that addresses the main pain points:

```python
# working_example.py
from dotenv import load_dotenv
from credentials import CredentialManager
from llm.generation_service import quick_generate_gemini
from memory import create_memory_service
from tools import get_tool_service

# IMPORTANT: Load environment first
load_dotenv()

def main():
    print("🧱 AI Lego Bricks - Working Example")
    
    # 1. Test basic LLM generation
    print("\n1. Testing LLM Generation...")
    try:
        response = quick_generate_gemini("What is 2+2?")
        print(f"✅ LLM Response: {response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return
    
    # 2. Test memory service (if Supabase configured)
    print("\n2. Testing Memory Service...")
    try:
        memory = create_memory_service("auto")
        # Store a test memory
        memory.store_memory("AI Lego Bricks is working!", {"test": True})
        
        # Retrieve it
        results = memory.retrieve_memories("AI Lego Bricks")
        print(f"✅ Memory Results: {len(results)} memories found")
    except Exception as e:
        print(f"⚠️ Memory Error (Supabase not configured?): {e}")
    
    # 3. Test tool service
    print("\n3. Testing Tool Service...")
    try:
        tool_service = get_tool_service()
        tools = tool_service.get_available_tools()
        print(f"✅ Available Tools: {len(tools)} tools")
    except Exception as e:
        print(f"❌ Tool Error: {e}")
    
    print("\n✅ Setup verification complete!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python working_example.py
```

## 🎨 Common Usage Patterns

### Pattern 1: Simple Text Generation
```python
from dotenv import load_dotenv
from llm.generation_service import quick_generate_gemini

load_dotenv()
response = quick_generate_gemini("Explain quantum computing")
print(response)
```

### Pattern 2: Conversation with Memory
```python
from dotenv import load_dotenv
from chat.conversation_service import create_gemini_conversation
from memory import create_memory_service

load_dotenv()

# Create conversation and memory
conv = create_gemini_conversation()
memory = create_memory_service("auto")

# Store context
memory.store_memory("User prefers technical explanations", {"user_id": "123"})

# Have conversation
response = conv.send_message("What is machine learning?")
print(response)
```

### Pattern 3: Tool-Enabled Agent
```python
from dotenv import load_dotenv
from agent_orchestration import AgentOrchestrator

load_dotenv()

# Create simple agent JSON
agent_config = {
    "name": "helper_agent",
    "steps": [{
        "id": "chat",
        "type": "llm_chat",
        "config": {"provider": "gemini"},
        "inputs": {"message": "Hello, what can you help me with?"}
    }]
}

orchestrator = AgentOrchestrator()
result = orchestrator.execute_workflow(agent_config, {})
print(result)
```

## 🔍 Troubleshooting Common Issues

### Issue: "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Issue: "CredentialManager not found"
```python
# Correct import
from credentials import CredentialManager
```

### Issue: "get_available_tools() method not found"
```python
# Correct usage
from tools import get_tool_service
tool_service = get_tool_service()
tools = tool_service.get_available_tools()  # Note: no extra ()
```

### Issue: Pydantic warnings
These are non-breaking. To suppress:
```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
```

## 📈 Next Steps

1. **Explore Examples**: Check `examples/` folder for working demos
2. **Try Agent Orchestration**: Look at `agent_orchestration/examples/`
3. **Add More Tools**: See `tools/` for integrations
4. **Setup Memory**: Follow `setup/SUPABASE_SETUP.md` for persistent storage
5. **Read Full Documentation**: Return to main README for advanced features

## 🤝 Getting Help

- **Import Issues**: Double-check the import patterns above
- **Environment Issues**: Ensure `.env` is loaded with `load_dotenv()`
- **Function Names**: Use the exact function names from the examples
- **API Keys**: Verify your API keys are correct in `.env`

**Common Function Names Reference:**
- `create_memory_service()` - NOT `create_memory()`
- `quick_generate_gemini()` - NOT `generate_gemini()`
- `get_tool_service()` - NOT `create_tool_service()`
- `create_text_client()` - NOT `create_client()`

---

This guide addresses the main pain points from user feedback. For complete documentation, see the main [README.md](README.md).