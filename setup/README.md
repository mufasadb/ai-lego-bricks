# 🚀 AI Lego Bricks - Complete Setup Guide

Welcome to the comprehensive setup guide for AI Lego Bricks! This guide will walk you through everything you need to get started with your modular AI agent system.

## 🎯 Overview

AI Lego Bricks is designed to be **easy to set up** and **production-ready** from day one. This guide covers:

- **Quick Start**: Get running in 5 minutes
- **Complete Installation**: Full feature setup with all services
- **Platform-Specific Instructions**: Windows, macOS, Linux
- **Troubleshooting**: Common issues and solutions
- **Production Deployment**: Security and performance considerations

## ⚡ Quick Start (5 minutes)

### Prerequisites
- Python 3.8+ (`python --version`)
- Git (`git --version`)
- At least one AI service API key (Google AI Studio recommended)

### 1. Clone and Install
```bash
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks
pip install -e .
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (minimum required)
nano .env  # or use your preferred editor
```

**Minimum required for Quick Start:**
```env
GOOGLE_AI_STUDIO_KEY=your_google_ai_studio_key_here
```

### 3. Verify Installation
```bash
ailego verify
```

### 4. Run Your First Agent
```bash
ailego run agent_orchestration/examples/basic_chat_agent.json
```

**🎉 Success!** You now have a working AI agent. Continue below for advanced features.

## 🛠️ Complete Installation

### Platform-Specific Setup

#### Windows
```cmd
# Install Python and Git (if not already installed)
# Download from python.org and git-scm.com

# Clone repository
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install
pip install -e .
```

#### macOS
```bash
# Install dependencies (if needed)
brew install python3 git

# Clone repository
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .
```

#### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/callmebeachy/ai-lego-bricks.git
cd ai-lego-bricks

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .
```

### Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your service credentials:

#### 🔑 Essential API Keys

**Google AI Studio (Recommended first choice):**
1. Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Create API key
3. Add to `.env`:
```env
GOOGLE_AI_STUDIO_KEY=your_api_key_here
```

**OpenAI (Alternative):**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create API key
3. Add to `.env`:
```env
OPENAI_API_KEY=your_api_key_here
```

**Anthropic (Alternative):**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create API key
3. Add to `.env`:
```env
ANTHROPIC_API_KEY=your_api_key_here
```

#### 🎯 Memory & Storage Services

**Supabase (Vector Database - Recommended):**
1. Create project at [supabase.com](https://supabase.com)
2. Get Project URL and anon key
3. Add to `.env`:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
```

**Neo4j (Graph Database - Optional):**
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

#### 🗣️ Speech Services

**Text-to-Speech:**
```env
# Already covered: GOOGLE_AI_STUDIO_KEY, OPENAI_API_KEY
# For local TTS (optional):
COQUI_MODEL_PATH=./models/tts
```

**Speech-to-Text:**
```env
# For local STT (optional):
WHISPER_MODEL_SIZE=base
```

#### 🎨 Image Generation

**DALL-E (OpenAI):**
```env
OPENAI_API_KEY=your_openai_key  # Same as above
```

**Imagen (Google):**
```env
GOOGLE_AI_STUDIO_KEY=your_google_key  # Same as above
```

## 🗄️ Database Setup

### Supabase Setup (Recommended)

#### 1. Create Supabase Project
1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Choose organization and enter project details
4. Wait for project creation (2-3 minutes)

#### 2. Get Credentials
1. **Project URL**: Found in project settings
2. **Anon Key**: Settings → API → "anon" key (NOT service_role)

#### 3. Setup Database Schema
1. Go to SQL Editor in Supabase
2. Copy contents of `setup/setup_supabase_pgvector.sql`
3. Paste and run the script
4. Verify tables and functions were created

#### 4. Verify Setup
```bash
python setup/setup_supabase.py
```

Expected output:
```
✅ PASS - Connection
✅ PASS - pgvector Extension
✅ PASS - memories Table
✅ PASS - match_memories Function
✅ PASS - Vector Operations
```

### Neo4j Setup (Optional)

#### Docker Installation
```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest
```

#### Desktop Installation
1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Install and create database
3. Set password and start database

## 🔍 Verification & Testing

### System Verification
```bash
# Basic verification
ailego verify

# Detailed verification
ailego verify --verbose

# Check system status
ailego status
```

### Component Testing

**Test LLM Connection:**
```python
from llm import create_text_client
client = create_text_client("gemini")
response = client.generate_text("Hello, world!")
print(response)
```

**Test Memory Service:**
```python
from memory import create_memory_service
memory = create_memory_service("auto")
memory_id = memory.store_memory("Test memory", {"type": "test"})
print(f"Stored memory: {memory_id}")
```

**Test Agent Execution:**
```bash
ailego run agent_orchestration/examples/basic_chat_agent.json
```

## 🚨 Troubleshooting

### Platform-Specific Issues

#### Windows
- **Path Issues**: Use forward slashes or raw strings: `r"C:\path\to\file"`
- **Environment Variables**: Use `set` instead of `export`
- **PowerShell**: Wrap variables in quotes: `$env:KEY="value"`

#### macOS
- **Xcode Tools**: Run `xcode-select --install` if needed
- **Permissions**: May need `sudo` for system-wide installs
- **Homebrew**: Install dependencies with `brew install python3 git`

#### Linux
- **Dependencies**: Install with `sudo apt install python3 python3-pip python3-venv git`
- **Build Tools**: May need `sudo apt install build-essential python3-dev`

### Common Issues

#### "Could not connect to Supabase"
- ✅ **Check credentials**: Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY`
- ✅ **Key type**: Use **anon** key, not service_role key
- ✅ **Project status**: Verify project is active in Supabase
- ✅ **Network**: Try accessing URL in browser
- ✅ **Firewall**: Check if corporate firewall blocks connection

#### "Module not found" errors
- ✅ **Installation**: Reinstall with `pip install -e .`
- ✅ **Virtual environment**: Activate correct environment
- ✅ **Dependencies**: Run `pip install -r requirements.txt`
- ✅ **Python path**: Check project directory is accessible

#### "API key invalid"
- ✅ **Copy/paste**: Check for extra spaces or newlines
- ✅ **Expiration**: Verify keys haven't expired
- ✅ **Permissions**: Ensure keys have necessary permissions
- ✅ **Regenerate**: Try creating new API key

#### ".env file not loading"
- ✅ **Location**: Ensure `.env` is in project root
- ✅ **Format**: Use `KEY=value` (no spaces around =)
- ✅ **Comments**: Don't use inline comments
- ✅ **Quotes**: Use quotes for values with spaces

### Performance Issues

#### "Slow response times"
- ✅ **Model choice**: Use smaller models for speed
- ✅ **Batch size**: Reduce batch sizes for memory operations
- ✅ **Caching**: Enable caching in configuration
- ✅ **Network**: Check internet connection stability

#### "Memory usage too high"
- ✅ **Cleanup**: Implement proper cleanup of large objects
- ✅ **Streaming**: Use streaming for large responses
- ✅ **Batch processing**: Process large datasets in chunks

## 🔐 Security & Production

### Security Best Practices

1. **Environment Variables**:
   - Never commit `.env` files
   - Use different keys for dev/staging/prod
   - Rotate API keys regularly

2. **Database Security**:
   - Use strong passwords
   - Enable SSL/TLS connections
   - Restrict database access by IP

3. **API Key Management**:
   - Set usage limits on API keys
   - Monitor API usage for anomalies
   - Use least-privilege access

### Production Deployment

#### Environment Setup
```bash
# Production environment
cp .env.example .env.production

# Edit with production credentials
nano .env.production
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "-m", "ailego", "run", "your_agent.json"]
```

#### Monitoring
- Set up logging for all services
- Monitor API usage and costs
- Set up alerts for failures
- Regular health checks

## 🎯 Next Steps

After completing setup:

1. **Learn the Basics**:
   - Run example agents
   - Understand the JSON configuration format
   - Explore available services

2. **Build Your First Custom Agent**:
   - Use `ailego create` to generate templates
   - Modify agent configurations
   - Add custom tools and prompts

3. **Advanced Features**:
   - Set up memory services for persistent knowledge
   - Configure multi-modal workflows
   - Integrate custom tools and APIs

4. **Production Ready**:
   - Implement proper error handling
   - Set up monitoring and logging
   - Deploy with appropriate security measures

## 📚 Additional Resources

### Documentation
- [Agent Orchestration Guide](../agent_orchestration/README.md)
- [Universal Tools Documentation](../tools/README.md)
- [Memory Services Guide](../memory/README.md)
- [LLM Integration Guide](../llm/README.md)

### External Resources
- [Supabase Documentation](https://supabase.com/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Google AI Studio](https://makersuite.google.com)
- [Anthropic Claude API](https://docs.anthropic.com)

### Community
- [GitHub Issues](https://github.com/callmebeachy/ai-lego-bricks/issues)
- [Contributing Guide](../CONTRIBUTING.md)
- [License](../LICENSE)

---

**🚀 Ready to build amazing AI agents?** Start with the examples and work your way up to custom workflows!