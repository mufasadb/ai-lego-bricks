# 🚀 Beachy's Project Assistant - Setup Guide

Complete setup instructions for configuring your LLM project assistant with memory capabilities.

## 📋 Quick Start

1. **Copy Environment File**
   ```bash
   cp .env.example .env
   ```

2. **Choose Your Setup**
   - **Supabase** (Recommended): Cloud-hosted PostgreSQL with pgvector
   - **Neo4j**: Graph database for complex relationships
   - **Both**: Use both for maximum capabilities

3. **Run Verification**
   ```bash
   python setup/setup_supabase.py  # For Supabase
   ```

## 🗂️ Setup Files

- **`setup_supabase_pgvector.sql`** - SQL script to setup pgvector in Supabase
- **`setup_supabase.py`** - Python script to verify Supabase setup
- **`SUPABASE_SETUP.md`** - Detailed Supabase setup instructions
- **`README.md`** - This file

## 🔧 Environment Configuration

### 1. Copy Environment File

```bash
cp .env.example .env
```

**Important**: Never commit the `.env` file to version control - it contains secrets!

### 2. Configure Required Services

Fill in your `.env` file with the appropriate values for the services you want to use:

#### Supabase (Recommended for Memory Storage)
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

#### Ollama (Local LLM Processing)
```env
OLLAMA_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2
```

#### Google AI Studio (Gemini Integration)
```env
GOOGLE_AI_STUDIO_KEY=your_google_ai_studio_key_here
```

## 🗄️ Database Setup

### Option A: Supabase (Recommended)

Supabase provides a cloud-hosted PostgreSQL database with pgvector extensions for advanced vector similarity search.

#### Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Choose your organization
4. Enter project name (e.g., "project-assistant-memory")
5. Enter a strong database password
6. Select region (choose closest to you)
7. Click "Create new project"

#### Step 2: Get Your Credentials

1. **Get Project URL:**
   - In your project dashboard, scroll down to "API Settings"
   - Copy the "Project URL" (looks like: `https://abc123.supabase.co`)

2. **Get API Key:**
   - In your project dashboard, go to "Settings" → "API"
   - Copy the "anon" key (starts with `eyJ...`)
   - **Important**: Use the "anon" key, NOT the "service_role" key

#### Step 3: Update Environment File

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Step 4: Setup Database Schema

1. In your Supabase project, go to "SQL Editor"
2. Copy the contents of `setup/setup_supabase_pgvector.sql`
3. Paste into the SQL Editor
4. Click "Run" to execute

#### Step 5: Verify Setup

```bash
cd setup
python setup_supabase.py
```

You should see all checks pass with ✅ symbols.

### Option B: Neo4j (Advanced Users)

Neo4j provides graph database capabilities for complex memory relationships.

#### Step 1: Install Neo4j

**Option A - Neo4j Desktop:**
1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Install and create a new database
3. Set a password

**Option B - Docker:**
```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

#### Step 2: Configure Environment

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

## 🤖 AI Service Setup

### Google AI Studio (Gemini)

1. Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the API key
5. Update your `.env`:
   ```env
   GOOGLE_AI_STUDIO_KEY=your_api_key_here
   ```

### Ollama (Local LLM)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull llama2
   ```
3. Update your `.env`:
   ```env
   OLLAMA_URL=http://localhost:11434
   OLLAMA_DEFAULT_MODEL=llama2
   ```

## 🎯 Prompt Management Setup

The system includes a comprehensive prompt management system for versioning, evaluation, and A/B testing.

### Configuration Options

Add these to your `.env` file:

```env
# Prompt Management Configuration
PROMPT_STORAGE_BACKEND=auto      # auto, file, or supabase
PROMPT_STORAGE_PATH=./prompts    # for file backend (development)
PROMPT_CACHE_TTL=3600           # cache timeout in seconds
PROMPT_EVALUATION_ENABLED=true  # enable execution logging
```

### Storage Backend Options

**1. Auto Detection (Recommended)**
- Uses Supabase if available, falls back to file storage
- Perfect for seamless dev-to-production workflow

**2. File Storage (Development)**
- JSON files stored locally
- Great for development and testing
- No external dependencies

**3. Supabase Storage (Production)**
- Uses same Supabase instance as memory storage
- Enables advanced evaluation and analytics
- Requires Supabase setup (see above)

### First-Time Setup

1. **Create sample prompts:**
   ```bash
   python prompt/example_usage.py
   ```

2. **Verify prompt system:**
   ```bash
   python -c "from prompt import create_prompt_service; print('✅ Prompt system working!')"
   ```

## 🔍 Verification

### Supabase Setup Verification

```bash
cd setup
python setup_supabase.py
```

**Expected Output:**
```
🚀 Supabase pgvector Setup Verification
==================================================
✅ PASS - Connection
✅ PASS - pgvector Extension
✅ PASS - memories Table
✅ PASS - match_memories Function
✅ PASS - Vector Operations

🎉 All checks passed! Your Supabase instance is ready for pgvector.
```

### Test Memory Operations

```python
from memory import create_memory_service

# Auto-detects and uses available services
memory_service = create_memory_service("auto")

# Store a memory
memory_id = memory_service.store_memory(
    "Machine learning is transforming software development",
    {"category": "AI", "importance": "high"}
)

# Search for similar memories
results = memory_service.retrieve_memories("AI development", limit=5)
for memory in results:
    print(f"- {memory.content}")
```

## 🔐 Security Best Practices

1. **Never commit `.env` files** - Add `.env` to your `.gitignore`
2. **Use strong passwords** for all database services
3. **Rotate API keys regularly** - especially for production use
4. **Use environment-specific configs** - separate dev/staging/prod
5. **Monitor API usage** - watch for unexpected usage patterns

## 🚨 Troubleshooting

### Common Issues

#### "Could not connect to Supabase"
- Check your `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`
- Ensure you're using the **anon** key, not the service role key
- Verify your project is active in Supabase dashboard

#### "pgvector extension not found"
- Run the SQL script in Supabase SQL Editor
- Ensure the `vector` extension is enabled
- Check that your Supabase project supports extensions

#### "match_memories function not found"
- Run the complete `setup_supabase_pgvector.sql` script
- Verify the function was created in the `public` schema

#### "Vector search returns no results"
- Check embedding dimensions match (384 for all-MiniLM-L6-v2)
- Lower the similarity threshold (try 0.1 - 0.3)
- Ensure test data has embeddings

### Getting Help

1. **Check the logs** - Enable debug logging in your application
2. **Review Supabase dashboard** - Check for errors in the dashboard
3. **Verify credentials** - Double-check all API keys and URLs
4. **Test connections** - Use the verification scripts

## 📚 Additional Resources

### Documentation
- [Supabase Vector/Embeddings Guide](https://supabase.com/docs/guides/ai/vector-columns)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Neo4j Documentation](https://neo4j.com/docs/)

### Models and Libraries
- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [Ollama Models](https://ollama.ai/library)

## 🎯 What's Next?

After completing setup:

1. **Test the memory service** with simple queries
2. **Explore the project structure** to understand the codebase
3. **Run your first project decomposition** using the LLM agent
4. **Customize the configuration** for your specific needs

Happy coding! 🚀