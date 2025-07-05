# Supabase pgvector Setup Guide

This guide walks you through setting up pgvector in your Supabase instance for advanced vector similarity search in the memory service.

## 🎯 Overview

The memory service can use either Neo4j or Supabase for storage. When using Supabase, pgvector enables:
- **Vector similarity search** - Find semantically similar memories
- **Hybrid search** - Combine vector and text search
- **Fast retrieval** - Optimized indexes for large datasets
- **SQL flexibility** - Custom queries and analytics

## 📋 Prerequisites

1. **Supabase Project**: Create a project at [supabase.com](https://supabase.com)
2. **Python Dependencies**: Ensure you have the required packages:
   ```bash
   pip install supabase sentence-transformers
   ```

## 🔧 Step-by-Step Setup

### 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click **"New Project"**
3. Choose your organization
4. Enter project name (e.g., "project-assistant-memory")
5. Enter a strong database password
6. Select region (choose closest to you)
7. Click **"Create new project"**

### 2. Get Your Supabase Credentials

#### Get Project URL:
1. In your project dashboard, scroll down to **"API Settings"** section
2. Copy the **"Project URL"** (looks like: `https://abc123.supabase.co`)

#### Get API Key:
1. In your project dashboard, go to **"Settings"** → **"API"**
2. Copy the **"anon"** key (starts with `eyJ...`)
3. **Important**: Use the **"anon"** key, NOT the **"service_role"** key

### 3. Configure Environment Variables

First, copy the environment template:
```bash
cp .env.example .env
```

Then update your `.env` file:
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

### 4. Enable pgvector Extension

1. Open your Supabase project dashboard
2. Go to **SQL Editor**
3. Run the following SQL to enable pgvector:
   ```sql
   create extension if not exists vector;
   ```

### 5. Set Up Database Schema

In the SQL Editor, run the complete setup script:

```sql
-- Copy and paste the entire contents of setup/setup_supabase_pgvector.sql
```

Or run it from the file:
1. Open `setup/setup_supabase_pgvector.sql` in this project
2. Copy the entire contents
3. Paste and execute in Supabase SQL Editor

### 6. Verify Setup

Run the verification script:
```bash
cd setup
python setup_supabase.py
```

This will check:
- ✅ Connection to Supabase
- ✅ pgvector extension enabled
- ✅ memories table with vector column
- ✅ match_memories function
- ✅ Vector operations working

## 🗄️ Database Schema

The setup creates a `memories` table with:

```sql
create table memories (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata jsonb default '{}',
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);
```

### Indexes Created:
- **Vector similarity**: `ivfflat` index for fast vector search
- **Metadata queries**: `gin` index for JSON queries  
- **Text search**: `gin` index for full-text search

## 🔍 Search Functions

### Vector Similarity Search
```sql
SELECT * FROM match_memories(
    '[0.1, 0.2, 0.3, ...]'::vector,  -- Query embedding
    0.5,  -- Similarity threshold (0-1)
    10    -- Max results
);
```

### Hybrid Search (Vector + Text)
```sql
SELECT * FROM hybrid_search_memories(
    'machine learning',               -- Text query
    '[0.1, 0.2, 0.3, ...]'::vector, -- Vector query
    0.5,  -- Similarity threshold
    10    -- Max results
);
```

## 🔐 Security Considerations

### Row Level Security (RLS)
The setup enables RLS with a policy allowing access for authenticated users:

```sql
-- Adjust this policy based on your security requirements
create policy "Allow all operations for authenticated users" on memories
    for all using (auth.role() = 'authenticated' or auth.role() = 'anon');
```

### API Keys
- **Anon key**: Safe for client-side use, respects RLS policies
- **Service role key**: Full access, keep secret, use server-side only

## 🚀 Usage in Code

Once set up, the memory service will automatically use pgvector:

```python
from memory import create_memory_service

# Auto-detects and uses Supabase if available
memory_service = create_memory_service("auto")

# Store a memory (automatically generates embeddings)
memory_id = memory_service.store_memory(
    "Machine learning is transforming software development",
    {"category": "AI", "importance": "high"}
)

# Search for similar memories
results = memory_service.retrieve_memories("AI development", limit=5)
for memory in results:
    print(f"- {memory.content}")
```

## 🎛️ Configuration Options

### Embedding Model
Change the embedding model in `.env`:
```bash
# Options: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Vector Dimensions
If you change the embedding model, update the vector dimension in the SQL schema:
```sql
-- For all-MiniLM-L6-v2: 384 dimensions
-- For all-mpnet-base-v2: 768 dimensions
embedding vector(384)  -- Change this number
```

## 🐛 Troubleshooting

### Common Issues

1. **"Invalid API key"**
   - Check your `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`
   - Ensure you're using the anon key, not the service role key

2. **"pgvector extension not found"**
   - Run `create extension if not exists vector;` in SQL Editor
   - Some Supabase projects may need to enable it manually

3. **"match_memories function not found"**
   - Run the complete `setup_supabase_pgvector.sql` script
   - Check the function was created in the `public` schema

4. **Vector search returns no results**
   - Check embedding dimensions match (384 for all-MiniLM-L6-v2)
   - Lower the similarity threshold (try 0.1 - 0.3)
   - Verify test data was inserted correctly

### Debug Mode
Enable debug logging to see what's happening:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your memory operations
```

## 📚 Additional Resources

- [Supabase Vector/Embeddings Guide](https://supabase.com/docs/guides/ai/vector-columns)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)

## 🆘 Getting Help

If you encounter issues:
1. Run `cd setup && python setup_supabase.py` to diagnose problems
2. Check the Supabase dashboard logs
3. Review the SQL Editor for any error messages
4. Ensure your .env file has the correct credentials
5. Make sure you copied .env.example to .env first