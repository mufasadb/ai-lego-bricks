# Memory Service

A flexible memory system for AI agents that supports multiple storage backends with vector embeddings for semantic search and retrieval.

## Features

- **Multiple Storage Backends**: Neo4j graph database and Supabase with pgvector
- **Vector Embeddings**: Semantic similarity search using sentence transformers
- **Graph Knowledge**: Entity and relationship extraction for structured memory
- **Metadata Support**: Rich metadata storage and filtering
- **Auto-detection**: Automatic backend selection based on available credentials

## Quick Start

```python
from memory import create_memory_service, Memory

# Auto-detect available service
memory_service = create_memory_service("auto")

# Store a memory
memory_id = memory_service.store_memory(
    "Client meeting about dashboard requirements",
    metadata={"type": "meeting", "client": "Acme Corp", "priority": "high"}
)

# Retrieve similar memories
memories = memory_service.retrieve_memories("dashboard requirements", limit=5)

# Get specific memory
memory = memory_service.get_memory_by_id(memory_id)
```

## Storage Backends

### Neo4j Graph Database
```python
# Environment variables
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Create service
memory_service = create_memory_service("neo4j")
```

### Supabase with pgvector
```python
# Environment variables
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key

# Create service
memory_service = create_memory_service("supabase")
```

## Core Operations

### Store Memory
```python
memory_id = memory_service.store_memory(
    content="Implemented user authentication with JWT tokens",
    metadata={
        "type": "development",
        "component": "auth",
        "status": "completed",
        "team": "backend"
    }
)
```

### Retrieve by Similarity
```python
# Find memories similar to a query
memories = memory_service.retrieve_memories(
    query="authentication issues",
    limit=10
)

for memory in memories:
    print(f"Content: {memory.content}")
    print(f"Metadata: {memory.metadata}")
    print(f"Timestamp: {memory.timestamp}")
```

### Update Memory
```python
success = memory_service.update_memory(
    memory_id=memory_id,
    content="Updated: Authentication system now includes refresh tokens",
    metadata={"status": "enhanced", "version": "2.0"}
)
```

### Delete Memories
```python
# Delete single memory
success = memory_service.delete_memory(memory_id)

# Delete multiple memories
results = memory_service.delete_memories([id1, id2, id3])

# Delete by search query
results = memory_service.delete_memories_by_search(
    query="old authentication notes",
    limit=5,
    confirm=True
)
```

## Advanced Features

### Graph Knowledge Extraction
```python
from memory.graph_formatter_service import GraphFormatterService
from llm import GenerationService
from prompt import PromptService

# Create graph formatter
formatter = GraphFormatterService(generation_service, prompt_service)

# Extract entities and relationships
graph_data = formatter.format_memory_as_graph(
    content="John Smith called about the API integration project",
    extraction_mode="comprehensive"
)

print(f"Entities: {graph_data.entities}")
print(f"Relationships: {graph_data.relationships}")
```

### Check Available Services
```python
from memory import get_available_services

available = get_available_services()
print(f"Neo4j available: {available['neo4j']}")
print(f"Supabase available: {available['supabase']}")
```

## Common Use Cases

### AI Agent Memory
```python
class AIAgent:
    def __init__(self):
        self.memory = create_memory_service("auto")
    
    def remember(self, event, context=None):
        return self.memory.store_memory(event, context)
    
    def recall(self, query, limit=5):
        return self.memory.retrieve_memories(query, limit)
    
    def learn_from_conversation(self, conversation):
        memory_id = self.memory.store_memory(
            conversation["content"],
            metadata={
                "type": "conversation",
                "user": conversation["user"],
                "timestamp": conversation["timestamp"]
            }
        )
        return memory_id
```

### Project Knowledge Base
```python
def build_project_knowledge():
    memory = create_memory_service("auto")
    
    # Store project decisions
    memory.store_memory(
        "Decided to use PostgreSQL for user data storage",
        metadata={"type": "decision", "component": "database", "date": "2024-01-15"}
    )
    
    # Store bug reports
    memory.store_memory(
        "Users reporting slow page loads on dashboard",
        metadata={"type": "bug", "severity": "medium", "component": "frontend"}
    )
    
    # Retrieve related memories
    related = memory.retrieve_memories("database performance issues")
    return related
```

## Requirements

- Python 3.8+
- For Neo4j: `neo4j`, `sentence-transformers`, `numpy`
- For Supabase: `supabase`, `sentence-transformers`, `numpy`
- For graph features: `llm` and `prompt` modules from this project

## Setup

1. Install dependencies for your chosen backend
2. Set environment variables for your storage service
3. Use `create_memory_service("auto")` for automatic detection
4. Start storing and retrieving memories with semantic search