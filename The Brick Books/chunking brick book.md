# Text Chunking Service

A flexible text chunking service that intelligently splits large documents while preserving natural language boundaries. Designed for LLM processing pipelines and document analysis workflows.

## Quick Start

```python
from chunking import create_chunking_service

# Simple chunking with defaults
service = create_chunking_service()
chunks = service.chunk_text("Your large document text here...")

# Custom configuration
service = create_chunking_service({
    'target_size': 1000,
    'tolerance': 200,
    'preserve_paragraphs': True
})
chunks = service.chunk_text(document_text)
```

## Available Strategies

The chunking service uses a hierarchical approach, preserving natural boundaries:

1. **Paragraph-based**: Splits on paragraph separators (`\n\n` by default)
2. **Sentence-based**: Splits on sentence endings (`.!?`)
3. **Word-based**: Splits on word boundaries
4. **Character-based**: Hard cutoff as final fallback

## Common Use Cases

### Processing Large Documents
```python
from chunking import ChunkingService, ChunkingConfig

# For embedding generation (typical 500-1000 tokens)
config = ChunkingConfig(target_size=800, tolerance=200)
service = ChunkingService(config)

with open('large_document.txt', 'r') as f:
    text = f.read()
    
chunks = service.chunk_text(text)
print(f"Split into {len(chunks)} chunks")
```

### Preserving Context for Analysis
```python
# Larger chunks for context preservation
config = ChunkingConfig(
    target_size=1500,
    tolerance=300,
    preserve_paragraphs=True,
    preserve_sentences=True
)
service = ChunkingService(config)
chunks = service.chunk_text(research_paper)
```

### Custom Separators
```python
# For structured text with custom delimiters
config = ChunkingConfig(
    target_size=600,
    tolerance=100,
    paragraph_separator="---",  # Custom separator
    sentence_pattern=r'[.!?;]+\s+'  # Include semicolons
)
service = ChunkingService(config)
```

## Factory Pattern

Use the factory for service management and caching:

```python
from chunking import ChunkingServiceFactory

factory = ChunkingServiceFactory()

# Get default service
service = factory.get_default_service()

# Create with custom config (cached automatically)
service = factory.get_or_create_service({
    'target_size': 1200,
    'tolerance': 150
})

# Clear cache when needed
factory.clear_cache()
```

## Configuration Options

- `target_size`: Desired chunk size in characters
- `tolerance`: Acceptable size variation (±tolerance)
- `preserve_paragraphs`: Try to keep paragraphs intact
- `preserve_sentences`: Try to keep sentences intact  
- `preserve_words`: Try to keep words intact
- `paragraph_separator`: String that separates paragraphs
- `sentence_pattern`: Regex pattern for sentence endings

## Integration Example

```python
# Typical LLM processing pipeline
from chunking import create_chunking_service

def process_document(document_path):
    """Process a document through chunking pipeline."""
    
    # Read document
    with open(document_path, 'r') as f:
        text = f.read()
    
    # Configure for embedding model context window
    chunking_service = create_chunking_service({
        'target_size': 900,  # Leave room for prompt + response
        'tolerance': 100,
        'preserve_paragraphs': True
    })
    
    # Generate chunks
    chunks = chunking_service.chunk_text(text)
    
    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        # Send to LLM, embedding model, etc.
        results.append(process_chunk(chunk))
    
    return results
```

The chunking service automatically handles edge cases like oversized paragraphs, maintains natural language flow, and provides consistent chunk sizes for downstream processing.