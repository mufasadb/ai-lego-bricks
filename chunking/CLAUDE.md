# Text Chunking Service - Detailed Technical Documentation

## 🔧 Architecture Overview

The Text Chunking Service is a sophisticated hierarchical text segmentation system designed for LLM processing pipelines. It uses a multi-tier fallback strategy to preserve natural language boundaries while maintaining consistent chunk sizes for downstream processing.

### Core Components

```
ChunkingService (Main Service)
├── ChunkingConfig (Configuration)
├── ChunkingServiceFactory (Factory Pattern)
├── create_chunking_service() (Convenience Function)
└── Multi-tier Chunking Algorithm
    ├── Paragraph-based Chunking
    ├── Sentence-based Chunking
    ├── Word-based Chunking
    └── Character-based Chunking (Last Resort)
```

## 📊 Chunking Algorithm Deep Dive

### Hierarchical Boundary Preservation

The service implements a hierarchical approach to text chunking:

1. **Paragraph-level Chunking** (Primary Strategy)
   - Splits on paragraph separators (default: `\n\n`)
   - Preserves complete paragraphs when possible
   - Maintains semantic coherence within chunks

2. **Sentence-level Chunking** (Secondary Strategy)
   - Uses regex pattern matching (default: `[.!?]+\s+`)
   - Preserves complete sentences when paragraphs are too large
   - Maintains readability and grammatical structure

3. **Word-level Chunking** (Tertiary Strategy)
   - Splits on whitespace boundaries
   - Preserves word integrity when sentences are too large
   - Maintains basic semantic units

4. **Character-level Chunking** (Last Resort)
   - Hard cutoff at character boundaries
   - Only used when words exceed maximum size
   - Preserves character integrity

### Size Management Strategy

```python
class ChunkingConfig:
    target_size: int           # Desired chunk size (characters)
    tolerance: int             # Acceptable deviation (±tolerance)
    min_size = target_size - tolerance
    max_size = target_size + tolerance
```

**Size Calculation Logic:**
- **Optimal Range**: `target_size ± tolerance`
- **Acceptance Criteria**: Chunk must be within `[min_size, max_size]`
- **Fallback Behavior**: If no boundary-preserving chunk fits, use next hierarchy level

### Chunking Process Flow

```
Input Text
    ↓
Check if text fits in single chunk
    ↓ (No)
Try paragraph-based chunking
    ↓ (Paragraph too large)
Try sentence-based chunking
    ↓ (Sentence too large)
Try word-based chunking
    ↓ (Word too large)
Force character-based chunking
    ↓
Return chunks
```

## 🔬 Implementation Details

### Core Chunking Algorithm

```python
def _get_next_chunk(self, text: str) -> Optional[str]:
    """
    Extract the next chunk using hierarchical boundary preservation.
    
    Priority Order:
    1. Paragraph boundaries (if preserve_paragraphs=True)
    2. Sentence boundaries (if preserve_sentences=True)
    3. Word boundaries (if preserve_words=True)
    4. Character boundaries (forced cutoff)
    """
    
    # If text fits in single chunk, return it
    if len(text) <= self.max_size:
        return text
    
    # Try paragraph-based chunking
    if self.config.preserve_paragraphs:
        chunk = self._chunk_by_paragraphs(text)
        if chunk:
            return chunk
    
    # Try sentence-based chunking
    if self.config.preserve_sentences:
        chunk = self._chunk_by_sentences(text)
        if chunk:
            return chunk
    
    # Try word-based chunking
    if self.config.preserve_words:
        chunk = self._chunk_by_words(text)
        if chunk:
            return chunk
    
    # Force character-based chunking
    return self._chunk_by_characters(text)
```

### Paragraph-based Chunking

```python
def _chunk_by_paragraphs(self, text: str) -> Optional[str]:
    """
    Chunk text by paragraph boundaries.
    
    Algorithm:
    1. Split text by paragraph separator
    2. Accumulate paragraphs until size limit
    3. Return chunk if within tolerance
    4. Return None if no valid paragraph chunk possible
    """
    paragraphs = text.split(self.config.paragraph_separator)
    
    current_chunk = ""
    for paragraph in paragraphs:
        test_chunk = current_chunk + paragraph
        if self.config.paragraph_separator in current_chunk:
            test_chunk += self.config.paragraph_separator
        
        if len(test_chunk) <= self.max_size:
            current_chunk = test_chunk
        else:
            # If we have accumulated paragraphs, return the chunk
            if current_chunk and len(current_chunk) >= self.min_size:
                return current_chunk
            # If first paragraph is too large, fall back to sentence chunking
            break
    
    # Return final chunk if valid
    if current_chunk and len(current_chunk) >= self.min_size:
        return current_chunk
    
    return None
```

### Sentence-based Chunking

```python
def _chunk_by_sentences(self, text: str) -> Optional[str]:
    """
    Chunk text by sentence boundaries using regex pattern.
    
    Algorithm:
    1. Split text using sentence pattern regex
    2. Accumulate sentences until size limit
    3. Return chunk if within tolerance
    """
    sentences = re.split(self.config.sentence_pattern, text)
    
    current_chunk = ""
    for sentence in sentences:
        test_chunk = current_chunk + sentence
        
        if len(test_chunk) <= self.max_size:
            current_chunk = test_chunk
        else:
            if current_chunk and len(current_chunk) >= self.min_size:
                return current_chunk
            break
    
    if current_chunk and len(current_chunk) >= self.min_size:
        return current_chunk
    
    return None
```

### Word-based Chunking

```python
def _chunk_by_words(self, text: str) -> Optional[str]:
    """
    Chunk text by word boundaries.
    
    Algorithm:
    1. Split text by whitespace
    2. Accumulate words until size limit
    3. Return chunk if within tolerance
    """
    words = text.split()
    
    current_chunk = ""
    for word in words:
        test_chunk = current_chunk + (" " if current_chunk else "") + word
        
        if len(test_chunk) <= self.max_size:
            current_chunk = test_chunk
        else:
            if current_chunk and len(current_chunk) >= self.min_size:
                return current_chunk
            break
    
    if current_chunk and len(current_chunk) >= self.min_size:
        return current_chunk
    
    return None
```

### Character-based Chunking (Force Cutoff)

```python
def _chunk_by_characters(self, text: str) -> str:
    """
    Force character-based chunking as last resort.
    
    Used when:
    - Single word exceeds max_size
    - No other boundary-preserving strategy works
    """
    return text[:self.max_size]
```

## 🏭 Factory Pattern Implementation

### ChunkingServiceFactory

```python
class ChunkingServiceFactory:
    """
    Factory for creating and caching chunking services.
    
    Benefits:
    - Service instance reuse
    - Configuration-based caching
    - Resource optimization
    """
    
    def __init__(self):
        self._cache = {}
    
    def get_or_create_service(self, config: Union[ChunkingConfig, Dict]) -> ChunkingService:
        """
        Get or create a chunking service with caching.
        
        Cache Key Generation:
        - Based on configuration parameters
        - Ensures unique services for unique configs
        """
        cache_key = self._generate_cache_key(config)
        
        if cache_key not in self._cache:
            if isinstance(config, dict):
                config = ChunkingConfig(**config)
            self._cache[cache_key] = ChunkingService(config)
        
        return self._cache[cache_key]
    
    def _generate_cache_key(self, config: Union[ChunkingConfig, Dict]) -> str:
        """Generate cache key from configuration."""
        if isinstance(config, dict):
            return str(sorted(config.items()))
        return str(config)
```

## 🔧 Configuration Deep Dive

### ChunkingConfig Parameters

```python
@dataclass
class ChunkingConfig:
    # Core Size Parameters
    target_size: int                    # Primary chunk size target
    tolerance: int                      # Acceptable size deviation
    
    # Boundary Preservation Flags
    preserve_paragraphs: bool = True    # Try paragraph boundaries first
    preserve_sentences: bool = True     # Try sentence boundaries second
    preserve_words: bool = True         # Try word boundaries third
    
    # Delimiter Configuration
    paragraph_separator: str = "\n\n"   # What separates paragraphs
    sentence_pattern: str = r'[.!?]+\s+' # Regex for sentence endings
```

### Advanced Configuration Examples

```python
# For embedding models (typical 512-1024 tokens)
embedding_config = ChunkingConfig(
    target_size=800,
    tolerance=200,
    preserve_paragraphs=True,
    preserve_sentences=True,
    preserve_words=True
)

# For LLM context windows (larger chunks)
llm_config = ChunkingConfig(
    target_size=2000,
    tolerance=500,
    preserve_paragraphs=True,
    preserve_sentences=True,
    preserve_words=True
)

# For tight size constraints
strict_config = ChunkingConfig(
    target_size=500,
    tolerance=50,
    preserve_paragraphs=True,
    preserve_sentences=True,
    preserve_words=False  # Allow word breaking if needed
)

# For structured text (e.g., code, markdown)
structured_config = ChunkingConfig(
    target_size=1000,
    tolerance=200,
    paragraph_separator="---",  # Custom separator
    sentence_pattern=r'[.!?;]+\s+',  # Include semicolons
    preserve_paragraphs=True,
    preserve_sentences=True,
    preserve_words=True
)
```

## 🎯 Use Case Patterns

### RAG (Retrieval-Augmented Generation) Applications

```python
def create_rag_chunks(document_text: str, embedding_model_context: int = 512):
    """
    Create chunks optimized for RAG applications.
    
    Considerations:
    - Embedding model context window
    - Semantic coherence
    - Overlap for context preservation
    """
    # Account for embedding model tokenization (rough estimate: 4 chars per token)
    target_chars = embedding_model_context * 4 * 0.8  # 80% utilization
    
    config = ChunkingConfig(
        target_size=int(target_chars),
        tolerance=int(target_chars * 0.2),  # 20% tolerance
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    return service.chunk_text(document_text)
```

### LLM Processing Pipeline

```python
def create_llm_processing_chunks(text: str, model_context: int = 4096):
    """
    Create chunks for LLM processing with context preservation.
    
    Considerations:
    - Model context window
    - Prompt overhead
    - Response space
    """
    # Reserve space for prompt and response
    available_chars = (model_context - 1000) * 4  # Rough char estimate
    
    config = ChunkingConfig(
        target_size=int(available_chars),
        tolerance=int(available_chars * 0.1),  # 10% tolerance
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    return service.chunk_text(text)
```

### Memory-Constrained Processing

```python
def create_memory_efficient_chunks(text: str, max_memory_mb: int = 100):
    """
    Create chunks for memory-constrained environments.
    
    Considerations:
    - Memory usage estimation
    - Processing overhead
    - Garbage collection efficiency
    """
    # Estimate chars per MB (rough: 1MB ≈ 1M chars)
    chars_per_mb = 1_000_000
    max_chars = max_memory_mb * chars_per_mb * 0.5  # 50% utilization
    
    config = ChunkingConfig(
        target_size=int(max_chars),
        tolerance=int(max_chars * 0.15),  # 15% tolerance
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    return service.chunk_text(text)
```

## 🔄 Integration Patterns

### With PDF Processing

```python
def chunk_pdf_content(pdf_path: str) -> List[str]:
    """
    Integrated PDF processing and chunking pipeline.
    """
    from pdf_to_text import extract_text_from_pdf
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Configure chunking for PDF content
    config = ChunkingConfig(
        target_size=1000,
        tolerance=200,
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    return service.chunk_text(text)
```

### With Memory Service

```python
def chunk_and_store_memories(text: str, memory_service) -> List[str]:
    """
    Chunk text and store in memory service.
    """
    chunks = create_chunking_service().chunk_text(text)
    
    memory_ids = []
    for i, chunk in enumerate(chunks):
        memory_id = memory_service.store_memory(
            content=chunk,
            metadata={
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": "chunking_service"
            }
        )
        memory_ids.append(memory_id)
    
    return memory_ids
```

### With Agent Orchestration

```python
def create_chunking_workflow_step():
    """
    Create chunking step for agent orchestration.
    """
    return {
        "type": "chunking",
        "name": "chunk_document",
        "config": {
            "target_size": 1000,
            "tolerance": 200,
            "preserve_paragraphs": True,
            "preserve_sentences": True,
            "preserve_words": True
        },
        "input_mapping": {
            "text": "document_text"
        },
        "output_mapping": {
            "chunks": "text_chunks"
        }
    }
```

## 🚀 Performance Optimizations

### Chunking Strategy Selection

```python
def optimize_chunking_strategy(text: str, target_size: int) -> ChunkingConfig:
    """
    Dynamically optimize chunking strategy based on text characteristics.
    """
    # Analyze text characteristics
    paragraph_count = text.count('\n\n')
    sentence_count = len(re.findall(r'[.!?]+', text))
    word_count = len(text.split())
    
    # Calculate average sizes
    avg_paragraph_size = len(text) / max(paragraph_count, 1)
    avg_sentence_size = len(text) / max(sentence_count, 1)
    avg_word_size = len(text) / max(word_count, 1)
    
    # Choose optimal strategy
    if avg_paragraph_size <= target_size * 1.5:
        # Paragraphs are reasonably sized
        return ChunkingConfig(
            target_size=target_size,
            tolerance=target_size // 5,
            preserve_paragraphs=True,
            preserve_sentences=True,
            preserve_words=True
        )
    elif avg_sentence_size <= target_size * 0.8:
        # Sentences are small enough
        return ChunkingConfig(
            target_size=target_size,
            tolerance=target_size // 10,
            preserve_paragraphs=False,
            preserve_sentences=True,
            preserve_words=True
        )
    else:
        # Need word-level chunking
        return ChunkingConfig(
            target_size=target_size,
            tolerance=target_size // 20,
            preserve_paragraphs=False,
            preserve_sentences=False,
            preserve_words=True
        )
```

### Memory-Efficient Processing

```python
def chunk_large_document_streaming(file_path: str, chunk_size: int = 1000):
    """
    Process large documents in streaming fashion to minimize memory usage.
    """
    config = ChunkingConfig(
        target_size=chunk_size,
        tolerance=chunk_size // 5,
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    
    # Process file in chunks to avoid loading entire file into memory
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""
        for line in f:
            buffer += line
            
            # Process when buffer gets large enough
            if len(buffer) > chunk_size * 3:  # 3x chunk size buffer
                chunks = service.chunk_text(buffer)
                
                # Yield all but last chunk (might be incomplete)
                for chunk in chunks[:-1]:
                    yield chunk
                
                # Keep last chunk as start of next buffer
                buffer = chunks[-1] if chunks else ""
        
        # Process remaining buffer
        if buffer:
            chunks = service.chunk_text(buffer)
            for chunk in chunks:
                yield chunk
```

## 🧪 Testing Strategies

### Unit Testing Patterns

```python
def test_chunking_boundary_preservation():
    """Test that natural boundaries are preserved."""
    config = ChunkingConfig(
        target_size=100,
        tolerance=20,
        preserve_paragraphs=True,
        preserve_sentences=True,
        preserve_words=True
    )
    
    service = ChunkingService(config)
    
    # Test paragraph preservation
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = service.chunk_text(text)
    
    # Verify paragraphs are not split
    for chunk in chunks:
        assert "\n\n" not in chunk or chunk.count("\n\n") < 2
    
    # Test sentence preservation
    text = "First sentence. Second sentence. Third sentence."
    chunks = service.chunk_text(text)
    
    # Verify sentences are not split
    for chunk in chunks:
        assert not chunk.endswith(".")
```

### Integration Testing

```python
def test_chunking_with_real_documents():
    """Test chunking with real document samples."""
    # Test with various document types
    test_cases = [
        ("legal_document.txt", 1000),
        ("technical_manual.txt", 800),
        ("novel_excerpt.txt", 1200),
        ("scientific_paper.txt", 900)
    ]
    
    for doc_path, target_size in test_cases:
        with open(doc_path, 'r') as f:
            text = f.read()
        
        config = ChunkingConfig(
            target_size=target_size,
            tolerance=target_size // 5,
            preserve_paragraphs=True,
            preserve_sentences=True,
            preserve_words=True
        )
        
        service = ChunkingService(config)
        chunks = service.chunk_text(text)
        
        # Verify chunk sizes
        for chunk in chunks:
            assert len(chunk) <= target_size + config.tolerance
            assert len(chunk) >= target_size - config.tolerance
```

## 🔍 Debugging and Monitoring

### Chunking Statistics

```python
def analyze_chunking_performance(text: str, config: ChunkingConfig) -> Dict:
    """
    Analyze chunking performance and provide statistics.
    """
    service = ChunkingService(config)
    chunks = service.chunk_text(text)
    
    # Calculate statistics
    chunk_sizes = [len(chunk) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(chunk_sizes),
        "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "size_variance": sum((s - config.target_size) ** 2 for s in chunk_sizes) / len(chunk_sizes),
        "boundary_preservation": {
            "paragraphs_split": sum(1 for chunk in chunks if "\n\n" in chunk),
            "sentences_split": sum(1 for chunk in chunks if re.search(r'[.!?]+\s+', chunk)),
            "words_split": sum(1 for chunk in chunks if " " in chunk)
        }
    }
```

### Performance Profiling

```python
def profile_chunking_performance(text: str, config: ChunkingConfig):
    """
    Profile chunking performance for optimization.
    """
    import time
    
    start_time = time.time()
    service = ChunkingService(config)
    
    # Profile each chunking strategy
    paragraph_time = 0
    sentence_time = 0
    word_time = 0
    
    # Mock the internal methods to measure time
    original_chunk_by_paragraphs = service._chunk_by_paragraphs
    original_chunk_by_sentences = service._chunk_by_sentences
    original_chunk_by_words = service._chunk_by_words
    
    def timed_paragraph_chunking(text):
        nonlocal paragraph_time
        start = time.time()
        result = original_chunk_by_paragraphs(text)
        paragraph_time += time.time() - start
        return result
    
    def timed_sentence_chunking(text):
        nonlocal sentence_time
        start = time.time()
        result = original_chunk_by_sentences(text)
        sentence_time += time.time() - start
        return result
    
    def timed_word_chunking(text):
        nonlocal word_time
        start = time.time()
        result = original_chunk_by_words(text)
        word_time += time.time() - start
        return result
    
    # Patch methods
    service._chunk_by_paragraphs = timed_paragraph_chunking
    service._chunk_by_sentences = timed_sentence_chunking
    service._chunk_by_words = timed_word_chunking
    
    # Run chunking
    chunks = service.chunk_text(text)
    total_time = time.time() - start_time
    
    return {
        "total_time": total_time,
        "paragraph_time": paragraph_time,
        "sentence_time": sentence_time,
        "word_time": word_time,
        "chunks_generated": len(chunks),
        "chars_per_second": len(text) / total_time
    }
```

## 🎯 Best Practices

### Configuration Guidelines

1. **Target Size Selection**
   - Consider downstream processing requirements
   - Account for tokenization overhead (4:1 char to token ratio)
   - Leave buffer space for prompts and responses

2. **Tolerance Settings**
   - Use 10-20% tolerance for strict requirements
   - Use 20-30% tolerance for flexible processing
   - Use 30%+ tolerance for memory-constrained environments

3. **Boundary Preservation**
   - Always preserve paragraphs for coherent content
   - Preserve sentences for readability
   - Preserve words for semantic integrity
   - Only break words for extreme size constraints

### Performance Optimization

1. **Service Reuse**
   - Use factory pattern for service caching
   - Avoid creating new services for same configuration
   - Clear cache periodically to prevent memory leaks

2. **Text Preprocessing**
   - Normalize whitespace before chunking
   - Remove unnecessary formatting
   - Consider text encoding issues

3. **Memory Management**
   - Use streaming processing for large documents
   - Process chunks immediately rather than accumulating
   - Clear intermediate variables

### Error Handling

```python
def robust_chunking(text: str, config: ChunkingConfig) -> List[str]:
    """
    Robust chunking with comprehensive error handling.
    """
    try:
        # Validate input
        if not text or not text.strip():
            return []
        
        # Validate configuration
        if config.target_size <= 0:
            raise ValueError("Target size must be positive")
        
        if config.tolerance < 0:
            raise ValueError("Tolerance cannot be negative")
        
        # Perform chunking
        service = ChunkingService(config)
        chunks = service.chunk_text(text)
        
        # Validate output
        if not chunks:
            return [text]  # Return original if no chunks generated
        
        # Verify chunk integrity
        reconstructed = "".join(chunks)
        if reconstructed != text:
            print(f"Warning: Chunking integrity check failed")
        
        return chunks
        
    except Exception as e:
        print(f"Chunking error: {e}")
        # Fallback to simple character-based chunking
        return [text[i:i+config.target_size] for i in range(0, len(text), config.target_size)]
```

This comprehensive documentation provides deep technical insight into the chunking service architecture, algorithms, and best practices for implementation and optimization.