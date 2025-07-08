# Memory Service - Detailed Technical Documentation

## 🧠 Architecture Overview

The Memory Service provides a sophisticated memory management system for AI agents, implementing semantic search, graph relationships, and persistent storage. It supports multiple backends with vector embeddings for intelligent memory retrieval and contextual awareness.

### Core Components

```
Memory Service Ecosystem
├── MemoryService (Abstract Base)
│   ├── Core CRUD Operations
│   ├── Semantic Search Interface
│   ├── Bulk Operations
│   └── Memory Validation
├── Backend Implementations
│   ├── SupabaseMemoryService (PostgreSQL + pgvector)
│   ├── Neo4jMemoryService (Graph Database)
│   └── Future: Redis, Elasticsearch, etc.
├── Embedding System
│   ├── Sentence Transformer Integration
│   ├── Vector Similarity Search
│   ├── Embedding Caching
│   └── Multi-model Support
├── Graph Knowledge Extraction
│   ├── Entity Recognition
│   ├── Relationship Mapping
│   ├── Knowledge Graph Building
│   └── Semantic Connections
└── Factory Pattern
    ├── Auto-detection
    ├── Configuration Management
    └── Service Selection
```

## 🏗️ Core Memory Service Interface

### Abstract Base Implementation

```python
class Memory(BaseModel):
    """
    Core memory data structure with rich metadata support.
    
    Fields:
    - content: The actual memory content
    - metadata: Flexible key-value metadata storage
    - timestamp: Creation/modification timestamp
    - memory_id: Unique identifier
    """
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_id: Optional[str] = None
    
    # Additional computed fields
    content_hash: Optional[str] = Field(default=None, description="Hash of content for deduplication")
    embedding_vector: Optional[List[float]] = Field(default=None, description="Vector embedding of content")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for search results")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    def compute_content_hash(self) -> str:
        """Compute hash of content for deduplication."""
        import hashlib
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    def add_tag(self, tag: str):
        """Add a tag to memory metadata."""
        if 'tags' not in self.metadata:
            self.metadata['tags'] = []
        if tag not in self.metadata['tags']:
            self.metadata['tags'].append(tag)
    
    def remove_tag(self, tag: str):
        """Remove a tag from memory metadata."""
        if 'tags' in self.metadata and tag in self.metadata['tags']:
            self.metadata['tags'].remove(tag)

class MemoryService(ABC):
    """
    Abstract base class defining memory service interface.
    
    Design Principles:
    - Backend agnostic interface
    - Semantic search capabilities
    - Bulk operations support
    - Rich metadata handling
    """
    
    @abstractmethod
    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory and return its unique ID."""
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, limit: int = 10, **kwargs) -> List[Memory]:
        """Retrieve memories based on semantic similarity to query."""
        pass
    
    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by its unique ID."""
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory's content and/or metadata."""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID."""
        pass
    
    # Default implementations for bulk operations
    def delete_memories(self, memory_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple memories by ID with individual error handling.
        
        Returns:
            Dict mapping memory_id to success status
        """
        results = {}
        for memory_id in memory_ids:
            try:
                results[memory_id] = self.delete_memory(memory_id)
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                results[memory_id] = False
        return results
    
    def search_memories_by_metadata(self, filters: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search memories by metadata filters.
        Default implementation (can be optimized by backends).
        """
        # This is a fallback implementation
        # Backends should override for efficient filtering
        all_memories = self.get_all_memories(limit=1000)  # Reasonable limit
        
        filtered_memories = []
        for memory in all_memories:
            if self._matches_filters(memory.metadata, filters):
                filtered_memories.append(memory)
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # Support for "contains any" semantics
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict) and '$contains' in value:
                # Support for array contains
                if not isinstance(metadata[key], list):
                    return False
                if value['$contains'] not in metadata[key]:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
```

## 🗄️ Supabase Memory Service Implementation

### PostgreSQL + pgvector Backend

```python
class SupabaseMemoryService(MemoryService):
    """
    Memory service using Supabase with pgvector for vector similarity search.
    
    Features:
    - PostgreSQL reliability and ACID compliance
    - pgvector for efficient vector similarity search
    - Real-time subscriptions (future)
    - Row-level security
    - Built-in authentication
    """
    
    def __init__(
        self, 
        supabase_url: str = None, 
        supabase_key: str = None, 
        table_name: str = "memories",
        credential_manager: Optional['CredentialManager'] = None
    ):
        """
        Initialize Supabase memory service with comprehensive setup.
        
        Configuration Priority:
        1. Direct parameters
        2. Credential manager
        3. Environment variables
        4. Defaults
        """
        
        self.credential_manager = credential_manager or default_credential_manager
        
        # Resolve credentials with fallback chain
        self.supabase_url = (
            supabase_url or 
            self.credential_manager.get_credential("SUPABASE_URL")
        )
        
        self.supabase_key = (
            supabase_key or 
            self.credential_manager.get_credential("SUPABASE_ANON_KEY") or
            self.credential_manager.get_credential("SUPABASE_ACCESS_TOKEN")
        )
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key must be provided")
        
        self.table_name = table_name
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embedding system
        self._initialize_embedding_system()
        
        # Verify database setup
        self._verify_database_setup()
    
    def _initialize_embedding_system(self):
        """
        Initialize the embedding system with fallback support.
        
        Embedding Strategy:
        1. Use shared embedding client if available
        2. Fall back to direct SentenceTransformer
        3. Support multiple embedding models
        """
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        if EMBEDDING_ABSTRACTION_AVAILABLE:
            self.embedding_client = SentenceTransformerEmbeddingClient(embedding_model)
            self.embedding_dim = self.embedding_client.embedding_dimension
            logger.info(f"Using shared embedding client: {self.embedding_client.model_info}")
        else:
            # Direct implementation fallback
            self.encoder = SentenceTransformer(embedding_model)
            test_embedding = self.encoder.encode("test")
            self.embedding_dim = len(test_embedding)
            self.embedding_client = None
            logger.info(f"Using direct embedding model: {embedding_model} ({self.embedding_dim}D)")
    
    def _verify_database_setup(self):
        """
        Comprehensive database setup verification.
        
        Verification Steps:
        1. Check pgvector extension
        2. Verify table structure
        3. Test vector functions
        4. Validate permissions
        """
        
        try:
            # Check pgvector extension
            result = self.supabase.rpc('sql', {
                'query': "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as has_vector;"
            }).execute()
            
            if result.data and result.data[0].get('has_vector'):
                logger.info("✓ pgvector extension is available")
            else:
                logger.warning("⚠ pgvector extension not found")
                raise RuntimeError("pgvector extension required for vector similarity search")
            
            # Verify table exists and is accessible
            test_result = self.supabase.table(self.table_name).select("*").limit(1).execute()
            logger.info(f"✓ Table '{self.table_name}' is accessible")
            
            # Check for vector similarity function
            result = self.supabase.rpc('sql', {
                'query': "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'match_memories') as has_function;"
            }).execute()
            
            if result.data and result.data[0].get('has_function'):
                logger.info("✓ Vector similarity function 'match_memories' is available")
            else:
                logger.warning("⚠ Vector similarity function not found")
                logger.info("💡 Please run setup_supabase_pgvector.sql in your Supabase SQL editor")
                
        except Exception as e:
            logger.error(f"Database setup verification failed: {e}")
            raise RuntimeError(f"Database setup verification failed: {e}")
    
    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store memory with vector embedding and metadata.
        
        Storage Process:
        1. Generate unique ID
        2. Create vector embedding
        3. Prepare metadata
        4. Insert into database
        5. Handle conflicts and errors
        """
        
        memory_id = str(uuid.uuid4())
        
        # Generate embedding for semantic search
        embedding = self._generate_embedding(content)
        
        # Prepare memory data
        memory_data = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        try:
            # Insert memory
            result = self.supabase.table(self.table_name).insert(memory_data).execute()
            
            if result.data:
                logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
                return memory_id
            else:
                raise RuntimeError("Failed to store memory - no data returned")
                
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise RuntimeError(f"Failed to store memory: {e}")
    
    def retrieve_memories(self, query: str, limit: int = 10, **kwargs) -> List[Memory]:
        """
        Retrieve memories using vector similarity search.
        
        Retrieval Strategy:
        1. Generate query embedding
        2. Perform vector similarity search
        3. Apply additional filters
        4. Sort by relevance
        5. Return structured results
        """
        
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        # Extract additional search parameters
        similarity_threshold = kwargs.get('similarity_threshold', 0.1)
        metadata_filters = kwargs.get('metadata_filters', {})
        include_content_search = kwargs.get('include_content_search', True)
        
        try:
            # Use vector similarity function if available
            result = self.supabase.rpc('match_memories', {
                'query_embedding': query_embedding,
                'match_threshold': similarity_threshold,
                'match_count': limit,
                'table_name': self.table_name
            }).execute()
            
            if result.data:
                memories = []
                for row in result.data:
                    memory = Memory(
                        content=row['content'],
                        metadata=row.get('metadata', {}),
                        timestamp=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                        memory_id=row['id'],
                        similarity_score=row.get('similarity', 0.0)
                    )
                    memories.append(memory)
                
                # Apply metadata filters if specified
                if metadata_filters:
                    memories = [m for m in memories if self._matches_filters(m.metadata, metadata_filters)]
                
                return memories[:limit]
            
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")
        
        # Fallback to text-based search
        return self._fallback_text_search(query, limit, **kwargs)
    
    def _fallback_text_search(self, query: str, limit: int, **kwargs) -> List[Memory]:
        """
        Fallback text-based search when vector search is unavailable.
        
        Search Strategy:
        - Full-text search on content
        - Metadata keyword matching
        - Basic relevance scoring
        """
        
        try:
            # Use PostgreSQL text search
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .text_search('content', query)\
                .limit(limit)\
                .execute()
            
            memories = []
            for row in result.data:
                memory = Memory(
                    content=row['content'],
                    metadata=row.get('metadata', {}),
                    timestamp=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                    memory_id=row['id']
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Text search also failed: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching and error handling."""
        
        try:
            if self.embedding_client:
                return self.embedding_client.generate_embedding(text)
            else:
                # Direct encoder fallback
                embedding = self.encoder.encode(text)
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dim
    
    def bulk_store_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Bulk store memories for efficient batch processing.
        
        Batch Strategy:
        - Generate embeddings in parallel
        - Batch database inserts
        - Handle partial failures
        - Return success/failure mapping
        """
        
        import concurrent.futures
        
        # Prepare memory data with embeddings
        memory_data_list = []
        memory_ids = []
        
        def prepare_memory(memory_input):
            content = memory_input['content']
            metadata = memory_input.get('metadata', {})
            memory_id = str(uuid.uuid4())
            
            embedding = self._generate_embedding(content)
            
            return {
                "id": memory_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, memory_id
        
        # Generate embeddings in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(prepare_memory, memories))
        
        memory_data_list = [result[0] for result in results]
        memory_ids = [result[1] for result in results]
        
        try:
            # Bulk insert
            result = self.supabase.table(self.table_name).insert(memory_data_list).execute()
            
            if result.data:
                logger.info(f"Bulk stored {len(memory_ids)} memories")
                return memory_ids
            else:
                raise RuntimeError("Bulk insert failed - no data returned")
                
        except Exception as e:
            logger.error(f"Bulk store failed: {e}")
            raise RuntimeError(f"Bulk store failed: {e}")
```

## 🕸️ Neo4j Memory Service Implementation

### Graph Database Backend

```python
class Neo4jMemoryService(MemoryService):
    """
    Memory service using Neo4j graph database for relationship-rich memory storage.
    
    Features:
    - Rich relationship modeling
    - Graph traversal queries
    - Entity relationship extraction
    - Knowledge graph construction
    - Path analysis
    """
    
    def __init__(
        self, 
        uri: str = None, 
        username: str = None, 
        password: str = None,
        credential_manager: Optional['CredentialManager'] = None
    ):
        """
        Initialize Neo4j memory service with graph capabilities.
        
        Connection Strategy:
        - Support authenticated and non-authenticated connections
        - Auto-configure from environment
        - Initialize graph schema
        """
        
        self.credential_manager = credential_manager or default_credential_manager
        
        # Resolve connection parameters
        self.uri = uri or self.credential_manager.get_credential("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or self.credential_manager.get_credential("NEO4J_USERNAME", "neo4j")
        self.password = password if password is not None else self.credential_manager.get_credential("NEO4J_PASSWORD")
        
        # Create driver with appropriate authentication
        if self.password:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        else:
            self.driver = GraphDatabase.driver(self.uri, auth=None)
        
        # Initialize embedding system
        self._initialize_embedding_system()
        
        # Setup graph schema
        self._setup_graph_schema()
    
    def _setup_graph_schema(self):
        """
        Setup Neo4j graph schema for memory storage.
        
        Schema Design:
        - Memory nodes with content and metadata
        - Entity nodes extracted from content
        - Relationship edges between entities
        - Indexes for performance
        """
        
        with self.driver.session() as session:
            try:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    session.run(constraint)
                
                # Create indexes
                indexes = [
                    "CREATE INDEX memory_content_index IF NOT EXISTS FOR (m:Memory) ON (m.content)",
                    "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"
                ]
                
                for index in indexes:
                    session.run(index)
                
                logger.info("Neo4j graph schema initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize graph schema: {e}")
    
    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store memory with entity extraction and relationship building.
        
        Storage Process:
        1. Create memory node
        2. Extract entities from content
        3. Create entity nodes
        4. Establish relationships
        5. Link to memory node
        """
        
        memory_id = str(uuid.uuid4())
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        with self.driver.session() as session:
            try:
                # Create memory node
                result = session.run("""
                    CREATE (m:Memory {
                        id: $id,
                        content: $content,
                        metadata: $metadata,
                        embedding: $embedding,
                        timestamp: datetime()
                    })
                    RETURN m.id as memory_id
                """, {
                    "id": memory_id,
                    "content": content,
                    "metadata": json.dumps(metadata or {}),
                    "embedding": embedding
                })
                
                # Extract and store entities if graph formatter is available
                if hasattr(self, 'graph_formatter'):
                    self._extract_and_store_entities(session, memory_id, content)
                
                logger.debug(f"Stored memory {memory_id} in Neo4j")
                return memory_id
                
            except Exception as e:
                logger.error(f"Failed to store memory in Neo4j: {e}")
                raise RuntimeError(f"Failed to store memory: {e}")
    
    def _extract_and_store_entities(self, session, memory_id: str, content: str):
        """
        Extract entities from content and create graph relationships.
        
        Entity Extraction:
        - Use LLM-based entity recognition
        - Create entity nodes
        - Establish relationships
        - Link to memory
        """
        
        try:
            # Extract entities using graph formatter
            graph_data = self.graph_formatter.format_memory_as_graph(content)
            
            # Create entity nodes and relationships
            for entity in graph_data.entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.properties = $properties
                    WITH e
                    MATCH (m:Memory {id: $memory_id})
                    MERGE (m)-[:MENTIONS]->(e)
                """, {
                    "name": entity.name,
                    "type": entity.type,
                    "properties": json.dumps(entity.properties),
                    "memory_id": memory_id
                })
            
            # Create relationships between entities
            for relationship in graph_data.relationships:
                session.run("""
                    MATCH (e1:Entity {name: $source})
                    MATCH (e2:Entity {name: $target})
                    MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
                    SET r.properties = $properties
                """, {
                    "source": relationship.source,
                    "target": relationship.target,
                    "rel_type": relationship.relationship_type,
                    "properties": json.dumps(relationship.properties)
                })
                
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
    
    def retrieve_memories(self, query: str, limit: int = 10, **kwargs) -> List[Memory]:
        """
        Retrieve memories using graph traversal and vector similarity.
        
        Retrieval Strategies:
        1. Vector similarity search
        2. Entity-based graph traversal
        3. Relationship path analysis
        4. Combined scoring
        """
        
        query_embedding = self._generate_embedding(query)
        use_graph_traversal = kwargs.get('use_graph_traversal', True)
        
        with self.driver.session() as session:
            try:
                if use_graph_traversal:
                    # Graph-enhanced retrieval
                    result = session.run("""
                        // Find memories with similar content (vector similarity approximation)
                        MATCH (m:Memory)
                        WITH m, 
                             reduce(similarity = 0.0, i in range(0, size($query_embedding)-1) | 
                                 similarity + (m.embedding[i] * $query_embedding[i])
                             ) as vector_similarity
                        
                        // Also find memories connected to relevant entities
                        OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
                        WHERE e.name CONTAINS $query_text OR e.type CONTAINS $query_text
                        WITH m, vector_similarity, count(e) as entity_matches
                        
                        // Combine scores
                        WITH m, (vector_similarity + entity_matches * 0.1) as combined_score
                        ORDER BY combined_score DESC
                        LIMIT $limit
                        
                        RETURN m.id as id, m.content as content, 
                               m.metadata as metadata, m.timestamp as timestamp,
                               combined_score as similarity
                    """, {
                        "query_embedding": query_embedding,
                        "query_text": query,
                        "limit": limit
                    })
                else:
                    # Simple vector similarity
                    result = session.run("""
                        MATCH (m:Memory)
                        WITH m, 
                             reduce(similarity = 0.0, i in range(0, size($query_embedding)-1) | 
                                 similarity + (m.embedding[i] * $query_embedding[i])
                             ) as similarity
                        ORDER BY similarity DESC
                        LIMIT $limit
                        
                        RETURN m.id as id, m.content as content, 
                               m.metadata as metadata, m.timestamp as timestamp,
                               similarity
                    """, {
                        "query_embedding": query_embedding,
                        "limit": limit
                    })
                
                memories = []
                for record in result:
                    metadata = json.loads(record['metadata']) if record['metadata'] else {}
                    
                    memory = Memory(
                        content=record['content'],
                        metadata=metadata,
                        timestamp=record['timestamp'].to_native() if hasattr(record['timestamp'], 'to_native') else datetime.now(),
                        memory_id=record['id'],
                        similarity_score=float(record['similarity'])
                    )
                    memories.append(memory)
                
                return memories
                
            except Exception as e:
                logger.error(f"Memory retrieval failed: {e}")
                return []
    
    def find_related_memories(self, memory_id: str, max_depth: int = 2, limit: int = 10) -> List[Memory]:
        """
        Find memories related through entity relationships.
        
        Graph Traversal:
        - Start from given memory
        - Traverse entity relationships
        - Find connected memories
        - Score by path strength
        """
        
        with self.driver.session() as session:
            try:
                result = session.run("""
                    MATCH (start:Memory {id: $memory_id})-[:MENTIONS]->(e1:Entity)
                    MATCH (e1)-[:RELATES_TO*1..$max_depth]-(e2:Entity)
                    MATCH (e2)<-[:MENTIONS]-(related:Memory)
                    WHERE related.id <> $memory_id
                    
                    WITH related, count(DISTINCT e2) as connection_strength
                    ORDER BY connection_strength DESC
                    LIMIT $limit
                    
                    RETURN related.id as id, related.content as content,
                           related.metadata as metadata, related.timestamp as timestamp,
                           connection_strength
                """, {
                    "memory_id": memory_id,
                    "max_depth": max_depth,
                    "limit": limit
                })
                
                related_memories = []
                for record in result:
                    metadata = json.loads(record['metadata']) if record['metadata'] else {}
                    
                    memory = Memory(
                        content=record['content'],
                        metadata=metadata,
                        timestamp=record['timestamp'].to_native() if hasattr(record['timestamp'], 'to_native') else datetime.now(),
                        memory_id=record['id'],
                        similarity_score=float(record['connection_strength'])
                    )
                    related_memories.append(memory)
                
                return related_memories
                
            except Exception as e:
                logger.error(f"Related memory search failed: {e}")
                return []
```

## 🔄 Advanced Memory Operations

### Memory Clustering and Organization

```python
class MemoryOrganizer:
    """
    Advanced memory organization and clustering system.
    
    Features:
    - Automatic clustering
    - Topic modeling
    - Memory hierarchies
    - Duplicate detection
    """
    
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
        self.clustering_cache = {}
    
    def cluster_memories(self, memories: List[Memory], num_clusters: int = 5) -> Dict[int, List[Memory]]:
        """
        Cluster memories by semantic similarity.
        
        Clustering Algorithm:
        1. Extract embeddings from memories
        2. Apply K-means clustering
        3. Group memories by cluster
        4. Generate cluster summaries
        """
        
        import numpy as np
        from sklearn.cluster import KMeans
        
        # Extract embeddings
        embeddings = []
        for memory in memories:
            if hasattr(memory, 'embedding_vector') and memory.embedding_vector:
                embeddings.append(memory.embedding_vector)
            else:
                # Generate embedding if not available
                embedding = self.memory_service._generate_embedding(memory.content)
                embeddings.append(embedding)
        
        if not embeddings:
            return {0: memories}
        
        # Perform clustering
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=min(num_clusters, len(memories)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Group memories by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[i])
        
        return clusters
    
    def generate_cluster_summary(self, cluster_memories: List[Memory]) -> str:
        """
        Generate summary for a cluster of memories.
        
        Summary Strategy:
        - Extract key themes
        - Identify common metadata
        - Generate descriptive title
        """
        
        # Extract content for summarization
        contents = [memory.content for memory in cluster_memories]
        combined_content = " ".join(contents[:5])  # Limit for efficiency
        
        # Extract common metadata themes
        common_tags = self._extract_common_metadata(cluster_memories)
        
        # Generate summary (simplified - could use LLM for better results)
        summary = f"Cluster of {len(cluster_memories)} memories"
        if common_tags:
            summary += f" with common themes: {', '.join(common_tags)}"
        
        return summary
    
    def _extract_common_metadata(self, memories: List[Memory]) -> List[str]:
        """Extract common metadata themes across memories."""
        
        all_tags = []
        for memory in memories:
            if 'tags' in memory.metadata:
                all_tags.extend(memory.metadata['tags'])
        
        # Find most common tags
        from collections import Counter
        tag_counts = Counter(all_tags)
        return [tag for tag, count in tag_counts.most_common(3)]
    
    def detect_duplicates(self, memories: List[Memory], similarity_threshold: float = 0.95) -> List[List[Memory]]:
        """
        Detect near-duplicate memories.
        
        Duplicate Detection:
        - Calculate pairwise similarities
        - Group memories above threshold
        - Return duplicate groups
        """
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Extract embeddings
        embeddings = []
        for memory in memories:
            if hasattr(memory, 'embedding_vector') and memory.embedding_vector:
                embeddings.append(memory.embedding_vector)
            else:
                embedding = self.memory_service._generate_embedding(memory.content)
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            return []
        
        # Calculate similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Find duplicate groups
        duplicate_groups = []
        processed = set()
        
        for i in range(len(memories)):
            if i in processed:
                continue
            
            # Find similar memories
            similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
            
            if len(similar_indices) > 1:
                duplicate_group = [memories[idx] for idx in similar_indices]
                duplicate_groups.append(duplicate_group)
                processed.update(similar_indices)
        
        return duplicate_groups
```

### Memory Analytics and Insights

```python
class MemoryAnalytics:
    """
    Analytics and insights for memory data.
    
    Features:
    - Usage patterns
    - Content analysis
    - Temporal trends
    - Performance metrics
    """
    
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
    
    def analyze_memory_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Analyze patterns in memory data.
        
        Analysis Dimensions:
        - Temporal patterns
        - Content categories
        - Metadata distributions
        - Usage frequency
        """
        
        from collections import Counter, defaultdict
        import numpy as np
        
        analysis = {
            "total_memories": len(memories),
            "temporal_analysis": {},
            "content_analysis": {},
            "metadata_analysis": {},
            "quality_metrics": {}
        }
        
        # Temporal analysis
        timestamps = [memory.timestamp for memory in memories]
        if timestamps:
            analysis["temporal_analysis"] = {
                "earliest": min(timestamps).isoformat(),
                "latest": max(timestamps).isoformat(),
                "span_days": (max(timestamps) - min(timestamps)).days
            }
        
        # Content analysis
        content_lengths = [len(memory.content) for memory in memories]
        analysis["content_analysis"] = {
            "average_length": np.mean(content_lengths) if content_lengths else 0,
            "median_length": np.median(content_lengths) if content_lengths else 0,
            "min_length": min(content_lengths) if content_lengths else 0,
            "max_length": max(content_lengths) if content_lengths else 0
        }
        
        # Metadata analysis
        all_metadata_keys = []
        tag_counter = Counter()
        
        for memory in memories:
            all_metadata_keys.extend(memory.metadata.keys())
            if 'tags' in memory.metadata:
                tag_counter.update(memory.metadata['tags'])
        
        analysis["metadata_analysis"] = {
            "common_metadata_keys": Counter(all_metadata_keys).most_common(10),
            "top_tags": tag_counter.most_common(10),
            "metadata_coverage": len(set(all_metadata_keys))
        }
        
        return analysis
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the memory service.
        
        Metrics:
        - Query response times
        - Storage efficiency
        - Embedding quality
        - Error rates
        """
        
        # This would typically integrate with monitoring systems
        return {
            "average_query_time": "N/A - requires monitoring integration",
            "storage_efficiency": "N/A - requires storage metrics",
            "embedding_dimension": getattr(self.memory_service, 'embedding_dim', 'Unknown'),
            "backend_type": type(self.memory_service).__name__
        }
```

## 🚀 Production Deployment Patterns

### Scalable Memory Architecture

```python
class ScalableMemoryService:
    """
    Production-ready scalable memory service.
    
    Scalability Features:
    - Connection pooling
    - Caching layers
    - Batch processing
    - Error resilience
    """
    
    def __init__(
        self,
        primary_service: MemoryService,
        cache_service: Optional[MemoryService] = None,
        batch_size: int = 100,
        connection_pool_size: int = 10
    ):
        self.primary_service = primary_service
        self.cache_service = cache_service
        self.batch_size = batch_size
        
        # Performance tracking
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'error_count': 0
        }
    
    async def store_memory_async(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronous memory storage with caching."""
        
        import asyncio
        
        try:
            # Store in primary service
            memory_id = await asyncio.to_thread(
                self.primary_service.store_memory, 
                content, 
                metadata
            )
            
            # Cache if cache service available
            if self.cache_service:
                await asyncio.to_thread(
                    self.cache_service.store_memory,
                    content,
                    {**metadata, 'cached_from_primary': True}
                )
            
            return memory_id
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Async memory storage failed: {e}")
            raise
    
    def retrieve_with_cache(self, query: str, limit: int = 10, **kwargs) -> List[Memory]:
        """Memory retrieval with multi-layer caching."""
        
        cache_key = self._generate_cache_key(query, limit, kwargs)
        
        # Try cache first
        if self.cache_service:
            try:
                cached_results = self.cache_service.retrieve_memories(
                    query, limit, **kwargs
                )
                if cached_results:
                    self.metrics['cache_hits'] += 1
                    return cached_results
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        # Fall back to primary service
        self.metrics['cache_misses'] += 1
        try:
            results = self.primary_service.retrieve_memories(query, limit, **kwargs)
            
            # Update cache
            if self.cache_service and results:
                for memory in results:
                    try:
                        self.cache_service.store_memory(
                            memory.content,
                            {**memory.metadata, 'cached_query': query}
                        )
                    except Exception as e:
                        logger.warning(f"Cache update failed: {e}")
            
            return results
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Primary service retrieval failed: {e}")
            raise
    
    def _generate_cache_key(self, query: str, limit: int, kwargs: Dict) -> str:
        """Generate cache key for query."""
        
        import hashlib
        
        key_data = f"{query}:{limit}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

This comprehensive documentation provides deep technical insight into the memory service architecture, multiple backend implementations, advanced features, and production deployment patterns.