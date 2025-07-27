from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from credentials import CredentialManager
import uuid
import os
import json
from datetime import datetime
from neo4j import GraphDatabase
import numpy as np
import logging
from .memory_service import MemoryService, Memory
from .graph_formatter_service import GraphMemoryFormat

# Import shared embedding service
try:
    from llm.embedding_client import SentenceTransformerEmbeddingClient
    from llm.llm_types import EmbeddingClient

    EMBEDDING_ABSTRACTION_AVAILABLE = True
except ImportError:
    # Fallback to direct sentence_transformers import
    from sentence_transformers import SentenceTransformer

    EMBEDDING_ABSTRACTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class Neo4jMemoryService(MemoryService):
    """Memory service using Neo4j graph database"""

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        credential_manager: Optional["CredentialManager"] = None,
    ):
        """
        Initialize Neo4j memory service

        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Neo4j username (optional, defaults to neo4j)
            password: Neo4j password (optional, uses no-auth if not provided)
            credential_manager: Optional credential manager for explicit credential handling
        """
        try:
            from credentials import default_credential_manager
        except ImportError:
            # Fallback for when running as standalone
            try:
                from credentials import default_credential_manager
            except ImportError:
                from credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.uri = uri or self.credential_manager.get_credential(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        self.username = username or self.credential_manager.get_credential(
            "NEO4J_USERNAME", "neo4j"
        )
        self.password = (
            password
            if password is not None
            else self.credential_manager.get_credential("NEO4J_PASSWORD")
        )

        # Create driver (no-auth if no password)
        if self.password:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
        else:
            self.driver = GraphDatabase.driver(self.uri, auth=None)

        # Initialize embedding client (use abstraction if available)
        if EMBEDDING_ABSTRACTION_AVAILABLE:
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_client = SentenceTransformerEmbeddingClient(embedding_model)
            logger.info(
                f"Using shared embedding client: {self.embedding_client.model_info}"
            )
        else:
            # Fallback to direct implementation
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_client = None
            logger.info(f"Using direct embedding model: {embedding_model}")

        # Initialize database constraints and indexes
        self._ensure_constraints()

    def _ensure_constraints(self):
        """Create necessary constraints and indexes"""
        with self.driver.session() as session:
            try:
                # Create unique constraint on memory ID
                session.run(
                    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE"
                )

                # Create index on content for text search
                session.run(
                    "CREATE INDEX memory_content_index IF NOT EXISTS FOR (m:Memory) ON (m.content)"
                )

                # Create index on timestamp
                session.run(
                    "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)"
                )

                logger.info("Neo4j constraints and indexes created successfully")

            except Exception as e:
                logger.warning(f"Failed to create constraints/indexes: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using embedding client"""
        if self.embedding_client:
            return self.embedding_client.generate_embedding(text)
        else:
            # Fallback to direct implementation
            embedding = self.encoder.encode(text)
            return embedding.tolist()

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def store_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in Neo4j with vector embedding"""
        memory_id = str(uuid.uuid4())
        embedding = self._generate_embedding(content)
        timestamp = datetime.now().isoformat()

        with self.driver.session() as session:
            try:
                # Store memory node
                query = """
                CREATE (m:Memory {
                    id: $id,
                    content: $content,
                    embedding: $embedding,
                    metadata_json: $metadata,
                    timestamp: $timestamp
                })
                RETURN m.id as id
                """

                # Serialize metadata to JSON string
                metadata_json = json.dumps(metadata or {})

                session.run(
                    query,
                    {
                        "id": memory_id,
                        "content": content,
                        "embedding": embedding,
                        "metadata": metadata_json,
                        "timestamp": timestamp,
                    },
                )

                # Extract entities from metadata and create relationships
                if metadata:
                    self._create_entity_relationships(session, memory_id, metadata)

                logger.info(f"Stored memory with ID: {memory_id}")
                return memory_id

            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                raise RuntimeError(f"Failed to store memory: {e}")

    def _create_entity_relationships(
        self, session, memory_id: str, metadata: Dict[str, Any]
    ):
        """Create relationships between memory and entities found in metadata"""
        for key, value in metadata.items():
            if isinstance(value, str) and value.strip():
                # Create entity node and relationship
                entity_query = """
                MERGE (e:Entity {name: $name, type: $type})
                WITH e
                MATCH (m:Memory {id: $memory_id})
                MERGE (m)-[:MENTIONS]->(e)
                """

                session.run(
                    entity_query, {"name": value, "type": key, "memory_id": memory_id}
                )

    def retrieve_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Retrieve memories using vector similarity and graph traversal"""
        query_embedding = self._generate_embedding(query)

        with self.driver.session() as session:
            try:
                # Get all memories and calculate similarities
                fetch_query = """
                MATCH (m:Memory)
                RETURN m.id as id, m.content as content, m.embedding as embedding, 
                       COALESCE(m.metadata_json, m.metadata) as metadata, m.timestamp as timestamp
                """

                result = session.run(fetch_query)

                memories_with_similarity = []
                for record in result:
                    memory_embedding = record["embedding"]
                    similarity = self._calculate_similarity(
                        query_embedding, memory_embedding
                    )

                    # Handle both old and new metadata formats
                    metadata_raw = record["metadata"]
                    if isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    else:
                        metadata = metadata_raw if metadata_raw else {}

                    memory = Memory(
                        content=record["content"],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        memory_id=record["id"],
                    )

                    memories_with_similarity.append((memory, similarity))

                # Sort by similarity and return top results
                memories_with_similarity.sort(key=lambda x: x[1], reverse=True)

                return [memory for memory, _ in memories_with_similarity[:limit]]

            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                # Fall back to text search
                return self._text_search_fallback(session, query, limit)

    def _text_search_fallback(self, session, query: str, limit: int) -> List[Memory]:
        """Fallback text search when vector search fails"""
        try:
            text_query = """
            MATCH (m:Memory)
            WHERE m.content CONTAINS $query
            RETURN m.id as id, m.content as content, COALESCE(m.metadata_json, m.metadata) as metadata, 
                   m.timestamp as timestamp
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """

            result = session.run(text_query, {"query": query, "limit": limit})

            memories = []
            for record in result:
                # Handle both old and new metadata formats
                metadata_raw = record["metadata"]
                if isinstance(metadata_raw, str):
                    metadata = json.loads(metadata_raw) if metadata_raw else {}
                else:
                    metadata = metadata_raw if metadata_raw else {}

                memory = Memory(
                    content=record["content"],
                    metadata=metadata,
                    timestamp=datetime.fromisoformat(record["timestamp"]),
                    memory_id=record["id"],
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Text search fallback failed: {e}")
            return []

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (m:Memory {id: $id})
                RETURN m.content as content, COALESCE(m.metadata_json, m.metadata) as metadata, m.timestamp as timestamp
                """

                result = session.run(query, {"id": memory_id})
                record = result.single()

                if record:
                    # Handle both old and new metadata formats
                    metadata_raw = record["metadata"]
                    if isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    else:
                        metadata = metadata_raw if metadata_raw else {}

                    return Memory(
                        content=record["content"],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        memory_id=memory_id,
                    )
                return None

            except Exception as e:
                logger.error(f"Failed to get memory by ID: {e}")
                return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (m:Memory {id: $id})
                DETACH DELETE m
                RETURN count(m) as deleted_count
                """

                result = session.run(query, {"id": memory_id})
                record = result.single()
                return record["deleted_count"] > 0

            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
                return False

    def delete_memories(self, memory_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple memories efficiently in a single transaction

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            Dict mapping memory_id -> success (True/False)
        """
        if not memory_ids:
            return {}

        with self.driver.session() as session:
            try:
                # Bulk delete in a single query
                query = """
                UNWIND $ids as memory_id
                OPTIONAL MATCH (m:Memory {id: memory_id})
                WITH memory_id, m
                DETACH DELETE m
                RETURN memory_id, CASE WHEN m IS NOT NULL THEN 1 ELSE 0 END as deleted
                """

                result = session.run(query, {"ids": memory_ids})

                # Build results dict
                results = {}
                for record in result:
                    memory_id = record["memory_id"]
                    deleted = record["deleted"] > 0
                    results[memory_id] = deleted

                # Make sure we have results for all requested IDs
                for memory_id in memory_ids:
                    if memory_id not in results:
                        results[memory_id] = False

                successful_deletes = sum(1 for success in results.values() if success)
                logger.info(
                    f"Bulk deleted {successful_deletes}/{len(memory_ids)} memories"
                )

                return results

            except Exception as e:
                logger.error(f"Failed to bulk delete memories: {e}")
                # Fallback to individual deletes
                return super().delete_memories(memory_ids)

    def update_memory(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory"""
        with self.driver.session() as session:
            try:
                embedding = self._generate_embedding(content)
                timestamp = datetime.now().isoformat()

                # Update memory node
                update_query = """
                MATCH (m:Memory {id: $id})
                SET m.content = $content,
                    m.embedding = $embedding,
                    m.timestamp = $timestamp
                """

                params = {
                    "id": memory_id,
                    "content": content,
                    "embedding": embedding,
                    "timestamp": timestamp,
                }

                if metadata is not None:
                    update_query += ", m.metadata_json = $metadata"
                    params["metadata"] = json.dumps(metadata)

                update_query += " RETURN count(m) as updated_count"

                result = session.run(update_query, params)
                record = result.single()

                # Update entity relationships if metadata changed
                if metadata is not None:
                    # Remove old relationships
                    session.run(
                        "MATCH (m:Memory {id: $id})-[r:MENTIONS]->() DELETE r",
                        {"id": memory_id},
                    )
                    # Create new relationships
                    self._create_entity_relationships(session, memory_id, metadata)

                return record["updated_count"] > 0

            except Exception as e:
                logger.error(f"Failed to update memory: {e}")
                return False

    def get_related_memories(self, memory_id: str, limit: int = 5) -> List[Memory]:
        """Get memories related through entity relationships"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (m1:Memory {id: $id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(m2:Memory)
                WHERE m1 <> m2
                RETURN DISTINCT m2.id as id, m2.content as content, COALESCE(m2.metadata_json, m2.metadata) as metadata, 
                       m2.timestamp as timestamp
                ORDER BY m2.timestamp DESC
                LIMIT $limit
                """

                result = session.run(query, {"id": memory_id, "limit": limit})

                memories = []
                for record in result:
                    # Handle both old and new metadata formats
                    metadata_raw = record["metadata"]
                    if isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    else:
                        metadata = metadata_raw if metadata_raw else {}

                    memory = Memory(
                        content=record["content"],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        memory_id=record["id"],
                    )
                    memories.append(memory)

                return memories

            except Exception as e:
                logger.error(f"Failed to get related memories: {e}")
                return []

    def store_graph_memory(
        self,
        graph_format: GraphMemoryFormat,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a graph-formatted memory with enhanced entity and relationship data

        Args:
            graph_format: GraphMemoryFormat object with entities and relationships
            additional_metadata: Optional additional metadata to merge

        Returns:
            Memory ID of the stored memory
        """
        memory_id = str(uuid.uuid4())

        # Combine metadata
        metadata = graph_format.extraction_metadata.copy()
        if additional_metadata:
            metadata.update(additional_metadata)

        # Add graph-specific metadata
        metadata.update(
            {
                "graph_formatted": True,
                "entity_count": len(graph_format.entities),
                "relationship_count": len(graph_format.relationships),
                "summary": graph_format.summary,
            }
        )

        # Generate embedding for the original content
        content_embedding = self._generate_embedding(graph_format.original_content)

        with self.driver.session() as session:
            try:
                # Store the memory node
                memory_query = """
                CREATE (m:Memory {
                    id: $id,
                    content: $content,
                    embedding: $embedding,
                    metadata_json: $metadata,
                    timestamp: $timestamp,
                    summary: $summary,
                    graph_formatted: true
                })
                """

                # Serialize metadata to JSON string
                metadata_json = json.dumps(metadata)

                session.run(
                    memory_query,
                    {
                        "id": memory_id,
                        "content": graph_format.original_content,
                        "embedding": content_embedding,
                        "metadata": metadata_json,
                        "timestamp": graph_format.timestamp.isoformat(),
                        "summary": graph_format.summary,
                    },
                )

                # Create enhanced entities and relationships
                self._create_graph_entities_and_relationships(
                    session, memory_id, graph_format
                )

                logger.info(f"Stored graph-formatted memory with ID: {memory_id}")
                return memory_id

            except Exception as e:
                logger.error(f"Failed to store graph memory: {e}")
                raise RuntimeError(f"Failed to store graph memory: {e}")

    def _create_graph_entities_and_relationships(
        self, session, memory_id: str, graph_format: GraphMemoryFormat
    ):
        """Create enhanced entities and relationships from graph format"""

        # Create entity nodes with properties
        for entity in graph_format.entities:
            entity_query = """
            MERGE (e:Entity {name: $name, type: $type})
            SET e.properties_json = $properties,
                e.last_updated = $timestamp
            WITH e
            MATCH (m:Memory {id: $memory_id})
            MERGE (m)-[:CONTAINS_ENTITY {
                confidence: 1.0,
                extraction_method: $extraction_method
            }]->(e)
            """

            # Serialize properties to JSON string
            properties_json = (
                json.dumps(entity.properties) if entity.properties else "{}"
            )

            session.run(
                entity_query,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "properties": properties_json,
                    "timestamp": datetime.now().isoformat(),
                    "memory_id": memory_id,
                    "extraction_method": graph_format.extraction_metadata.get(
                        "extraction_method", "unknown"
                    ),
                },
            )

        # Create relationships between entities
        for relationship in graph_format.relationships:
            relationship_query = """
            MATCH (source:Entity {name: $source_name})
            MATCH (target:Entity {name: $target_name})
            MATCH (m:Memory {id: $memory_id})
            MERGE (source)-[r:GRAPH_RELATIONSHIP {
                type: $rel_type,
                confidence: $confidence,
                properties_json: $properties,
                memory_id: $memory_id,
                created_at: $timestamp
            }]->(target)
            SET r.from_memory = $memory_id
            """

            # Serialize properties to JSON string
            properties_json = (
                json.dumps(relationship.properties) if relationship.properties else "{}"
            )

            session.run(
                relationship_query,
                {
                    "source_name": relationship.source_entity,
                    "target_name": relationship.target_entity,
                    "rel_type": relationship.relationship_type,
                    "confidence": relationship.confidence,
                    "properties": properties_json,
                    "memory_id": memory_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def retrieve_memories_by_entity(
        self, entity_name: str, limit: int = 10
    ) -> List[Memory]:
        """Retrieve memories that contain a specific entity"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (e:Entity {name: $entity_name})<-[:CONTAINS_ENTITY]-(m:Memory)
                RETURN DISTINCT m.id as id, m.content as content, COALESCE(m.metadata_json, m.metadata) as metadata, 
                       m.timestamp as timestamp, m.summary as summary
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """

                result = session.run(
                    query, {"entity_name": entity_name, "limit": limit}
                )

                memories = []
                for record in result:
                    # Handle both old and new metadata formats
                    metadata_raw = record["metadata"]
                    if isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    else:
                        metadata = metadata_raw if metadata_raw else {}

                    memory = Memory(
                        content=record["content"],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        memory_id=record["id"],
                    )
                    memories.append(memory)

                return memories

            except Exception as e:
                logger.error(f"Failed to retrieve memories by entity: {e}")
                return []

    def retrieve_memories_by_relationship_type(
        self, relationship_type: str, limit: int = 10
    ) -> List[Memory]:
        """Retrieve memories that contain a specific type of relationship"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (source)-[r:GRAPH_RELATIONSHIP {type: $rel_type}]->(target)
                MATCH (m:Memory {id: r.memory_id})
                RETURN DISTINCT m.id as id, m.content as content, COALESCE(m.metadata_json, m.metadata) as metadata, 
                       m.timestamp as timestamp, m.summary as summary
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """

                result = session.run(
                    query, {"rel_type": relationship_type, "limit": limit}
                )

                memories = []
                for record in result:
                    # Handle both old and new metadata formats
                    metadata_raw = record["metadata"]
                    if isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    else:
                        metadata = metadata_raw if metadata_raw else {}

                    memory = Memory(
                        content=record["content"],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        memory_id=record["id"],
                    )
                    memories.append(memory)

                return memories

            except Exception as e:
                logger.error(f"Failed to retrieve memories by relationship type: {e}")
                return []

    def get_entity_relationships(
        self, entity_name: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all relationships for a specific entity"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (source:Entity {name: $entity_name})-[r:GRAPH_RELATIONSHIP]->(target:Entity)
                RETURN source.name as source_entity,
                       target.name as target_entity,
                       r.type as relationship_type,
                       r.confidence as confidence,
                       COALESCE(r.properties_json, r.properties) as properties,
                       r.memory_id as memory_id
                ORDER BY r.confidence DESC
                LIMIT $limit
                """

                result = session.run(
                    query, {"entity_name": entity_name, "limit": limit}
                )

                relationships = []
                for record in result:
                    # Handle both old and new properties formats
                    properties_raw = record["properties"]
                    if isinstance(properties_raw, str):
                        properties = (
                            json.loads(properties_raw) if properties_raw else {}
                        )
                    else:
                        properties = properties_raw if properties_raw else {}

                    relationship = {
                        "source_entity": record["source_entity"],
                        "target_entity": record["target_entity"],
                        "relationship_type": record["relationship_type"],
                        "confidence": record["confidence"],
                        "properties": properties,
                        "memory_id": record["memory_id"],
                    }
                    relationships.append(relationship)

                return relationships

            except Exception as e:
                logger.error(f"Failed to get entity relationships: {e}")
                return []

    def close(self):
        """Close the database connection"""
        self.driver.close()
