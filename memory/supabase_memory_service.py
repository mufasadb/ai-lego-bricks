from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from credentials import CredentialManager
import uuid
import os
from datetime import datetime
from supabase import create_client, Client
import logging
from .memory_service import MemoryService, Memory

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


class SupabaseMemoryService(MemoryService):
    """Memory service using Supabase with RAG (Retrieval-Augmented Generation)"""

    def __init__(
        self,
        supabase_url: str = None,
        supabase_key: str = None,
        table_name: str = "memories",
        credential_manager: Optional["CredentialManager"] = None,
    ):
        """
        Initialize Supabase memory service with pgvector support

        Args:
            supabase_url: Supabase project URL (defaults to env var)
            supabase_key: Supabase anon key (defaults to env var)
            table_name: Name of the table to store memories
            credential_manager: Optional credential manager for explicit credential handling
        """
        from credentials import default_credential_manager

        self.credential_manager = credential_manager or default_credential_manager
        self.supabase_url = supabase_url or self.credential_manager.get_credential(
            "SUPABASE_URL"
        )
        self.supabase_key = (
            supabase_key
            or self.credential_manager.get_credential("SUPABASE_ANON_KEY")
            or self.credential_manager.get_credential("SUPABASE_ACCESS_TOKEN")
        )
        self.table_name = table_name

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided via parameters or environment variables"
            )

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Initialize embedding client (use abstraction if available)
        if EMBEDDING_ABSTRACTION_AVAILABLE:
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_client = SentenceTransformerEmbeddingClient(embedding_model)
            self.embedding_dim = self.embedding_client.embedding_dimension
            logger.info(
                f"Using shared embedding client: {self.embedding_client.model_info}"
            )
        else:
            # Fallback to direct implementation
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.encoder = SentenceTransformer(embedding_model)
            test_embedding = self.encoder.encode("test")
            self.embedding_dim = len(test_embedding)
            self.embedding_client = None
            logger.info(
                f"Using direct embedding model: {embedding_model} with {self.embedding_dim} dimensions"
            )

        # Verify pgvector setup
        self._verify_pgvector_setup()

    def _verify_pgvector_setup(self):
        """Verify that pgvector is set up correctly"""
        try:
            # Check if pgvector extension exists
            result = self.supabase.rpc(
                "sql",
                {
                    "query": "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as has_vector;"
                },
            ).execute()

            if result.data and result.data[0].get("has_vector"):
                logger.info("✓ pgvector extension is installed")
            else:
                logger.warning(
                    "⚠ pgvector extension not found - vector search may not work"
                )

            # Check if memories table exists with vector column
            result = self.supabase.table(self.table_name).select("*").limit(1).execute()
            logger.info(f"✓ Table {self.table_name} exists and is accessible")

            # Check if match_memories function exists
            result = self.supabase.rpc(
                "sql",
                {
                    "query": "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'match_memories') as has_function;"
                },
            ).execute()

            if result.data and result.data[0].get("has_function"):
                logger.info("✓ match_memories function is available")
            else:
                logger.warning(
                    "⚠ match_memories function not found - falling back to text search"
                )

        except Exception as e:
            logger.error(f"Failed to verify pgvector setup: {e}")
            logger.info(
                "💡 Please run setup_supabase_pgvector.sql in your Supabase SQL editor"
            )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using embedding client"""
        if self.embedding_client:
            return self.embedding_client.generate_embedding(text)
        else:
            # Fallback to direct implementation
            embedding = self.encoder.encode(text)
            return embedding.tolist()

    def store_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in Supabase with pgvector embedding"""
        memory_id = str(uuid.uuid4())
        embedding = self._generate_embedding(content)

        # Format embedding as pgvector string
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        memory_data = {
            "id": memory_id,
            "content": content,
            "embedding": embedding_str,  # pgvector format
            "metadata": metadata or {},
            # created_at and updated_at will be set by database defaults/triggers
        }

        try:
            self.supabase.table(self.table_name).insert(memory_data).execute()
            logger.info(f"Stored memory with ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            logger.error(f"Error details: {str(e)}")
            raise RuntimeError(f"Failed to store memory: {e}")

    def retrieve_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Retrieve memories using pgvector similarity search"""
        query_embedding = self._generate_embedding(query)
        # Format query embedding as pgvector string
        query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        try:
            # Try pgvector similarity search first
            try:
                logger.debug(f"Attempting vector search for query: {query[:50]}...")
                result = self.supabase.rpc(
                    "match_memories",
                    {
                        "query_embedding": query_embedding_str,
                        "match_threshold": 0.3,  # Lower threshold = more permissive
                        "match_count": limit,
                    },
                ).execute()

                if result.data:
                    logger.info(f"Vector search returned {len(result.data)} results")
                    memories = []
                    for row in result.data:
                        # Handle different possible timestamp formats
                        timestamp_str = row.get("created_at")
                        if timestamp_str:
                            if timestamp_str.endswith("+00:00"):
                                timestamp = datetime.fromisoformat(
                                    timestamp_str.replace("+00:00", "+00:00")
                                )
                            else:
                                timestamp = datetime.fromisoformat(timestamp_str)
                        else:
                            timestamp = datetime.now()

                        memory = Memory(
                            content=row["content"],
                            metadata=row.get("metadata", {}),
                            timestamp=timestamp,
                            memory_id=row["id"],
                        )
                        memories.append(memory)

                    return memories
                else:
                    logger.info(
                        "Vector search returned no results, trying text search..."
                    )

            except Exception as vector_error:
                logger.warning(f"Vector search failed: {vector_error}")
                logger.info("Falling back to text search...")

            # Fallback to text search
            result = (
                self.supabase.table(self.table_name)
                .select("*")
                .ilike("content", f"%{query}%")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            memories = []
            for row in result.data:
                # Handle timestamp parsing
                timestamp_str = row.get("created_at")
                if timestamp_str:
                    if timestamp_str.endswith("+00:00"):
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("+00:00", "+00:00")
                        )
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = datetime.now()

                memory = Memory(
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                    timestamp=timestamp,
                    memory_id=row["id"],
                )
                memories.append(memory)

            logger.info(f"Text search returned {len(memories)} results")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []  # Return empty list instead of raising to be more resilient

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        try:
            result = (
                self.supabase.table(self.table_name)
                .select("*")
                .eq("id", memory_id)
                .execute()
            )

            if result.data:
                row = result.data[0]
                return Memory(
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                    timestamp=datetime.fromisoformat(row["created_at"]),
                    memory_id=row["id"],
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get memory by ID: {e}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            result = (
                self.supabase.table(self.table_name)
                .delete()
                .eq("id", memory_id)
                .execute()
            )
            return len(result.data) > 0

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def update_memory(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory with new embedding"""
        try:
            embedding = self._generate_embedding(content)
            # Format embedding as pgvector string
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            update_data = {
                "content": content,
                "embedding": embedding_str,  # pgvector format
                # updated_at will be set by database trigger
            }

            if metadata is not None:
                update_data["metadata"] = metadata

            result = (
                self.supabase.table(self.table_name)
                .update(update_data)
                .eq("id", memory_id)
                .execute()
            )
            return len(result.data) > 0

        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False
