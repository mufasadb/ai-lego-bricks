"""
Tests for memory services including store, retrieve, and delete operations.

Converted from test/ directory files to pytest format.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from memory.memory_factory import create_memory_service
from memory.memory_service import MemoryService
from memory.supabase_memory_service import SupabaseMemoryService
from memory.neo4j_memory_service import Neo4jMemoryService


class TestMemoryServices:
    """Test suite for memory services."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_memories = [
            {"content": "Test memory 1 - sample content", "metadata": {"test": "memory1"}},
            {"content": "Test memory 2 - different content", "metadata": {"test": "memory2"}},
            {"content": "Test memory 3 - for deletion", "metadata": {"test": "delete_me"}},
        ]
        
    def test_memory_service_creation_auto(self):
        """Test automatic memory service creation."""
        service = create_memory_service("auto")
        assert service is not None
        assert isinstance(service, (SupabaseMemoryService, Neo4jMemoryService))
    
    def test_memory_service_creation_supabase(self):
        """Test Supabase memory service creation."""
        with patch('memory.memory_factory.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key: {
                'SUPABASE_URL': 'https://test.supabase.co',
                'SUPABASE_ANON_KEY': 'test_key'
            }.get(key)
            
            service = create_memory_service("supabase")
            assert service is not None
            assert isinstance(service, SupabaseMemoryService)
    
    def test_memory_service_creation_neo4j(self):
        """Test Neo4j memory service creation."""
        with patch('memory.memory_factory.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key: {
                'NEO4J_URI': 'bolt://localhost:7687',
                'NEO4J_USERNAME': 'neo4j',
                'NEO4J_PASSWORD': 'test_password'
            }.get(key)
            
            service = create_memory_service("neo4j")
            assert service is not None
            assert isinstance(service, Neo4jMemoryService)
    
    @pytest.mark.integration
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving a memory."""
        service = create_memory_service("auto")
        
        # Store a memory
        memory_id = service.store_memory(
            self.test_memories[0]["content"],
            self.test_memories[0]["metadata"]
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        
        # Retrieve the memory
        retrieved = service.get_memory_by_id(memory_id)
        assert retrieved is not None
        assert retrieved.content == self.test_memories[0]["content"]
        assert retrieved.metadata == self.test_memories[0]["metadata"]
        
        # Clean up
        service.delete_memory(memory_id)
    
    @pytest.mark.integration
    def test_delete_memory(self):
        """Test deleting a memory."""
        service = create_memory_service("auto")
        
        # Store a memory
        memory_id = service.store_memory(
            self.test_memories[2]["content"],
            self.test_memories[2]["metadata"]
        )
        
        # Verify it exists
        retrieved = service.get_memory_by_id(memory_id)
        assert retrieved is not None
        
        # Delete it
        deleted = service.delete_memory(memory_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved = service.get_memory_by_id(memory_id)
        assert retrieved is None
    
    @pytest.mark.integration
    def test_delete_nonexistent_memory(self):
        """Test deleting a non-existent memory."""
        service = create_memory_service("auto")
        
        # Try to delete a non-existent memory
        deleted = service.delete_memory("non-existent-id-12345")
        assert deleted is False
    
    @pytest.mark.integration
    def test_bulk_delete_memories(self):
        """Test bulk deleting memories."""
        service = create_memory_service("auto")
        
        # Store multiple memories
        stored_ids = []
        for memory_data in self.test_memories:
            memory_id = service.store_memory(
                memory_data["content"], 
                memory_data["metadata"]
            )
            stored_ids.append(memory_id)
        
        # Delete first two memories
        delete_ids = stored_ids[:2]
        results = service.delete_memories(delete_ids)
        
        # Verify results
        assert len(results) == len(delete_ids)
        for memory_id in delete_ids:
            assert memory_id in results
            assert results[memory_id] is True
        
        # Verify memories are gone
        for memory_id in delete_ids:
            retrieved = service.get_memory_by_id(memory_id)
            assert retrieved is None
        
        # Verify remaining memory still exists
        remaining_memory = service.get_memory_by_id(stored_ids[2])
        assert remaining_memory is not None
        
        # Clean up
        service.delete_memory(stored_ids[2])
    
    @pytest.mark.integration
    def test_bulk_delete_mixed_valid_invalid(self):
        """Test bulk delete with mix of valid and invalid IDs."""
        service = create_memory_service("auto")
        
        # Store one memory
        memory_id = service.store_memory(
            self.test_memories[0]["content"],
            self.test_memories[0]["metadata"]
        )
        
        # Mix valid and invalid IDs
        mixed_ids = [memory_id, "fake-id-1", "fake-id-2"]
        results = service.delete_memories(mixed_ids)
        
        # Verify results
        assert len(results) == 3
        assert results[memory_id] is True  # Valid ID should be deleted
        assert results["fake-id-1"] is False  # Invalid ID should fail
        assert results["fake-id-2"] is False  # Invalid ID should fail
    
    @pytest.mark.integration
    def test_search_after_delete(self):
        """Test that deleted memories don't appear in search results."""
        service = create_memory_service("auto")
        
        # Store a memory with unique content
        unique_content = "Unique test content for search delete test"
        memory_id = service.store_memory(unique_content, {"test": "search_delete"})
        
        # Search for it before deletion
        results = service.retrieve_memories("unique test content search delete", limit=5)
        found_before = any(result.memory_id == memory_id for result in results)
        assert found_before is True
        
        # Delete it
        deleted = service.delete_memory(memory_id)
        assert deleted is True
        
        # Search again after deletion
        results = service.retrieve_memories("unique test content search delete", limit=5)
        found_after = any(result.memory_id == memory_id for result in results)
        assert found_after is False
    
    @pytest.mark.integration
    def test_retrieve_memories_similarity_search(self):
        """Test similarity search retrieval."""
        service = create_memory_service("auto")
        
        # Store memories with similar content
        memory_ids = []
        for i, memory_data in enumerate(self.test_memories):
            memory_id = service.store_memory(
                memory_data["content"],
                memory_data["metadata"]
            )
            memory_ids.append(memory_id)
        
        # Search for similar content
        results = service.retrieve_memories("test memory content", limit=5)
        
        # Should find some results
        assert len(results) > 0
        
        # Results should have the expected structure
        for result in results:
            assert hasattr(result, 'memory_id')
            assert hasattr(result, 'content')
            assert hasattr(result, 'metadata')
            assert hasattr(result, 'similarity')
            assert 0 <= result.similarity <= 1
        
        # Clean up
        service.delete_memories(memory_ids)
    
    def test_empty_bulk_delete(self):
        """Test bulk delete with empty list."""
        service = create_memory_service("auto")
        
        # Test empty list
        results = service.delete_memories([])
        assert results == {}