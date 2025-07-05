from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Memory(BaseModel):
    """Data structure for storing memory items"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_id: Optional[str] = None
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }

class MemoryService(ABC):
    """Abstract base class for memory services"""
    
    @abstractmethod
    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory and return its ID"""
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Retrieve memories based on similarity/relevance to query"""
        pass
    
    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory"""
        pass
    
    def delete_memories(self, memory_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple memories by ID
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Dict mapping memory_id -> success (True/False)
        """
        results = {}
        for memory_id in memory_ids:
            try:
                results[memory_id] = self.delete_memory(memory_id)
            except Exception:
                results[memory_id] = False
        return results
    
    def delete_memories_by_search(self, query: str, limit: int = 10, confirm: bool = False) -> Dict[str, bool]:
        """
        Delete memories matching a search query
        
        Args:
            query: Search query to find memories to delete
            limit: Maximum number of memories to delete
            confirm: If True, actually delete. If False, return what would be deleted
            
        Returns:
            Dict mapping memory_id -> success (True/False)
        """
        # Find memories matching the search
        memories_to_delete = self.retrieve_memories(query, limit=limit)
        
        if not memories_to_delete:
            return {}
        
        memory_ids = [memory.memory_id for memory in memories_to_delete]
        
        if confirm:
            return self.delete_memories(memory_ids)
        else:
            # Return what would be deleted (dry run)
            return {memory_id: False for memory_id in memory_ids}  # False = not actually deleted