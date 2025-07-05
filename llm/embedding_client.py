import os
from typing import List
from sentence_transformers import SentenceTransformer
from .llm_types import EmbeddingClient


class SentenceTransformerEmbeddingClient(EmbeddingClient):
    """Embedding client using SentenceTransformers library"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding client
        
        Args:
            model_name: Name of the SentenceTransformer model to use
                       Defaults to EMBEDDING_MODEL env var or "all-MiniLM-L6-v2"
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.encoder = SentenceTransformer(self.model_name)
        
        # Cache embedding dimension
        test_embedding = self.encoder.encode("test")
        self._embedding_dim = len(test_embedding)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.encoder.encode(text)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.encoder.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this client"""
        return self._embedding_dim
    
    @property
    def model_info(self) -> str:
        """Get information about the embedding model"""
        return f"SentenceTransformer({self.model_name}) - {self._embedding_dim}d"