"""
Text chunking service package.

This package provides:
- Text chunking for semantic processing
- Multiple chunking strategies
- Integration with embedding services
- Support for various document types
"""

from .chunking_service import ChunkingService
from .chunking_factory import (
    ChunkingServiceFactory,
    create_chunking_service,
)

__all__ = [
    "ChunkingService", 
    "ChunkingServiceFactory",
    "create_chunking_service",
]