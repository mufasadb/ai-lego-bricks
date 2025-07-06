"""
Tests for chunking service functionality.

Converted from test/ directory to pytest format.
"""

import pytest
from chunking.chunking_service import ChunkingService, ChunkingConfig


class TestChunkingService:
    """Test suite for chunking service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChunkingConfig(target_size=100, tolerance=20)
        self.service = ChunkingService(self.config)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        result = self.service.chunk_text("")
        assert result == []
    
    def test_short_text(self):
        """Test chunking text shorter than target size."""
        text = "This is a short text."
        result = self.service.chunk_text(text)
        assert result == [text]
    
    def test_paragraph_preservation(self):
        """Test that paragraphs are preserved when possible."""
        text = "This is the first paragraph. It has multiple sentences.\n\nThis is the second paragraph. It also has sentences.\n\nThis is the third paragraph."
        result = self.service.chunk_text(text)
        
        # Should preserve paragraph boundaries
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= self.config.target_size + self.config.tolerance
    
    def test_sentence_fallback(self):
        """Test sentence fallback when paragraphs are too large."""
        # Create a long paragraph that exceeds max size
        long_paragraph = "This is a very long paragraph that will definitely exceed the maximum chunk size. " * 3
        
        result = self.service.chunk_text(long_paragraph)
        
        # Should break at sentence boundaries
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= self.config.target_size + self.config.tolerance
    
    def test_word_fallback(self):
        """Test word fallback when sentences are too large."""
        # Create a very long sentence
        long_sentence = "This is an extremely long sentence that goes on and on and on and definitely exceeds the maximum chunk size without any punctuation marks to break it up naturally."
        
        result = self.service.chunk_text(long_sentence)
        
        # Should break at word boundaries
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= self.config.target_size + self.config.tolerance
    
    def test_hard_cut_fallback(self):
        """Test hard cut when even words are too large."""
        # Create a single very long word
        long_word = "a" * 200
        
        result = self.service.chunk_text(long_word)
        
        # Should perform hard cut
        assert len(result) >= 1
        assert len(result[0]) <= self.config.target_size + self.config.tolerance
    
    def test_mixed_content(self):
        """Test chunking with mixed content types."""
        text = """This is a paragraph with multiple sentences. It should be chunked appropriately.

This is another paragraph. It has different content.

Here's a third paragraph with a very long sentence that might need to be broken up at the sentence level if it exceeds the maximum chunk size limit.

Final paragraph."""
        
        result = self.service.chunk_text(text)
        
        # Verify all chunks are within size limits
        for chunk in result:
            assert len(chunk) <= self.config.target_size + self.config.tolerance
            assert (len(chunk) >= self.config.target_size - self.config.tolerance or 
                   chunk == result[-1])  # Last chunk might be shorter
    
    def test_custom_paragraph_separator(self):
        """Test custom paragraph separator."""
        config = ChunkingConfig(target_size=50, tolerance=10, paragraph_separator="\n")
        service = ChunkingService(config)
        
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        result = service.chunk_text(text)
        
        # Should respect custom separator
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= 60  # target + tolerance
    
    def test_preserve_flags(self):
        """Test disabling preservation flags."""
        config = ChunkingConfig(
            target_size=50, 
            tolerance=10, 
            preserve_paragraphs=False,
            preserve_sentences=False,
            preserve_words=False
        )
        service = ChunkingService(config)
        
        text = "This is a test paragraph.\n\nThis is another paragraph."
        result = service.chunk_text(text)
        
        # Should perform hard cuts
        for chunk in result:
            assert len(chunk) <= 60  # target + tolerance

    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Test valid config
        config = ChunkingConfig(target_size=100, tolerance=20)
        assert config.target_size == 100
        assert config.tolerance == 20
        
        # Test invalid target size
        with pytest.raises(ValueError):
            ChunkingConfig(target_size=0, tolerance=10)
        
        # Test invalid tolerance
        with pytest.raises(ValueError):
            ChunkingConfig(target_size=100, tolerance=-5)
    
    def test_chunk_boundaries(self):
        """Test that chunk boundaries are respected."""
        text = "First chunk content. " * 10  # Should be chunked
        result = self.service.chunk_text(text)
        
        # Verify no chunk exceeds maximum size
        max_size = self.config.target_size + self.config.tolerance
        for chunk in result:
            assert len(chunk) <= max_size
            
        # Verify all original content is preserved
        reconstructed = "".join(result)
        assert reconstructed == text
    
    def test_edge_case_very_small_target(self):
        """Test with very small target size."""
        config = ChunkingConfig(target_size=10, tolerance=5)
        service = ChunkingService(config)
        
        text = "This is a test sentence that will need heavy chunking."
        result = service.chunk_text(text)
        
        # Should still produce reasonable chunks
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 15  # target + tolerance
            assert len(chunk) > 0
    
    def test_unicode_text_handling(self):
        """Test chunking with unicode characters."""
        text = "This is a test with émojis 🚀 and spëcial characters ñoñó."
        result = self.service.chunk_text(text)
        
        # Should handle unicode properly
        assert len(result) >= 1
        reconstructed = "".join(result)
        assert reconstructed == text