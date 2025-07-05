import unittest
from chunking.chunking_service import ChunkingService, ChunkingConfig


class TestChunkingService(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ChunkingConfig(target_size=100, tolerance=20)
        self.service = ChunkingService(self.config)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        result = self.service.chunk_text("")
        self.assertEqual(result, [])
    
    def test_short_text(self):
        """Test chunking text shorter than target size."""
        text = "This is a short text."
        result = self.service.chunk_text(text)
        self.assertEqual(result, [text])
    
    def test_paragraph_preservation(self):
        """Test that paragraphs are preserved when possible."""
        text = "This is the first paragraph. It has multiple sentences.\n\nThis is the second paragraph. It also has sentences.\n\nThis is the third paragraph."
        result = self.service.chunk_text(text)
        
        # Should preserve paragraph boundaries
        self.assertTrue(len(result) > 1)
        for chunk in result:
            self.assertTrue(len(chunk) <= self.config.target_size + self.config.tolerance)
    
    def test_sentence_fallback(self):
        """Test sentence fallback when paragraphs are too large."""
        # Create a long paragraph that exceeds max size
        long_paragraph = "This is a very long paragraph that will definitely exceed the maximum chunk size. " * 3
        
        result = self.service.chunk_text(long_paragraph)
        
        # Should break at sentence boundaries
        self.assertTrue(len(result) > 1)
        for chunk in result:
            self.assertTrue(len(chunk) <= self.config.target_size + self.config.tolerance)
    
    def test_word_fallback(self):
        """Test word fallback when sentences are too large."""
        # Create a very long sentence
        long_sentence = "This is an extremely long sentence that goes on and on and on and definitely exceeds the maximum chunk size without any punctuation marks to break it up naturally."
        
        result = self.service.chunk_text(long_sentence)
        
        # Should break at word boundaries
        self.assertTrue(len(result) >= 1)
        for chunk in result:
            self.assertTrue(len(chunk) <= self.config.target_size + self.config.tolerance)
    
    def test_hard_cut_fallback(self):
        """Test hard cut when even words are too large."""
        # Create a single very long word
        long_word = "a" * 200
        
        result = self.service.chunk_text(long_word)
        
        # Should perform hard cut
        self.assertTrue(len(result) >= 1)
        self.assertTrue(len(result[0]) <= self.config.target_size + self.config.tolerance)
    
    def test_mixed_content(self):
        """Test chunking with mixed content types."""
        text = """This is a paragraph with multiple sentences. It should be chunked appropriately.

This is another paragraph. It has different content.

Here's a third paragraph with a very long sentence that might need to be broken up at the sentence level if it exceeds the maximum chunk size limit.

Final paragraph."""
        
        result = self.service.chunk_text(text)
        
        # Verify all chunks are within size limits
        for chunk in result:
            self.assertTrue(len(chunk) <= self.config.target_size + self.config.tolerance)
            self.assertTrue(len(chunk) >= self.config.target_size - self.config.tolerance or 
                          chunk == result[-1])  # Last chunk might be shorter
    
    def test_custom_paragraph_separator(self):
        """Test custom paragraph separator."""
        config = ChunkingConfig(target_size=50, tolerance=10, paragraph_separator="\n")
        service = ChunkingService(config)
        
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        result = service.chunk_text(text)
        
        # Should respect custom separator
        self.assertTrue(len(result) >= 1)
        for chunk in result:
            self.assertTrue(len(chunk) <= 60)  # target + tolerance
    
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
            self.assertTrue(len(chunk) <= 60)  # target + tolerance


if __name__ == '__main__':
    unittest.main()