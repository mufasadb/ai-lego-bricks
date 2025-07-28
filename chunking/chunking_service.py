import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    target_size: int
    tolerance: int
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    preserve_words: bool = True
    paragraph_separator: str = "\n\n"
    sentence_pattern: str = r"[.!?]+\s+"


class ChunkingService:
    """
    A service for intelligently chunking text while preserving natural boundaries.

    Attempts to preserve paragraphs first, then sentences, then words, based on
    the target size and tolerance limits.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.min_size = config.target_size - config.tolerance
        self.max_size = config.target_size + config.tolerance

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into segments based on the configuration.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        remaining_text = text.strip()

        while remaining_text:
            chunk = self._get_next_chunk(remaining_text)
            if chunk:
                chunks.append(chunk)
                remaining_text = remaining_text[len(chunk) :].lstrip()
            else:
                # Fallback: take whatever remains
                chunks.append(remaining_text)
                break

        return chunks

    def _get_next_chunk(self, text: str) -> Optional[str]:
        """Get the next chunk from the text."""
        if len(text) <= self.max_size:
            return text

        # Try paragraph-based chunking first
        if self.config.preserve_paragraphs:
            chunk = self._chunk_by_paragraphs(text)
            if chunk:
                return chunk

        # Fallback to sentence-based chunking
        if self.config.preserve_sentences:
            chunk = self._chunk_by_sentences(text)
            if chunk:
                return chunk

        # Fallback to word-based chunking
        if self.config.preserve_words:
            chunk = self._chunk_by_words(text)
            if chunk:
                return chunk

        # Final fallback: hard cut at max_size
        return text[: self.max_size]

    def _chunk_by_paragraphs(self, text: str) -> Optional[str]:
        """Attempt to chunk by paragraphs."""
        paragraphs = text.split(self.config.paragraph_separator)

        chunk = ""
        for paragraph in paragraphs:
            potential_chunk = (
                chunk + (self.config.paragraph_separator if chunk else "") + paragraph
            )

            if len(potential_chunk) > self.max_size:
                # If we have accumulated content within limits, return it
                if len(chunk) >= self.min_size:
                    return chunk
                # If single paragraph is too large, let sentence chunking handle it
                elif len(paragraph) > self.max_size:
                    return None
                # If adding this paragraph would exceed max but we haven't reached min,
                # check if we can at least return what we have
                else:
                    return chunk if chunk else None

            chunk = potential_chunk

            # If we've reached a good size, return the chunk
            if len(chunk) >= self.min_size:
                return chunk

        # Return whatever we accumulated if it's not empty
        return chunk if chunk else None

    def _chunk_by_sentences(self, text: str) -> Optional[str]:
        """Attempt to chunk by sentences."""
        sentences = re.split(self.config.sentence_pattern, text)

        chunk = ""
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Add sentence separator back (except for the last sentence)
            separator = ". " if i < len(sentences) - 1 else ""
            potential_chunk = chunk + (" " if chunk else "") + sentence + separator

            if len(potential_chunk) > self.max_size:
                # If we have accumulated content within limits, return it
                if len(chunk) >= self.min_size:
                    return chunk
                # If single sentence is too large, let word chunking handle it
                elif len(sentence) > self.max_size:
                    return None
                # If we haven't reached min size, try to include this sentence anyway
                else:
                    return chunk if chunk else None

            chunk = potential_chunk

            # If we've reached a good size, return the chunk
            if len(chunk) >= self.min_size:
                return chunk

        return chunk if chunk else None

    def _chunk_by_words(self, text: str) -> Optional[str]:
        """Attempt to chunk by words."""
        words = text.split()

        chunk_words = []
        for word in words:
            potential_chunk = " ".join(chunk_words + [word])

            if len(potential_chunk) > self.max_size:
                # If we have accumulated content within limits, return it
                current_chunk = " ".join(chunk_words)
                if len(current_chunk) >= self.min_size:
                    return current_chunk
                # If single word is too large, we'll have to break it
                elif len(word) > self.max_size:
                    return current_chunk if current_chunk else word[: self.max_size]
                # If we haven't reached min size, include this word anyway
                else:
                    return current_chunk if current_chunk else word

            chunk_words.append(word)

            # If we've reached a good size, return the chunk
            if len(potential_chunk) >= self.min_size:
                return potential_chunk

        return " ".join(chunk_words) if chunk_words else None
