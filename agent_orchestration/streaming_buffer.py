"""
Enhanced streaming buffer implementation for intelligent stream forwarding
"""

import re
import time
from typing import Generator, List, Dict, Any, Optional
from .models import StreamBufferConfig


class StreamBuffer:
    """
    Enhanced streaming buffer that forwards content based on configurable strategies.
    
    Supports multiple forwarding strategies:
    - sentence: Forward on complete sentences
    - time: Forward after max buffer time
    - chunk_size: Forward after N characters
    - word_count: Forward after N words
    - immediate: Forward immediately (pass-through)
    """
    
    def __init__(self, config: StreamBufferConfig):
        """Initialize streaming buffer with configuration."""
        self.config = config
        self.buffer = ""
        self.last_forward_time = time.time()
        self.total_forwarded = 0
        self.sentence_count = 0
        
    def process_chunk(self, chunk: str) -> List[str]:
        """
        Process incoming chunk and return any content ready for forwarding.
        
        Args:
            chunk: Incoming text chunk
            
        Returns:
            List of text chunks ready to forward (empty if nothing ready)
        """
        if not chunk:
            return []
            
        # Add to buffer
        self.buffer += chunk
        
        # Check forwarding strategy
        if self.config.forward_on == "immediate":
            # Pass-through mode
            self.buffer = ""  # Clear buffer
            self.total_forwarded += len(chunk)
            return [chunk]
            
        elif self.config.forward_on == "sentence":
            return self._forward_on_sentences()
            
        elif self.config.forward_on == "time":
            return self._forward_on_time()
            
        elif self.config.forward_on == "chunk_size":
            return self._forward_on_chunk_size()
            
        elif self.config.forward_on == "word_count":
            return self._forward_on_word_count()
            
        else:
            # Default to sentence-based forwarding
            return self._forward_on_sentences()
    
    def flush(self) -> List[str]:
        """
        Flush any remaining content in buffer.
        
        Returns:
            Any remaining buffered content
        """
        if self.buffer.strip():
            content = self.buffer
            self.buffer = ""
            self.total_forwarded += len(content)
            return [content]
        return []
    
    def _forward_on_sentences(self) -> List[str]:
        """Forward content when complete sentences are available."""
        sentences = self._extract_complete_sentences()
        
        if len(sentences) >= self.config.sentence_count:
            # Forward the required number of sentences
            to_forward = sentences[:self.config.sentence_count]
            
            # Remove forwarded sentences from buffer
            for sentence in to_forward:
                # Remove first occurrence of each sentence
                sentence_pattern = re.escape(sentence.strip())
                self.buffer = re.sub(sentence_pattern + r'\s*', '', self.buffer, count=1)
            
            content = ' '.join(to_forward)
            self.total_forwarded += len(content)
            self.sentence_count += len(to_forward)
            self.last_forward_time = time.time()
            
            return [content]
        
        # Check if we've exceeded max buffer time
        if time.time() - self.last_forward_time > self.config.max_buffer_time:
            return self._force_forward()
            
        return []
    
    def _forward_on_time(self) -> List[str]:
        """Forward content based on time threshold."""
        if (time.time() - self.last_forward_time > self.config.max_buffer_time and 
            len(self.buffer) >= self.config.min_chunk_length):
            return self._force_forward()
        return []
    
    def _forward_on_chunk_size(self) -> List[str]:
        """Forward content when buffer reaches size threshold."""
        if len(self.buffer) >= self.config.chunk_size:
            # Forward chunk_size characters
            to_forward = self.buffer[:self.config.chunk_size]
            self.buffer = self.buffer[self.config.chunk_size:]
            self.total_forwarded += len(to_forward)
            self.last_forward_time = time.time()
            return [to_forward]
            
        # Check time threshold as backup
        if time.time() - self.last_forward_time > self.config.max_buffer_time:
            return self._force_forward()
            
        return []
    
    def _forward_on_word_count(self) -> List[str]:
        """Forward content when buffer reaches word count threshold."""
        words = self.buffer.split()
        
        if len(words) >= self.config.word_count:
            # Forward word_count words
            words_to_forward = words[:self.config.word_count]
            remaining_words = words[self.config.word_count:]
            
            content = ' '.join(words_to_forward)
            self.buffer = ' '.join(remaining_words)
            self.total_forwarded += len(content)
            self.last_forward_time = time.time()
            
            return [content]
        
        # Check time threshold as backup
        if time.time() - self.last_forward_time > self.config.max_buffer_time:
            return self._force_forward()
            
        return []
    
    def _extract_complete_sentences(self) -> List[str]:
        """
        Extract complete sentences from the buffer.
        
        Returns:
            List of complete sentences found in buffer
        """
        # Enhanced sentence detection pattern
        # Matches sentences ending with . ! ? followed by space or end of string
        sentence_pattern = r'[^.!?]*[.!?]+(?:\s+|$)'
        
        sentences = re.findall(sentence_pattern, self.buffer)
        
        # Filter out very short sentences and clean up
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.config.min_chunk_length:
                complete_sentences.append(sentence)
        
        return complete_sentences
    
    def _force_forward(self) -> List[str]:
        """Force forward any buffered content."""
        if self.buffer.strip():
            content = self.buffer
            self.buffer = ""
            self.total_forwarded += len(content)
            self.last_forward_time = time.time()
            return [content]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_length": len(self.buffer),
            "total_forwarded": self.total_forwarded,
            "sentences_processed": self.sentence_count,
            "time_since_last_forward": time.time() - self.last_forward_time,
            "forward_strategy": self.config.forward_on
        }


def create_streaming_generator(
    source_generator: Generator[str, None, str],
    buffer_config: StreamBufferConfig
) -> Generator[str, None, str]:
    """
    Create an enhanced streaming generator with intelligent buffering.
    
    Args:
        source_generator: Source generator producing text chunks
        buffer_config: Buffer configuration for forwarding strategy
        
    Yields:
        Buffered and intelligently forwarded text chunks
        
    Returns:
        Final complete response
    """
    buffer = StreamBuffer(buffer_config)
    complete_response = ""
    
    try:
        for chunk in source_generator:
            complete_response += chunk
            
            # Process chunk through buffer
            ready_chunks = buffer.process_chunk(chunk)
            
            # Yield any ready chunks
            for ready_chunk in ready_chunks:
                yield ready_chunk
        
        # Flush any remaining content
        final_chunks = buffer.flush()
        for final_chunk in final_chunks:
            yield final_chunk
            
    except Exception as e:
        # Ensure we flush buffer even on error
        final_chunks = buffer.flush()
        for final_chunk in final_chunks:
            yield final_chunk
        raise e
    
    return complete_response


def enhance_step_streaming(
    streaming_generator: Generator[str, None, str],
    step_config: Optional[Dict[str, Any]] = None
) -> Generator[str, None, str]:
    """
    Enhance a step's streaming output with intelligent buffering.
    
    Args:
        streaming_generator: Original streaming generator
        step_config: Step configuration (may contain stream_buffer config)
        
    Returns:
        Enhanced streaming generator with intelligent buffering
    """
    # Extract buffer configuration
    if step_config and "stream_buffer" in step_config:
        buffer_config = StreamBufferConfig(**step_config["stream_buffer"])
    else:
        # Use default sentence-based buffering
        buffer_config = StreamBufferConfig()
    
    # Create enhanced generator
    return create_streaming_generator(streaming_generator, buffer_config)