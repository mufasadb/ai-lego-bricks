from chunking_service import ChunkingService, ChunkingConfig


def main():
    """Demonstrate the chunking service functionality."""
    
    # Sample text for demonstration
    sample_text = """
    Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

    Most NLP techniques rely on machine learning to derive meaning from human languages. This involves training algorithms on large datasets of text to learn patterns and relationships within language. Common tasks in NLP include sentiment analysis, named entity recognition, machine translation, and text summarization.

    Modern NLP systems use various approaches including statistical methods, neural networks, and transformer architectures. These systems have achieved remarkable success in tasks like language translation, chatbots, and content generation. The field continues to evolve rapidly with new breakthroughs emerging regularly.

    Applications of NLP are widespread across industries. In healthcare, NLP helps analyze medical records and research papers. In finance, it's used for fraud detection and automated trading. Customer service benefits from chatbots and sentiment analysis. Social media platforms use NLP for content moderation and recommendation systems.
    """
    
    # Example 1: Basic chunking with default settings
    print("=== Example 1: Basic Chunking ===")
    config1 = ChunkingConfig(target_size=200, tolerance=50)
    service1 = ChunkingService(config1)
    chunks1 = service1.chunk_text(sample_text)
    
    print(f"Target size: {config1.target_size} ± {config1.tolerance}")
    print(f"Number of chunks: {len(chunks1)}")
    for i, chunk in enumerate(chunks1, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print("-" * 40)
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    # Example 2: Smaller chunks with tighter tolerance
    print("\n\n=== Example 2: Smaller Chunks ===")
    config2 = ChunkingConfig(target_size=100, tolerance=20)
    service2 = ChunkingService(config2)
    chunks2 = service2.chunk_text(sample_text)
    
    print(f"Target size: {config2.target_size} ± {config2.tolerance}")
    print(f"Number of chunks: {len(chunks2)}")
    for i, chunk in enumerate(chunks2, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print("-" * 40)
        print(chunk[:80] + "..." if len(chunk) > 80 else chunk)
    
    # Example 3: Custom paragraph separator
    print("\n\n=== Example 3: Custom Separator ===")
    text_with_custom_sep = sample_text.replace('\n\n', ' | ')
    config3 = ChunkingConfig(target_size=150, tolerance=30, paragraph_separator=' | ')
    service3 = ChunkingService(config3)
    chunks3 = service3.chunk_text(text_with_custom_sep)
    
    print(f"Using custom separator: ' | '")
    print(f"Number of chunks: {len(chunks3)}")
    for i, chunk in enumerate(chunks3, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print("-" * 40)
        print(chunk[:80] + "..." if len(chunk) > 80 else chunk)
    
    # Example 4: Testing preservation settings
    print("\n\n=== Example 4: No Paragraph Preservation ===")
    config4 = ChunkingConfig(
        target_size=100, 
        tolerance=20, 
        preserve_paragraphs=False
    )
    service4 = ChunkingService(config4)
    chunks4 = service4.chunk_text(sample_text)
    
    print(f"Paragraph preservation disabled")
    print(f"Number of chunks: {len(chunks4)}")
    for i, chunk in enumerate(chunks4, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print("-" * 40)
        print(chunk[:80] + "..." if len(chunk) > 80 else chunk)


if __name__ == "__main__":
    main()