#!/usr/bin/env python3
"""
Example usage of the PDF-to-Text service
"""

import sys
import os
from pdf_to_text_service import (
    PDFToTextService, PDFExtractOptions, extract_text_from_pdf, pdf_to_text_file,
    extract_with_vision_fallback, extract_tables_from_pdf, extract_text_from_scanned_pdf,
    extract_pdf_for_rag
)


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic PDF Text Extraction ===")
    
    # Create service instance
    service = PDFToTextService()
    
    # Example PDF file (you'll need to provide a real PDF file)
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        print("Please provide a PDF file to test with.")
        return
    
    try:
        # Extract text with default options
        result = service.extract_text_from_file(pdf_file)
        
        print(f"Extracted text from {result.page_count} pages:")
        print(f"First 500 characters: {result.text[:500]}...")
        print(f"Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_advanced_usage():
    """Advanced usage with custom options"""
    print("\n=== Advanced PDF Text Extraction ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    service = PDFToTextService()
    
    # Custom extraction options
    options = PDFExtractOptions(
        output_format="html",  # Extract as HTML
        preserve_layout=True,  # Preserve layout information
        page_range=(0, 3)      # Extract only first 3 pages
    )
    
    try:
        result = service.extract_text_from_file(pdf_file, options)
        
        print(f"Extracted HTML from pages 1-3 of {result.page_count} total pages:")
        print(f"First 300 characters: {result.text[:300]}...")
        
        if result.pages:
            print(f"Individual page data available for {len(result.pages)} pages")
            for page in result.pages:
                print(f"  Page {page['page_number']}: {len(page['text'])} characters")
                
    except Exception as e:
        print(f"Error: {e}")


def example_convenience_functions():
    """Using convenience functions"""
    print("\n=== Using Convenience Functions ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    try:
        # Quick text extraction
        text = extract_text_from_pdf(pdf_file)
        print(f"Quick extraction: {len(text)} characters")
        
        # Convert to text file
        output_file = pdf_to_text_file(pdf_file, "extracted_text.txt")
        print(f"Saved text to: {output_file}")
        
        # Check if file was created
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"First line of extracted text: {first_line}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_pdf_info():
    """Get PDF information without extracting text"""
    print("\n=== PDF Information ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    service = PDFToTextService()
    
    try:
        info = service.get_pdf_info(pdf_file)
        print(f"PDF Information:")
        print(f"  Pages: {info['page_count']}")
        print(f"  Encrypted: {info['is_encrypted']}")
        print(f"  File size: {info['file_size']} bytes")
        print(f"  Metadata: {info['metadata']}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_with_user_file():
    """Example that works with a user-provided file"""
    print("\n=== User File Example ===")
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_pdf_file>")
        print("Or run without arguments to see other examples")
        return
    
    pdf_file = sys.argv[1]
    
    if not os.path.exists(pdf_file):
        print(f"File not found: {pdf_file}")
        return
    
    service = PDFToTextService()
    
    try:
        print(f"Processing: {pdf_file}")
        
        # Get basic info
        info = service.get_pdf_info(pdf_file)
        print(f"Pages: {info['page_count']}")
        print(f"File size: {info['file_size']} bytes")
        
        # Extract text
        result = service.extract_text_from_file(pdf_file)
        print(f"Extracted {len(result.text)} characters")
        
        # Save to file
        output_file = pdf_file + ".txt"
        service.extract_text_to_file(pdf_file, output_file)
        print(f"Saved text to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")










def example_vision_processing():
    """Example using vision-based PDF processing"""
    print("\n=== Vision-Based PDF Processing ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    try:
        print("1. Smart extraction with vision fallback:")
        result = extract_with_vision_fallback(pdf_file)
        print(f"   Processing method used: {result.processing_method}")
        print(f"   Vision processing used: {result.vision_processing_used}")
        print(f"   Text length: {len(result.text)} characters")
        if result.vision_text:
            print(f"   Vision text length: {len(result.vision_text)} characters")
        
        print("\n2. Table extraction using vision:")
        table_result = extract_tables_from_pdf(pdf_file)
        if table_result.vision_text:
            print(f"   Extracted table data: {table_result.vision_text[:200]}...")
        else:
            print("   No table data extracted")
        
        print("\n3. Scanned PDF text extraction:")
        scanned_text = extract_text_from_scanned_pdf(pdf_file)
        print(f"   Scanned text (first 200 chars): {scanned_text[:200]}...")
        
        print("\n4. Force vision processing:")
        service = PDFToTextService()
        vision_options = PDFExtractOptions(
            force_vision_processing=True,
            vision_prompt="Extract all text with special attention to headers, footers, and any structured data."
        )
        vision_result = service.extract_text_from_file(pdf_file, vision_options)
        
        if vision_result.vision_text:
            print(f"   Force vision result: {vision_result.vision_text[:150]}...")
        else:
            print("   Force vision processing failed")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Vision processing requires GOOGLE_AI_STUDIO_KEY and Gemini API access")


def example_advanced_vision_features():
    """Advanced vision processing features"""
    print("\n=== Advanced Vision Features ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    service = PDFToTextService()
    
    try:
        print("1. Custom vision prompt for specific extraction:")
        custom_options = PDFExtractOptions(
            force_vision_processing=True,
            vision_prompt="Focus on extracting numerical data, dates, and any financial information from this document."
        )
        custom_result = service.extract_text_from_file(pdf_file, custom_options)
        if custom_result.vision_text:
            print(f"   Custom extraction: {custom_result.vision_text[:200]}...")
        
        print("\n2. Table-focused extraction:")
        table_options = PDFExtractOptions(
            force_vision_processing=True,
            extract_tables=True
        )
        table_result = service.extract_text_from_file(pdf_file, table_options)
        if table_result.vision_text:
            print(f"   Table data: {table_result.vision_text[:200]}...")
        
        print("\n3. Image text extraction (for documents with embedded text images):")
        image_options = PDFExtractOptions(
            force_vision_processing=True,
            extract_images_text=True
        )
        image_result = service.extract_text_from_file(pdf_file, image_options)
        if image_result.vision_text:
            print(f"   Image text: {image_result.vision_text[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")


def example_rag_integration():
    """Example showing RAG integration with existing chunking service"""
    print("\n=== RAG Integration with Existing Chunking Service ===")
    
    pdf_file = "example.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Example PDF file '{pdf_file}' not found.")
        return
    
    try:
        print("1. RAG-ready chunking using existing chunking service:")
        rag_chunks = extract_pdf_for_rag(pdf_file, chunk_size=800, chunk_overlap=100)
        print(f"   Created {len(rag_chunks)} chunks")
        if rag_chunks:
            print(f"   First chunk preview: {rag_chunks[0]['text'][:100]}...")
            print(f"   Chunk metadata: {len(rag_chunks[0]['text'])} chars, {rag_chunks[0]['word_count']} words")
        
        print("\n2. Manual chunking with different parameters:")
        service = PDFToTextService()
        options = PDFExtractOptions(
            create_semantic_chunks=True,
            chunk_size=500,  # Smaller chunks
            chunk_overlap=50,  # Less overlap
            preserve_section_boundaries=True
        )
        result = service.extract_text_from_file(pdf_file, options)
        
        if result.semantic_chunks:
            print(f"   Small chunks created: {len(result.semantic_chunks)}")
            avg_size = sum(chunk['character_count'] for chunk in result.semantic_chunks) / len(result.semantic_chunks)
            print(f"   Average chunk size: {avg_size:.0f} characters")
        
        print("\n3. Ready for vector database storage:")
        if rag_chunks:
            print("   Chunks are ready to be embedded and stored in vector database")
            print("   Each chunk contains: text, character_count, word_count, chunk_id, start_position")
            print(f"   Example chunk keys: {list(rag_chunks[0].keys())}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("PDF-to-Text Service Example Usage")
    print("=" * 50)
    
    # Check if user provided a file
    if len(sys.argv) > 1:
        example_with_user_file()
    else:
        # Run all examples
        example_basic_usage()
        example_advanced_usage()
        example_convenience_functions()
        example_pdf_info()
        
        
        # Vision processing examples
        example_vision_processing()
        example_advanced_vision_features()
        
        # RAG integration example
        example_rag_integration()
        
        print("\n" + "=" * 50)
        print("To test with your own PDF file, run:")
        print("python example_usage.py path/to/your/file.pdf")
        print("\nNote: For vision features, ensure you have:")
        print("- GOOGLE_AI_STUDIO_KEY environment variable for Gemini vision processing")