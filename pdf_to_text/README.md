# PDF to Text Service

Extract text from PDFs and images with AI-powered vision fallback and RAG-ready chunking.

## Features

- **PDF Text Extraction**: PyMuPDF-based text extraction with multiple output formats
- **Vision AI Integration**: Gemini Vision API for scanned PDFs and complex layouts
- **Image Processing**: Extract text from images (PNG, JPEG, etc.)
- **RAG Integration**: Semantic chunking for vector databases
- **Flexible Options**: Custom prompts, table extraction, layout preservation

## Quick Start

### Basic PDF Text Extraction

```python
from pdf_to_text import extract_text_from_pdf

# Simple text extraction
text = extract_text_from_pdf("document.pdf")
print(text)
```

### Advanced PDF Processing

```python
from pdf_to_text import PDFToTextService, PDFExtractOptions

service = PDFToTextService()
options = PDFExtractOptions(
    preserve_layout=True,
    page_range=(0, 5),  # First 5 pages
    use_vision_fallback=True  # Auto-fallback for scanned PDFs
)

result = service.extract_text_from_file("document.pdf", options)
print(f"Extracted {len(result.text)} characters from {result.page_count} pages")
```

### Vision-Based Text Extraction

```python
from pdf_to_text import extract_text_from_scanned_pdf, extract_tables_from_pdf

# Extract text from scanned/image-based PDFs
text = extract_text_from_scanned_pdf("scanned_document.pdf")

# Extract table data with vision AI
table_result = extract_tables_from_pdf("document_with_tables.pdf")
print(table_result.vision_text)
```

### Visual-to-Text Service

```python
from pdf_to_text.visual_to_text_service import VisualToTextService

service = VisualToTextService()

# Extract from images
result = service.extract_text_from_file("screenshot.png")
print(result.text)

# Extract from base64 images
result = service.extract_text_from_base64_image(base64_string)
print(result.text)
```

## RAG Integration

```python
from pdf_to_text import extract_pdf_for_rag

# Extract and chunk for vector databases
chunks = extract_pdf_for_rag("document.pdf", chunk_size=1000, chunk_overlap=200)

for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['character_count']} chars")
    # Ready for embedding: chunk['text']
```

## Common Use Cases

### Document Processing Pipeline

```python
from pdf_to_text import PDFToTextService, PDFExtractOptions

def process_document(pdf_path):
    service = PDFToTextService()
    options = PDFExtractOptions(
        use_vision_fallback=True,
        create_semantic_chunks=True,
        chunk_size=800
    )
    
    result = service.extract_text_from_file(pdf_path, options)
    
    return {
        'text': result.text,
        'chunks': result.semantic_chunks,
        'method': result.processing_method,
        'pages': result.page_count
    }
```

### Table Extraction

```python
from pdf_to_text import extract_tables_from_pdf

# Extract structured table data
result = extract_tables_from_pdf("financial_report.pdf")
if result.vision_text:
    print("Table data:", result.vision_text)
```

### Batch Processing

```python
import os
from pdf_to_text import extract_text_from_pdf

def process_pdf_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            
            # Save extracted text
            output_path = pdf_path.replace('.pdf', '.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
```

## Configuration

### Environment Variables

```bash
# Required for vision processing
export GOOGLE_AI_STUDIO_KEY="your_gemini_api_key"
```

### Dependencies

The service integrates with:
- **LLM Service**: Gemini vision clients (`../llm/vision_clients.py`)
- **Chunking Service**: Semantic chunking (`../chunking/chunking_service.py`)

## API Reference

### Key Classes

- `PDFToTextService`: Main PDF processing service
- `VisualToTextService`: Unified PDF/image processing
- `PDFExtractOptions`: Configuration for PDF extraction
- `VisualExtractOptions`: Configuration for visual content

### Quick Functions

- `extract_text_from_pdf(file_path)`: Simple PDF text extraction
- `extract_text_from_scanned_pdf(file_path)`: Vision-based extraction
- `extract_tables_from_pdf(file_path)`: Table-focused extraction
- `extract_pdf_for_rag(file_path)`: RAG-ready chunking
- `extract_text_from_visual(file_path)`: Unified visual processing

## Error Handling

```python
try:
    result = service.extract_text_from_file("document.pdf")
except FileNotFoundError:
    print("PDF file not found")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

## Performance Notes

- Vision processing limited to first 5 pages for cost control
- Automatic fallback to vision when traditional extraction yields <50 characters
- Chunking service integration provides semantic boundaries
- Base64 image processing for memory efficiency