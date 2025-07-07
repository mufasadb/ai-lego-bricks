# Visual Content Processing Guide

## Overview

The AI Lego Bricks project includes a comprehensive **Visual to Text Service** that can extract text from various visual content formats including PDFs, images, and base64 encoded data. This service provides advanced capabilities like bounding box extraction for precise text positioning.

## Supported Formats

### Input Types
- **PDFs**: `.pdf` files
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- **Base64 Images**: Direct base64 encoded image strings

### Processing Methods
- **Traditional Text Extraction**: Fast OCR-based extraction for PDFs
- **Vision Processing**: AI-powered visual analysis using Gemini Vision
- **Hybrid Approach**: Automatically falls back to vision processing when needed

## Key Features

### 1. **Bounding Box Extraction**
Get precise coordinates for text elements:
```python
options = VisualExtractOptions(include_bounding_boxes=True)
result = service.extract_text_from_file("invoice.pdf", options)

# Access bounding box data
for bbox in result.bounding_boxes:
    print(f"Text location: {bbox}")  # "Text: [x1, y1, x2, y2]"
```

### 2. **Custom Vision Prompts**
Tailor extraction for specific use cases:
```python
options = VisualExtractOptions(
    vision_prompt="Extract subscription dates and billing information from this invoice"
)
```

### 3. **Table Extraction**
Specialized handling for tabular data:
```python
options = VisualExtractOptions(extract_tables=True)
```

### 4. **PDF to Images Conversion**
Convert PDF pages to base64 images for processing:
```python
images = service.convert_pdf_to_base64_images("document.pdf", dpi=150)
```

## Service Architecture

### Core Service Class
```python
from pdf_to_text.visual_to_text_service import VisualToTextService, VisualExtractOptions

service = VisualToTextService()
```

### Configuration Options
```python
options = VisualExtractOptions(
    # Vision processing
    vision_prompt="Custom extraction prompt",
    include_bounding_boxes=True,
    extract_tables=True,
    extract_images_text=True,
    
    # PDF-specific
    preserve_layout=True,
    page_range=(0, 5),  # Process first 5 pages
    pdf_flags=3,
    
    # Image processing
    image_dpi=150,
    image_format="png",
    
    # RAG integration
    create_semantic_chunks=True,
    chunk_size=1000,
    chunk_overlap=200
)
```

### Result Structure
```python
class VisualTextResult:
    text: str                           # Extracted text
    source_type: str                   # "pdf", "image", "base64_image"
    metadata: Dict[str, Any]           # File metadata
    page_count: Optional[int]          # Number of pages (PDFs)
    vision_processing_used: bool       # Whether vision AI was used
    processing_method: str             # "traditional" or "vision"
    bounding_boxes: Optional[List]     # Text coordinate data
    converted_images: Optional[List]   # Base64 images if converted
    semantic_chunks: Optional[List]    # RAG-ready chunks
```

## Usage Patterns

### 1. **Basic Text Extraction**
```python
# Simple text extraction
service = VisualToTextService()
result = service.extract_text_from_file("document.pdf")
print(result.text)
```

### 2. **Invoice Processing with Bounding Boxes**
```python
# Extract subscription dates with precise positioning
options = VisualExtractOptions(
    include_bounding_boxes=True,
    vision_prompt="""
    Extract subscription dates from this invoice.
    Look for subscription periods, billing cycles, or service dates.
    Provide bounding box coordinates for each date found.
    """,
    extract_tables=True
)

result = service.extract_text_from_file("zoom_invoice.pdf", options)

# Parse subscription dates
import re
pattern = r'(\w{3}\s+\d{1,2},?\s+\d{4})\s*-\s*(\w{3}\s+\d{1,2},?\s+\d{4})'
matches = re.findall(pattern, result.text)

for start_date, end_date in matches:
    print(f"Subscription period: {start_date} to {end_date}")
```

### 3. **Base64 Image Processing**
```python
# Process image data directly
result = service.extract_text_from_base64_image(
    base64_image_string,
    options=VisualExtractOptions(include_bounding_boxes=True)
)
```

### 4. **Batch PDF Processing**
```python
# Convert PDF to images and process each page
images = service.convert_pdf_to_base64_images("presentation.pdf")

for i, image_b64 in enumerate(images):
    page_result = service.extract_text_from_base64_image(image_b64)
    print(f"Page {i+1}: {page_result.text}")
```

## Agent Orchestration Integration

### Document Processing Step
```json
{
  "id": "extract_visual_content",
  "type": "document_processing",
  "config": {
    "include_bounding_boxes": true,
    "vision_prompt": "Extract key information with precise coordinates",
    "extract_tables": true,
    "preserve_layout": true
  },
  "inputs": {
    "file_path": "document.pdf"
  },
  "outputs": ["extracted_data"]
}
```

### Base64 Image Processing Step
```json
{
  "id": "process_screenshot",
  "type": "document_processing",
  "config": {
    "include_bounding_boxes": true,
    "vision_prompt": "Extract text from this screenshot"
  },
  "inputs": {
    "base64_image": "{{input.image_data}}"
  },
  "outputs": ["text_data"]
}
```

### Supported Configuration Options
- **`include_bounding_boxes`**: Enable coordinate extraction
- **`vision_prompt`**: Custom prompt for vision processing
- **`extract_tables`**: Focus on tabular data
- **`extract_images_text`**: Extract text from embedded images
- **`preserve_layout`**: Maintain original text layout
- **`page_range`**: Process specific page range (PDFs)
- **`semantic_analysis`**: Create semantic chunks for RAG

## Convenience Functions

### Quick Access Functions
```python
from pdf_to_text.visual_to_text_service import (
    extract_text_from_visual,           # Quick text extraction
    extract_text_from_base64_image,     # Base64 processing
    convert_pdf_to_images,              # PDF conversion
    extract_with_bounding_boxes,        # Coordinate extraction
    extract_tables_from_visual          # Table-focused extraction
)

# Examples
text = extract_text_from_visual("document.pdf")
bbox_result = extract_with_bounding_boxes("invoice.pdf")
pdf_images = convert_pdf_to_images("presentation.pdf", dpi=200)
```

## Real-World Use Cases

### 1. **Invoice Processing**
- Extract subscription dates with bounding boxes
- Parse billing amounts and tax information
- Identify table structures for line items

### 2. **Document Analysis**
- Convert PDF reports to searchable text
- Extract key information with precise positioning
- Process scanned documents using vision AI

### 3. **Screenshot Analysis**
- Extract text from application screenshots
- Analyze form data and UI elements
- Process mobile app interfaces

### 4. **Financial Document Processing**
- Parse bank statements and receipts
- Extract transaction details with coordinates
- Process insurance forms and claims

## Integration with Other Services

### Memory Integration
```python
# Extract and store in memory with semantic chunks
options = VisualExtractOptions(
    create_semantic_chunks=True,
    chunk_size=500,
    chunk_overlap=100
)

result = service.extract_text_from_file("report.pdf", options)

# Store chunks in memory service
for chunk in result.semantic_chunks:
    memory_service.store_memory(chunk['text'], {"source": "report.pdf"})
```

### LLM Integration
```python
# Extract text and process with LLM
result = service.extract_text_from_file("document.pdf")

# Send to LLM for analysis
from llm.generation_service import quick_generate_gemini
analysis = quick_generate_gemini(f"Analyze this document: {result.text}")
```

## Error Handling

### Common Error Scenarios
```python
try:
    result = service.extract_text_from_file("document.pdf", options)
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Unsupported file format: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

### Best Practices
- Always validate file existence before processing
- Check file extensions against supported formats
- Handle vision processing dependencies gracefully
- Provide fallback options when vision processing fails
- Use try-catch blocks for robust error handling

## Performance Considerations

### Optimization Tips
- Use traditional extraction for text-heavy PDFs (faster)
- Enable vision processing only when needed (API costs)
- Limit page ranges for large PDFs
- Adjust DPI settings for image conversion (balance quality vs. speed)
- Use semantic chunking for RAG applications

### Vision Processing Costs
- Vision processing uses AI APIs (costs money)
- Automatically enabled for image-heavy PDFs
- Can be forced on/off via configuration
- Limited to first 5 pages by default to control costs

## Testing and Validation

### Test Files Location
Place test files in these directories:
- `test_documents/`
- `examples/`
- Root directory (for quick tests)

### Example Test
```python
# Test with real Zoom invoice
result = service.extract_text_from_file("zoom_invoice_august.pdf", options)
print(f"Extracted: {result.text[:200]}...")
print(f"Method: {result.processing_method}")
print(f"Found dates: {parse_subscription_dates(result.text)}")
```

## Future Enhancements

### Planned Features
- OCR confidence scores
- Multi-language support
- Table structure recognition
- Form field identification
- Handwriting recognition
- Document classification

### Extension Points
- Custom vision model integration
- Additional file format support
- Advanced bounding box features
- Integration with document databases
- Workflow automation tools

This visual content processing system provides a powerful foundation for extracting structured data from diverse visual content while maintaining precision and flexibility for various use cases.