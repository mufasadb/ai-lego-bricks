# PDF to Text Service - Detailed Technical Documentation

## 📄 Architecture Overview

The PDF to Text Service provides a comprehensive document processing pipeline that combines traditional PDF extraction with advanced AI vision capabilities. It seamlessly integrates with chunking services for RAG applications and supports multiple extraction strategies based on document complexity.

### Core Components

```
PDF to Text Ecosystem
├── PDFToTextService (Main Service)
│   ├── Traditional PDF Extraction (PyMuPDF)
│   ├── Vision-based Extraction (Gemini Vision)
│   ├── Hybrid Processing Pipeline
│   └── Quality Assessment
├── VisualToTextService (Unified Visual Processing)
│   ├── PDF Page to Image Conversion
│   ├── Image Text Extraction
│   ├── Base64 Image Processing
│   └── Bounding Box Extraction
├── Extraction Options System
│   ├── PDFExtractOptions (PDF-specific)
│   ├── VisualExtractOptions (Image/Visual)
│   ├── Output Format Control
│   └── Processing Preferences
├── Integration Layers
│   ├── Chunking Service Integration
│   ├── LLM Vision Client Integration
│   ├── Agent Orchestration Support
│   └── Batch Processing
└── Quality and Validation
    ├── Extraction Quality Assessment
    ├── Content Validation
    ├── Error Recovery
    └── Performance Optimization
```

## 🏗️ Core Service Implementation

### PDFToTextService Architecture

```python
class PDFToTextService:
    """
    Main PDF to text service with intelligent extraction strategies.
    
    Design Principles:
    - Multi-strategy extraction (traditional + vision)
    - Quality-based fallback mechanisms
    - Configurable output formats
    - RAG-ready processing
    - Error resilience
    """
    
    def __init__(self, credential_manager: Optional['CredentialManager'] = None):
        """
        Initialize PDF service with vision capabilities.
        
        Initialization Process:
        1. Setup credential management
        2. Initialize vision clients
        3. Setup chunking integration
        4. Configure extraction engines
        """
        
        self.credential_manager = credential_manager or default_credential_manager
        
        # Initialize vision capabilities
        self.vision_available = self._initialize_vision_clients()
        
        # Initialize chunking integration
        self.chunking_available = self._initialize_chunking_service()
        
        # Performance metrics
        self.extraction_metrics = {
            'traditional_extractions': 0,
            'vision_extractions': 0,
            'hybrid_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0.0
        }
    
    def _initialize_vision_clients(self) -> bool:
        """
        Initialize vision processing capabilities.
        
        Vision Setup:
        - Check for vision API credentials
        - Initialize Gemini Vision client
        - Test basic functionality
        - Configure fallback options
        """
        
        try:
            # Check for Gemini Vision credentials
            if self.credential_manager.has_credential('GOOGLE_AI_STUDIO_KEY'):
                from llm.vision_clients import GeminiVisionClient
                
                self.vision_client = GeminiVisionClient(
                    credential_manager=self.credential_manager
                )
                
                # Test vision client
                test_success = self._test_vision_client()
                
                if test_success:
                    logger.info("✓ Vision processing capabilities initialized")
                    return True
                else:
                    logger.warning("⚠ Vision client test failed")
                    return False
            else:
                logger.info("No vision API credentials found - vision features disabled")
                return False
                
        except ImportError as e:
            logger.warning(f"Vision dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Vision initialization failed: {e}")
            return False
    
    def _test_vision_client(self) -> bool:
        """Test vision client with minimal functionality check."""
        
        try:
            # Create a simple test image (1x1 pixel)
            from PIL import Image
            import io
            import base64
            
            test_image = Image.new('RGB', (1, 1), color='white')
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            test_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Test vision analysis
            result = self.vision_client.analyze_image_base64(
                test_b64, 
                "Describe this image briefly"
            )
            
            return bool(result and len(result) > 0)
            
        except Exception as e:
            logger.warning(f"Vision client test failed: {e}")
            return False
    
    def extract_text_from_file(
        self, 
        file_path: str, 
        options: Optional[PDFExtractOptions] = None
    ) -> PDFExtractionResult:
        """
        Extract text from PDF file using intelligent strategy selection.
        
        Extraction Pipeline:
        1. Validate input file
        2. Assess document complexity
        3. Select optimal extraction strategy
        4. Execute extraction with fallback
        5. Post-process and validate results
        """
        
        import time
        start_time = time.time()
        
        # Stage 1: Input Validation
        self._validate_pdf_file(file_path)
        
        # Stage 2: Options Setup
        options = options or PDFExtractOptions()
        
        # Stage 3: Strategy Selection
        extraction_strategy = self._select_extraction_strategy(file_path, options)
        
        # Stage 4: Execute Extraction
        try:
            result = self._execute_extraction(file_path, options, extraction_strategy)
            
            # Stage 5: Post-processing
            result = self._post_process_result(result, options)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(extraction_strategy, processing_time, True)
            
            return result
            
        except Exception as e:
            # Error handling and fallback
            logger.error(f"Extraction failed with strategy {extraction_strategy}: {e}")
            
            # Try fallback strategy
            fallback_result = self._execute_fallback_extraction(file_path, options, e)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics('fallback', processing_time, False)
            
            return fallback_result
    
    def _select_extraction_strategy(self, file_path: str, options: PDFExtractOptions) -> str:
        """
        Select optimal extraction strategy based on document analysis.
        
        Strategy Selection Criteria:
        1. User preferences in options
        2. Document complexity assessment
        3. Available processing capabilities
        4. Performance considerations
        """
        
        # User-forced vision processing
        if options.force_vision_processing:
            if self.vision_available:
                return 'vision_only'
            else:
                logger.warning("Vision processing forced but not available - using traditional")
                return 'traditional'
        
        # User-disabled vision processing
        if options.disable_vision_fallback:
            return 'traditional'
        
        # Assess document complexity
        complexity_score = self._assess_document_complexity(file_path)
        
        # Strategy selection logic
        if complexity_score > 0.7 and self.vision_available:
            return 'vision_primary'
        elif complexity_score > 0.4 and self.vision_available:
            return 'hybrid'
        else:
            return 'traditional'
    
    def _assess_document_complexity(self, file_path: str) -> float:
        """
        Assess document complexity for strategy selection.
        
        Complexity Factors:
        - Number of images per page
        - Text density
        - Layout complexity
        - Font variations
        - Table presence
        """
        
        try:
            import pymupdf as fitz
            
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return 1.0  # Empty document - high complexity
            
            complexity_factors = []
            
            # Sample first few pages for analysis
            sample_pages = min(3, total_pages)
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                
                # Factor 1: Image density
                images = page.get_images()
                image_density = len(images) / max(1, page.rect.width * page.rect.height / 100000)
                
                # Factor 2: Text extraction quality
                text = page.get_text()
                text_length = len(text.strip())
                
                # Factor 3: Drawing objects (potential tables/graphics)
                drawings = page.get_drawings()
                drawing_density = len(drawings) / max(1, page.rect.width * page.rect.height / 100000)
                
                # Factor 4: Font complexity
                fonts = page.get_fonts()
                font_variety = len(set(font[4] for font in fonts)) / max(1, len(fonts))
                
                # Combine factors
                page_complexity = (
                    min(image_density * 0.3, 0.3) +
                    min(drawing_density * 0.2, 0.2) +
                    min(font_variety * 0.2, 0.2) +
                    (0.3 if text_length < 100 else 0.0)  # Low text suggests scanned content
                )
                
                complexity_factors.append(page_complexity)
            
            doc.close()
            
            # Return average complexity
            return sum(complexity_factors) / len(complexity_factors)
            
        except Exception as e:
            logger.warning(f"Document complexity assessment failed: {e}")
            return 0.5  # Default to medium complexity
    
    def _execute_extraction(
        self, 
        file_path: str, 
        options: PDFExtractOptions, 
        strategy: str
    ) -> PDFExtractionResult:
        """
        Execute extraction using selected strategy.
        
        Strategy Implementations:
        - traditional: PyMuPDF text extraction
        - vision_only: Full vision-based processing
        - vision_primary: Vision with traditional fallback
        - hybrid: Combined approach with quality assessment
        """
        
        if strategy == 'traditional':
            return self._extract_traditional(file_path, options)
        
        elif strategy == 'vision_only':
            return self._extract_vision_only(file_path, options)
        
        elif strategy == 'vision_primary':
            return self._extract_vision_primary(file_path, options)
        
        elif strategy == 'hybrid':
            return self._extract_hybrid(file_path, options)
        
        else:
            raise ValueError(f"Unknown extraction strategy: {strategy}")
```

### Traditional PDF Extraction Implementation

```python
def _extract_traditional(self, file_path: str, options: PDFExtractOptions) -> PDFExtractionResult:
    """
    Traditional PDF text extraction using PyMuPDF.
    
    Extraction Features:
    - Multiple output formats (text, dict, json, html, xml)
    - Layout preservation options
    - Page range selection
    - Metadata extraction
    - Table detection
    """
    
    import pymupdf as fitz
    
    try:
        doc = fitz.open(file_path)
        
        # Determine page range
        start_page, end_page = self._resolve_page_range(doc, options.page_range)
        
        extracted_pages = []
        full_text = ""
        metadata = {}
        
        # Extract metadata
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc),
            'processed_pages': end_page - start_page
        }
        
        # Process each page
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            
            # Extract text based on output format
            if options.output_format == "text":
                page_text = page.get_text()
            elif options.output_format == "dict":
                page_text = page.get_text("dict")
            elif options.output_format == "json":
                page_text = page.get_text("json")
            elif options.output_format == "html":
                page_text = page.get_text("html")
            elif options.output_format == "xml":
                page_text = page.get_text("xml")
            else:
                page_text = page.get_text()
            
            # Apply layout preservation if requested
            if options.preserve_layout and isinstance(page_text, str):
                page_text = self._preserve_layout(page_text)
            
            extracted_pages.append({
                'page_number': page_num + 1,
                'content': page_text,
                'images': len(page.get_images()),
                'tables': self._detect_tables_traditional(page) if options.extract_tables else []
            })
            
            # Accumulate full text (for text format only)
            if isinstance(page_text, str):
                full_text += page_text + "\n\n"
        
        doc.close()
        
        # Create result
        result = PDFExtractionResult(
            text=full_text,
            pages=extracted_pages,
            metadata=metadata,
            page_count=len(extracted_pages),
            processing_method='traditional',
            extraction_quality=self._assess_extraction_quality(full_text),
            success=True
        )
        
        # Add semantic chunks if requested
        if options.create_semantic_chunks and self.chunking_available:
            result.semantic_chunks = self._create_semantic_chunks(full_text, options)
        
        return result
        
    except Exception as e:
        logger.error(f"Traditional extraction failed: {e}")
        raise RuntimeError(f"Traditional PDF extraction failed: {e}")

def _preserve_layout(self, text: str) -> str:
    """
    Preserve text layout through intelligent formatting.
    
    Layout Preservation:
    - Maintain line breaks
    - Preserve indentation
    - Keep table-like structures
    - Remove excessive whitespace
    """
    
    import re
    
    # Split into lines
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Preserve leading whitespace for indentation
        leading_space = len(line) - len(line.lstrip())
        content = line.strip()
        
        if content:
            # Maintain reasonable indentation
            preserved_indent = min(leading_space, 8)  # Max 8 spaces
            processed_line = ' ' * preserved_indent + content
            processed_lines.append(processed_line)
        else:
            # Keep empty lines but limit consecutive empty lines
            if not processed_lines or processed_lines[-1].strip():
                processed_lines.append('')
    
    return '\n'.join(processed_lines)

def _detect_tables_traditional(self, page) -> List[Dict[str, Any]]:
    """
    Detect tables using traditional PDF analysis.
    
    Table Detection:
    - Analyze drawing objects for table lines
    - Detect grid patterns
    - Extract table cell content
    - Format as structured data
    """
    
    try:
        tables = []
        
        # Get drawings (lines, rectangles)
        drawings = page.get_drawings()
        
        # Simple table detection based on rectangular patterns
        # This is a simplified implementation
        horizontal_lines = []
        vertical_lines = []
        
        for drawing in drawings:
            items = drawing.get('items', [])
            for item in items:
                if item[0] == 'l':  # Line
                    x1, y1, x2, y2 = item[1:5]
                    if abs(y1 - y2) < 1:  # Horizontal line
                        horizontal_lines.append((x1, x2, y1))
                    elif abs(x1 - x2) < 1:  # Vertical line
                        vertical_lines.append((y1, y2, x1))
        
        # If we have both horizontal and vertical lines, likely a table
        if len(horizontal_lines) > 1 and len(vertical_lines) > 1:
            tables.append({
                'type': 'detected_table',
                'horizontal_lines': len(horizontal_lines),
                'vertical_lines': len(vertical_lines),
                'bbox': self._calculate_table_bbox(horizontal_lines, vertical_lines)
            })
        
        return tables
        
    except Exception as e:
        logger.warning(f"Table detection failed: {e}")
        return []
```

### Vision-based Extraction Implementation

```python
def _extract_vision_only(self, file_path: str, options: PDFExtractOptions) -> PDFExtractionResult:
    """
    Vision-only extraction using AI vision models.
    
    Vision Processing:
    - Convert PDF pages to images
    - Process with vision AI
    - Extract text with bounding boxes
    - Handle complex layouts
    - Process tables and graphics
    """
    
    if not self.vision_available:
        raise RuntimeError("Vision processing not available")
    
    try:
        # Convert PDF to images
        images = self._pdf_to_images(file_path, options)
        
        # Determine page range
        start_page, end_page = self._resolve_page_range_from_images(images, options.page_range)
        
        extracted_pages = []
        full_text = ""
        
        # Process each page image
        for page_num in range(start_page, end_page):
            if page_num >= len(images):
                break
            
            page_image = images[page_num]
            
            # Create vision prompt based on options
            vision_prompt = self._create_vision_prompt(options)
            
            # Process with vision AI
            vision_result = self._process_page_with_vision(page_image, vision_prompt, options)
            
            extracted_pages.append({
                'page_number': page_num + 1,
                'content': vision_result['text'],
                'vision_analysis': vision_result.get('analysis', ''),
                'bounding_boxes': vision_result.get('bounding_boxes', []),
                'tables': vision_result.get('tables', []),
                'confidence': vision_result.get('confidence', 0.0)
            })
            
            full_text += vision_result['text'] + "\n\n"
        
        # Create result
        result = PDFExtractionResult(
            text=full_text,
            pages=extracted_pages,
            metadata={
                'extraction_method': 'vision_only',
                'pages_processed': len(extracted_pages),
                'vision_model': getattr(self.vision_client, 'model', 'unknown')
            },
            page_count=len(extracted_pages),
            processing_method='vision_only',
            extraction_quality=self._assess_extraction_quality(full_text),
            success=True
        )
        
        # Add semantic chunks if requested
        if options.create_semantic_chunks and self.chunking_available:
            result.semantic_chunks = self._create_semantic_chunks(full_text, options)
        
        return result
        
    except Exception as e:
        logger.error(f"Vision-only extraction failed: {e}")
        raise RuntimeError(f"Vision extraction failed: {e}")

def _pdf_to_images(self, file_path: str, options: PDFExtractOptions) -> List[str]:
    """
    Convert PDF pages to base64 images for vision processing.
    
    Conversion Process:
    - Load PDF document
    - Render pages at specified DPI
    - Convert to specified image format
    - Encode as base64 strings
    - Optimize for vision AI processing
    """
    
    import pymupdf as fitz
    import base64
    import io
    
    try:
        doc = fitz.open(file_path)
        images = []
        
        # Configure image conversion
        dpi = options.image_dpi or 150
        image_format = options.image_format or "png"
        
        # Matrix for DPI scaling
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image for processing
            img_data = pix.tobytes(image_format)
            
            # Encode as base64
            b64_string = base64.b64encode(img_data).decode('utf-8')
            images.append(b64_string)
        
        doc.close()
        return images
        
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        raise RuntimeError(f"PDF to image conversion failed: {e}")

def _create_vision_prompt(self, options: PDFExtractOptions) -> str:
    """
    Create optimized vision prompt based on extraction options.
    
    Prompt Engineering:
    - Task-specific instructions
    - Output format specifications
    - Quality requirements
    - Special handling instructions
    """
    
    if options.vision_prompt:
        return options.vision_prompt
    
    # Base prompt
    prompt_parts = [
        "Extract all visible text from this document page accurately.",
        "Maintain the original structure and formatting where possible."
    ]
    
    # Add specific instructions based on options
    if options.extract_tables:
        prompt_parts.append(
            "Pay special attention to tables - preserve table structure "
            "and clearly indicate table boundaries and cell contents."
        )
    
    if options.extract_images_text:
        prompt_parts.append(
            "Also extract any text that appears within images or graphics on the page."
        )
    
    if options.include_bounding_boxes:
        prompt_parts.append(
            "Include approximate position information for major text blocks."
        )
    
    # Output format instructions
    if options.output_format == "json":
        prompt_parts.append(
            "Format the output as structured JSON with clear hierarchies."
        )
    elif options.output_format == "html":
        prompt_parts.append(
            "Format the output as clean HTML preserving document structure."
        )
    
    return " ".join(prompt_parts)

def _process_page_with_vision(
    self, 
    page_image: str, 
    prompt: str, 
    options: PDFExtractOptions
) -> Dict[str, Any]:
    """
    Process single page image with vision AI.
    
    Processing Pipeline:
    1. Send image to vision model
    2. Parse response
    3. Extract structured information
    4. Validate and clean results
    5. Calculate confidence scores
    """
    
    try:
        # Call vision client
        vision_response = self.vision_client.analyze_image_base64(
            page_image, 
            prompt,
            **self._get_vision_parameters(options)
        )
        
        # Parse response
        parsed_result = self._parse_vision_response(vision_response, options)
        
        # Calculate confidence score
        confidence = self._calculate_vision_confidence(parsed_result, vision_response)
        
        return {
            'text': parsed_result.get('text', ''),
            'analysis': parsed_result.get('analysis', ''),
            'tables': parsed_result.get('tables', []),
            'bounding_boxes': parsed_result.get('bounding_boxes', []),
            'confidence': confidence,
            'raw_response': vision_response
        }
        
    except Exception as e:
        logger.error(f"Vision processing failed: {e}")
        return {
            'text': '',
            'analysis': f"Vision processing failed: {e}",
            'tables': [],
            'bounding_boxes': [],
            'confidence': 0.0
        }

def _get_vision_parameters(self, options: PDFExtractOptions) -> Dict[str, Any]:
    """Get vision model parameters based on options."""
    
    params = {
        'temperature': 0.1,  # Low temperature for consistency
        'max_tokens': 4000   # Sufficient for page content
    }
    
    # Adjust based on extraction requirements
    if options.extract_tables:
        params['max_tokens'] = 6000  # More tokens for table data
    
    return params
```

### Hybrid Extraction Implementation

```python
def _extract_hybrid(self, file_path: str, options: PDFExtractOptions) -> PDFExtractionResult:
    """
    Hybrid extraction combining traditional and vision approaches.
    
    Hybrid Strategy:
    1. Start with traditional extraction
    2. Assess extraction quality
    3. Use vision for low-quality pages
    4. Combine results intelligently
    5. Validate final output
    """
    
    try:
        # Step 1: Traditional extraction
        traditional_result = self._extract_traditional(file_path, options)
        
        # Step 2: Quality assessment
        page_qualities = self._assess_page_extraction_quality(traditional_result)
        
        # Step 3: Identify pages needing vision processing
        low_quality_pages = [
            i for i, quality in enumerate(page_qualities) 
            if quality < 0.6  # Quality threshold
        ]
        
        if not low_quality_pages:
            # Traditional extraction is sufficient
            traditional_result.processing_method = 'hybrid_traditional_only'
            return traditional_result
        
        # Step 4: Vision processing for low-quality pages
        vision_improvements = {}
        
        if self.vision_available:
            images = self._pdf_to_images(file_path, options)
            vision_prompt = self._create_vision_prompt(options)
            
            for page_idx in low_quality_pages:
                if page_idx < len(images):
                    vision_result = self._process_page_with_vision(
                        images[page_idx], 
                        vision_prompt, 
                        options
                    )
                    
                    # Use vision result if it's better
                    if vision_result['confidence'] > 0.5:
                        vision_improvements[page_idx] = vision_result
        
        # Step 5: Combine results
        final_result = self._combine_extraction_results(
            traditional_result, 
            vision_improvements,
            options
        )
        
        final_result.processing_method = 'hybrid'
        return final_result
        
    except Exception as e:
        logger.error(f"Hybrid extraction failed: {e}")
        # Fallback to traditional only
        return self._extract_traditional(file_path, options)

def _assess_page_extraction_quality(self, result: PDFExtractionResult) -> List[float]:
    """
    Assess extraction quality for each page.
    
    Quality Metrics:
    - Text length and density
    - Character variety
    - Word formation
    - Sentence structure
    - Special character ratio
    """
    
    qualities = []
    
    for page in result.pages:
        content = page.get('content', '')
        
        if not isinstance(content, str):
            content = str(content)
        
        # Basic quality metrics
        text_length = len(content.strip())
        
        if text_length == 0:
            qualities.append(0.0)
            continue
        
        # Character variety (higher is better)
        unique_chars = len(set(content.lower()))
        char_variety = min(unique_chars / 50.0, 1.0)  # Normalize to 50 unique chars
        
        # Word formation quality
        words = content.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            word_quality = min(avg_word_length / 6.0, 1.0)  # Normalize to 6 chars avg
        else:
            word_quality = 0.0
        
        # Special character ratio (lower is better for quality)
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(text_length, 1)
        special_quality = max(0.0, 1.0 - special_ratio * 2)  # Penalize high special char ratio
        
        # Sentence structure (presence of punctuation)
        sentence_endings = content.count('.') + content.count('!') + content.count('?')
        sentence_quality = min(sentence_endings / max(len(words) / 15, 1), 1.0)
        
        # Combine quality factors
        overall_quality = (
            char_variety * 0.3 +
            word_quality * 0.3 +
            special_quality * 0.2 +
            sentence_quality * 0.2
        )
        
        qualities.append(overall_quality)
    
    return qualities

def _combine_extraction_results(
    self,
    traditional_result: PDFExtractionResult,
    vision_improvements: Dict[int, Dict[str, Any]],
    options: PDFExtractOptions
) -> PDFExtractionResult:
    """
    Intelligently combine traditional and vision extraction results.
    
    Combination Strategy:
    - Use vision results for improved pages
    - Maintain traditional results for good pages
    - Merge metadata appropriately
    - Update quality assessments
    """
    
    # Start with traditional result
    combined_result = traditional_result
    combined_pages = traditional_result.pages.copy()
    
    # Apply vision improvements
    full_text_parts = []
    
    for i, page in enumerate(combined_pages):
        if i in vision_improvements:
            # Replace with vision result
            vision_data = vision_improvements[i]
            
            updated_page = {
                'page_number': page['page_number'],
                'content': vision_data['text'],
                'original_content': page['content'],  # Keep original for reference
                'vision_analysis': vision_data.get('analysis', ''),
                'bounding_boxes': vision_data.get('bounding_boxes', []),
                'tables': vision_data.get('tables', []),
                'confidence': vision_data.get('confidence', 0.0),
                'extraction_method': 'vision_enhanced'
            }
            
            combined_pages[i] = updated_page
            full_text_parts.append(vision_data['text'])
        else:
            # Keep traditional result
            page['extraction_method'] = 'traditional'
            content = page.get('content', '')
            if isinstance(content, str):
                full_text_parts.append(content)
            else:
                full_text_parts.append(str(content))
    
    # Rebuild full text
    combined_result.text = '\n\n'.join(full_text_parts)
    combined_result.pages = combined_pages
    
    # Update metadata
    combined_result.metadata.update({
        'hybrid_processing': True,
        'vision_enhanced_pages': len(vision_improvements),
        'total_pages': len(combined_pages)
    })
    
    # Recalculate quality
    combined_result.extraction_quality = self._assess_extraction_quality(combined_result.text)
    
    return combined_result
```

## 🖼️ Visual Content Processing

### VisualToTextService Implementation

```python
class VisualToTextService:
    """
    Unified visual content processing service.
    
    Supports:
    - PDF page processing
    - Image file processing
    - Base64 image processing
    - Multi-format output
    - Bounding box extraction
    """
    
    def __init__(self, credential_manager: Optional['CredentialManager'] = None):
        """Initialize visual processing with comprehensive format support."""
        
        self.credential_manager = credential_manager or default_credential_manager
        
        # Initialize vision client
        self.vision_client = self._initialize_vision_client()
        
        # Supported formats
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_pdf_formats = {'.pdf'}
        
        # Processing metrics
        self.processing_metrics = {
            'images_processed': 0,
            'pdfs_processed': 0,
            'base64_processed': 0,
            'total_processing_time': 0.0
        }
    
    def extract_text_from_file(
        self, 
        file_path: str, 
        options: Optional[VisualExtractOptions] = None
    ) -> VisualExtractionResult:
        """
        Extract text from any supported visual file format.
        
        Unified Processing:
        - Auto-detect file type
        - Apply appropriate processing pipeline
        - Return consistent result format
        - Handle errors gracefully
        """
        
        import time
        start_time = time.time()
        
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Setup options
            options = options or VisualExtractOptions()
            
            # Determine file type and processing method
            file_extension = pathlib.Path(file_path).suffix.lower()
            
            if file_extension in self.supported_pdf_formats:
                result = self._process_pdf_visual(file_path, options)
                self.processing_metrics['pdfs_processed'] += 1
            
            elif file_extension in self.supported_image_formats:
                result = self._process_image_file(file_path, options)
                self.processing_metrics['images_processed'] += 1
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_metrics['total_processing_time'] += processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Visual text extraction failed for {file_path}: {e}")
            raise RuntimeError(f"Visual text extraction failed: {e}")
    
    def extract_text_from_base64_image(
        self,
        base64_image: str,
        options: Optional[VisualExtractOptions] = None,
        image_format: str = "png"
    ) -> VisualExtractionResult:
        """
        Extract text from base64-encoded image.
        
        Base64 Processing:
        - Decode and validate image
        - Process with vision AI
        - Extract structured information
        - Return comprehensive results
        """
        
        import time
        start_time = time.time()
        
        try:
            options = options or VisualExtractOptions()
            
            # Validate base64 image
            self._validate_base64_image(base64_image)
            
            # Create vision prompt
            vision_prompt = self._create_visual_prompt(options)
            
            # Process with vision AI
            vision_response = self.vision_client.analyze_image_base64(
                base64_image,
                vision_prompt,
                **self._get_vision_parameters(options)
            )
            
            # Parse and structure response
            structured_result = self._parse_visual_response(vision_response, options)
            
            # Create result object
            result = VisualExtractionResult(
                text=structured_result.get('text', ''),
                bounding_boxes=structured_result.get('bounding_boxes', []),
                tables=structured_result.get('tables', []),
                metadata={
                    'processing_method': 'base64_vision',
                    'image_format': image_format,
                    'vision_model': getattr(self.vision_client, 'model', 'unknown'),
                    'processing_time': time.time() - start_time
                },
                confidence=structured_result.get('confidence', 0.0),
                success=True
            )
            
            # Update metrics
            self.processing_metrics['base64_processed'] += 1
            self.processing_metrics['total_processing_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Base64 image processing failed: {e}")
            return VisualExtractionResult(
                text="",
                error_message=str(e),
                success=False
            )
    
    def _create_visual_prompt(self, options: VisualExtractOptions) -> str:
        """
        Create optimized prompt for visual content processing.
        
        Prompt Engineering:
        - Content-specific instructions
        - Output format requirements
        - Quality specifications
        - Special feature requests
        """
        
        if options.vision_prompt:
            return options.vision_prompt
        
        prompt_parts = [
            "Analyze this image and extract all visible text content accurately."
        ]
        
        # Add specific extraction requirements
        if options.extract_tables:
            prompt_parts.append(
                "Pay special attention to any tables, preserving their structure "
                "and clearly organizing the data in rows and columns."
            )
        
        if options.include_bounding_boxes:
            prompt_parts.append(
                "Include approximate position information (bounding boxes) "
                "for major text regions and elements."
            )
        
        if options.extract_images_text:
            prompt_parts.append(
                "Also extract text that appears within any embedded images, "
                "logos, or graphical elements."
            )
        
        # Output format specification
        if options.output_format == "json":
            prompt_parts.append(
                "Structure the output as clean JSON with hierarchical organization."
            )
        elif options.output_format == "html":
            prompt_parts.append(
                "Format the output as semantic HTML preserving document structure."
            )
        elif options.output_format == "xml":
            prompt_parts.append(
                "Format the output as well-formed XML with clear element hierarchy."
            )
        
        return " ".join(prompt_parts)
```

## 🧩 Integration and Advanced Features

### RAG Integration System

```python
class RAGIntegrationService:
    """
    Specialized service for RAG (Retrieval-Augmented Generation) integration.
    
    RAG Features:
    - Optimized chunking strategies
    - Metadata enrichment
    - Vector embedding preparation
    - Document hierarchy preservation
    """
    
    def __init__(
        self,
        pdf_service: PDFToTextService,
        chunking_service: Optional['ChunkingService'] = None,
        embedding_service: Optional['EmbeddingService'] = None
    ):
        self.pdf_service = pdf_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
    
    def process_for_rag(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process PDF for RAG with optimized chunking and metadata.
        
        RAG Processing Pipeline:
        1. Extract text with metadata preservation
        2. Create semantic chunks with overlap
        3. Enrich chunks with metadata
        4. Generate embeddings (optional)
        5. Prepare for vector database storage
        """
        
        # Step 1: Extract text with comprehensive metadata
        extraction_options = PDFExtractOptions(
            preserve_layout=True,
            extract_tables=True,
            create_semantic_chunks=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        extraction_result = self.pdf_service.extract_text_from_file(
            file_path, 
            extraction_options
        )
        
        if not extraction_result.success:
            raise RuntimeError(f"PDF extraction failed: {extraction_result.error_message}")
        
        # Step 2: Process chunks for RAG
        rag_chunks = []
        
        # Use semantic chunks if available, otherwise create them
        if extraction_result.semantic_chunks:
            chunks = extraction_result.semantic_chunks
        else:
            chunks = self._create_fallback_chunks(extraction_result.text, chunk_size, chunk_overlap)
        
        # Step 3: Enrich each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': i,
                'source_file': file_path,
                'file_name': os.path.basename(file_path),
                'total_chunks': len(chunks),
                'character_count': len(chunk),
                'extraction_method': extraction_result.processing_method
            }
            
            # Add document metadata if available
            if include_metadata and extraction_result.metadata:
                chunk_metadata.update({
                    'document_title': extraction_result.metadata.get('title', ''),
                    'document_author': extraction_result.metadata.get('author', ''),
                    'document_pages': extraction_result.metadata.get('page_count', 0),
                    'creation_date': extraction_result.metadata.get('creation_date', '')
                })
            
            # Generate embedding if service available
            embedding = None
            if self.embedding_service:
                try:
                    embedding = self.embedding_service.generate_embedding(chunk)
                except Exception as e:
                    logger.warning(f"Embedding generation failed for chunk {i}: {e}")
            
            rag_chunk = {
                'text': chunk,
                'metadata': chunk_metadata,
                'embedding': embedding
            }
            
            rag_chunks.append(rag_chunk)
        
        return rag_chunks
    
    def _create_fallback_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Create chunks when semantic chunking is not available."""
        
        if self.chunking_service:
            from chunking import ChunkingConfig
            
            config = ChunkingConfig(
                target_size=chunk_size,
                tolerance=chunk_size // 5,
                preserve_paragraphs=True,
                preserve_sentences=True
            )
            
            self.chunking_service.config = config
            return self.chunking_service.chunk_text(text)
        else:
            # Simple character-based chunking with overlap
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                
                # Try to end at a sentence boundary
                if end < len(text):
                    last_sentence = chunk.rfind('.')
                    if last_sentence > chunk_size * 0.7:  # At least 70% of chunk size
                        chunk = chunk[:last_sentence + 1]
                        end = start + len(chunk)
                
                chunks.append(chunk.strip())
                start = end - chunk_overlap
                
                if start >= len(text):
                    break
            
            return [chunk for chunk in chunks if chunk.strip()]
```

This comprehensive documentation provides deep technical insight into the PDF to text service architecture, multiple extraction strategies, vision integration, and advanced features for production use.