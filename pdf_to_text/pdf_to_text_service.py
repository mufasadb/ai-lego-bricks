import os
import pathlib
import base64
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pymupdf

# Vision processing capabilities for image-heavy PDFs
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))
    from generation_service import quick_generate_gemini
    from vision_clients import GeminiVisionClient
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Import existing chunking service
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chunking'))
    from chunking_service import ChunkingService, ChunkingConfig
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    print("Warning: Chunking service not available. Chunking features disabled.")


class PDFExtractOptions(BaseModel):
    """Options for PDF text extraction"""
    output_format: str = Field(default="text", description="Output format: 'text', 'dict', 'json', 'html', 'xml'")
    flags: int = Field(default=3, description="Text extraction flags")
    include_images: bool = Field(default=False, description="Include image descriptions")
    preserve_layout: bool = Field(default=False, description="Preserve text layout")
    page_range: Optional[tuple] = Field(default=None, description="Page range as (start, end) tuple")
    
    
    # Vision-based Processing Options (for image-heavy PDFs)
    use_vision_fallback: bool = Field(default=False, description="Use vision models when text extraction fails")
    force_vision_processing: bool = Field(default=False, description="Force use of vision models even for text-based PDFs")
    vision_prompt: Optional[str] = Field(default=None, description="Custom prompt for vision-based extraction")
    extract_tables: bool = Field(default=False, description="Focus on extracting table data")
    extract_images_text: bool = Field(default=False, description="Extract text from images within PDF")
    
    
    # RAG Integration Options
    create_semantic_chunks: bool = Field(default=False, description="Create semantic chunks for RAG applications")
    chunk_size: int = Field(default=1000, description="Target size for text chunks (characters)")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks (characters)")
    preserve_section_boundaries: bool = Field(default=True, description="Avoid breaking chunks mid-section")


class PDFTextResult(BaseModel):
    """Result of PDF text extraction"""
    text: str
    page_count: int
    metadata: Dict[str, Any]
    pages: Optional[List[Dict[str, Any]]] = None
    
    
    # Vision Processing Results
    vision_text: Optional[str] = None
    vision_processing_used: bool = False
    extracted_tables: Optional[List[Dict[str, Any]]] = None
    processing_method: str = "traditional"  # "traditional", "vision"
    
    
    # RAG Integration Results
    semantic_chunks: Optional[List[Dict[str, Any]]] = None


class PDFToTextService:
    """
    A service for extracting text from PDF documents using PyMuPDF
    """
    
    def __init__(self):
        """Initialize the PDF-to-text service"""
        self.supported_formats = ['.pdf']
        self._vision_client = None
    
    def extract_text_from_file(self, file_path: str, options: Optional[PDFExtractOptions] = None) -> PDFTextResult:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            options: Extraction options
            
        Returns:
            PDFTextResult with extracted text and metadata
        """
        if options is None:
            options = PDFExtractOptions()
        
        # Validate file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        file_ext = pathlib.Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.supported_formats}")
        
        # Open and process PDF
        try:
            with pymupdf.open(file_path) as doc:
                return self._extract_text_from_document(doc, options)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")
    
    def extract_text_from_bytes(self, pdf_bytes: bytes, options: Optional[PDFExtractOptions] = None) -> PDFTextResult:
        """
        Extract text from PDF bytes
        
        Args:
            pdf_bytes: PDF file content as bytes
            options: Extraction options
            
        Returns:
            PDFTextResult with extracted text and metadata
        """
        if options is None:
            options = PDFExtractOptions()
        
        try:
            with pymupdf.open("pdf", pdf_bytes) as doc:
                return self._extract_text_from_document(doc, options)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF bytes: {str(e)}")
    
    def _extract_text_from_document(self, doc: pymupdf.Document, options: PDFExtractOptions) -> PDFTextResult:
        """
        Internal method to extract text from a PyMuPDF document
        
        Args:
            doc: PyMuPDF document object
            options: Extraction options
            
        Returns:
            PDFTextResult with extracted text and metadata
        """
        # Get metadata
        metadata = doc.metadata
        page_count = doc.page_count
        
        # Determine page range
        start_page = 0
        end_page = page_count
        
        if options.page_range:
            start_page = max(0, options.page_range[0])
            end_page = min(page_count, options.page_range[1])
        
        # Extract text
        text_parts = []
        pages_data = []
        
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            
            # Extract text based on format
            if options.output_format == "text":
                page_text = page.get_text(flags=options.flags)
            elif options.output_format == "dict":
                page_text = str(page.get_text("dict", flags=options.flags))
            elif options.output_format == "json":
                page_text = str(page.get_text("json", flags=options.flags))
            elif options.output_format == "html":
                page_text = page.get_text("html", flags=options.flags)
            elif options.output_format == "xml":
                page_text = page.get_text("xml", flags=options.flags)
            else:
                page_text = page.get_text(flags=options.flags)
            
            text_parts.append(page_text)
            
            # Store page data if requested
            if options.preserve_layout:
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "bbox": list(page.rect),
                    "rotation": page.rotation
                })
        
        # Join text with page separators
        if options.preserve_layout:
            combined_text = chr(12).join(text_parts)  # Form feed character as separator
        else:
            combined_text = "\n".join(text_parts)
        
        result = PDFTextResult(
            text=combined_text,
            page_count=page_count,
            metadata=metadata,
            pages=pages_data if options.preserve_layout else None
        )
        
        # Check if we should use vision processing for image-heavy PDFs
        should_use_vision = (
            options.force_vision_processing or 
            (options.use_vision_fallback and len(combined_text.strip()) < 50)  # Very little text extracted
        )
        
        if should_use_vision and VISION_AVAILABLE:
            try:
                vision_result = self._process_with_vision(doc, options)
                if vision_result:
                    result.vision_text = vision_result
                    result.vision_processing_used = True
                    result.processing_method = "vision"
                    # Use vision text as primary if traditional extraction failed
                    if len(combined_text.strip()) < 50:
                        result.text = vision_result
            except Exception as e:
                print(f"Warning: Vision processing failed: {e}")
        
        # Create semantic chunks if requested
        if options.create_semantic_chunks and CHUNKING_AVAILABLE:
            text_to_chunk = result.text
            result.semantic_chunks = self._create_semantic_chunks_with_service(
                text_to_chunk, options
            )
        
        return result
    
    def extract_text_to_file(self, input_path: str, output_path: str, options: Optional[PDFExtractOptions] = None) -> None:
        """
        Extract text from PDF and save to file
        
        Args:
            input_path: Path to input PDF file
            output_path: Path to output text file
            options: Extraction options
        """
        result = self.extract_text_from_file(input_path, options)
        
        # Write to file with UTF-8 encoding
        pathlib.Path(output_path).write_text(result.text, encoding='utf-8')
    
    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get PDF document information without extracting text
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata and info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with pymupdf.open(file_path) as doc:
                return {
                    "page_count": doc.page_count,
                    "metadata": doc.metadata,
                    "is_encrypted": doc.is_encrypted,
                    "is_pdf": doc.is_pdf,
                    "file_size": os.path.getsize(file_path)
                }
        except Exception as e:
            raise RuntimeError(f"Error reading PDF info: {str(e)}")
    
    
    def _get_vision_client(self):
        """Get or create vision client instance"""
        if not VISION_AVAILABLE:
            raise RuntimeError("Vision processing not available. Please install required dependencies.")
        
        if self._vision_client is None:
            self._vision_client = GeminiVisionClient()
        
        return self._vision_client
    
    
    
    
    
    
    
    def _create_semantic_chunks_with_service(self, text: str, options: PDFExtractOptions) -> List[Dict[str, Any]]:
        """Create semantic chunks using the existing chunking service"""
        try:
            # Configure the existing chunking service
            chunk_config = ChunkingConfig(
                target_size=options.chunk_size,
                tolerance=options.chunk_overlap,
                preserve_paragraphs=options.preserve_section_boundaries,
                preserve_sentences=True,
                preserve_words=True
            )
            
            chunking_service = ChunkingService(chunk_config)
            chunk_texts = chunking_service.chunk_text(text)
            
            # Convert to the expected format with metadata
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunks.append({
                    "chunk_id": i,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "start_position": text.find(chunk_text[:50]) if len(chunk_text) > 50 else text.find(chunk_text)
                })
            
            return chunks
            
        except Exception as e:
            print(f"Warning: Chunking with service failed: {e}")
            # Fallback to basic list if available
            if text:
                return [{
                    "chunk_id": 0,
                    "text": text,
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "start_position": 0
                }]
            return []
    
    def _process_with_vision(self, doc: pymupdf.Document, options: PDFExtractOptions) -> Optional[str]:
        """Process PDF using vision models for image-heavy PDFs"""
        try:
            # Get vision client
            vision_client = self._get_vision_client()
            
            # Convert first few pages to images and process
            page_texts = []
            max_pages = min(doc.page_count, 5)  # Limit for API cost
            
            for page_num in range(max_pages):
                page = doc[page_num]
                # Convert page to image
                pix = page.get_pixmap(dpi=150)  # Good quality for OCR
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode()
                
                # Process with vision model
                page_text = self._extract_text_from_image_with_client(img_base64, options, vision_client)
                if page_text:
                    page_texts.append(page_text)
            
            return "\n\n".join(page_texts) if page_texts else None
            
        except Exception as e:
            print(f"Vision processing error: {e}")
            return None
    
    def _extract_text_from_image_with_client(self, img_base64: str, options: PDFExtractOptions, 
                                           vision_client) -> Optional[str]:
        """Extract text from image using vision client"""
        try:
            # Build prompt based on options
            if options.vision_prompt:
                prompt = options.vision_prompt
            elif options.extract_tables:
                prompt = "Extract all table data from this image. Format as structured text with clear column separators."
            elif options.extract_images_text:
                prompt = "Extract all text visible in this image, including text within any embedded images or diagrams."
            else:
                prompt = "Extract all text from this PDF page image. Maintain the original structure and formatting as much as possible."
            
            # Use the vision client to process the image
            return vision_client.analyze_image(img_base64, prompt)
                    
        except Exception as e:
            print(f"Image text extraction error: {e}")
            return None


# Convenience functions for quick usage
def extract_text_from_pdf(file_path: str, preserve_layout: bool = False) -> str:
    """
    Quick function to extract text from a PDF file
    
    Args:
        file_path: Path to PDF file
        preserve_layout: Whether to preserve text layout
        
    Returns:
        Extracted text as string
    """
    service = PDFToTextService()
    options = PDFExtractOptions(preserve_layout=preserve_layout)
    result = service.extract_text_from_file(file_path, options)
    return result.text


def pdf_to_text_file(input_path: str, output_path: str = None) -> str:
    """
    Convert PDF to text file
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output text file (optional)
        
    Returns:
        Path to output file
    """
    if output_path is None:
        output_path = input_path + ".txt"
    
    service = PDFToTextService()
    service.extract_text_to_file(input_path, output_path)
    return output_path


def extract_with_vision_fallback(file_path: str) -> PDFTextResult:
    """
    Extract text with automatic vision fallback for challenging PDFs
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        PDFTextResult with best available extraction method
    """
    service = PDFToTextService()
    options = PDFExtractOptions(
        use_vision_fallback=True
    )
    return service.extract_text_from_file(file_path, options)








def extract_tables_from_pdf(file_path: str) -> PDFTextResult:
    """
    Extract table data from PDF using vision processing
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        PDFTextResult with table-focused extraction
    """
    service = PDFToTextService()
    options = PDFExtractOptions(
        force_vision_processing=True,
        extract_tables=True
    )
    return service.extract_text_from_file(file_path, options)


def extract_text_from_scanned_pdf(file_path: str) -> str:
    """
    Extract text from scanned/image-based PDFs using vision processing
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text string
    """
    service = PDFToTextService()
    options = PDFExtractOptions(
        force_vision_processing=True,
        extract_images_text=True
    )
    result = service.extract_text_from_file(file_path, options)
    return result.vision_text or result.text




def extract_pdf_for_rag(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Extract and chunk PDF for RAG applications using the existing chunking service
    
    Args:
        file_path: Path to PDF file
        chunk_size: Target size for chunks (characters)
        chunk_overlap: Overlap tolerance for chunks (characters)
        
    Returns:
        List of semantic chunks ready for vector embedding
    """
    if not CHUNKING_AVAILABLE:
        print("Warning: Chunking service not available. Returning full text as single chunk.")
        
    service = PDFToTextService()
    options = PDFExtractOptions(
        create_semantic_chunks=CHUNKING_AVAILABLE,  # Only enable if service available
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_section_boundaries=True,
        use_vision_fallback=True
    )
    result = service.extract_text_from_file(file_path, options)
    
    # Return chunks if available, otherwise return full text as single chunk
    if result.semantic_chunks:
        return result.semantic_chunks
    else:
        # Fallback: return full text as single chunk
        return [{
            "chunk_id": 0,
            "text": result.text,
            "character_count": len(result.text),
            "word_count": len(result.text.split()),
            "start_position": 0
        }]