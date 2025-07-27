import os
import pathlib
import base64
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pymupdf
from PIL import Image
import io

# Vision processing capabilities
try:
    from llm.generation_service import quick_generate_gemini
    from llm.vision_clients import GeminiVisionClient

    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Import existing chunking service
try:
    from chunking.chunking_service import ChunkingService, ChunkingConfig

    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    print("Warning: Chunking service not available. Chunking features disabled.")


class VisualExtractOptions(BaseModel):
    """Options for visual text extraction from images and PDFs"""

    output_format: str = Field(
        default="text",
        description="Output format: 'text', 'dict', 'json', 'html', 'xml'",
    )

    # PDF-specific options
    pdf_flags: int = Field(default=3, description="Text extraction flags for PDFs")
    preserve_layout: bool = Field(default=False, description="Preserve text layout")
    page_range: Optional[tuple] = Field(
        default=None, description="Page range as (start, end) tuple for PDFs"
    )

    # Vision processing options
    vision_prompt: Optional[str] = Field(
        default=None, description="Custom prompt for vision-based extraction"
    )
    extract_tables: bool = Field(
        default=False, description="Focus on extracting table data"
    )
    extract_images_text: bool = Field(
        default=False, description="Extract text from images within content"
    )
    include_bounding_boxes: bool = Field(
        default=False, description="Include bounding box information for text elements"
    )

    # Image processing options
    image_dpi: int = Field(default=150, description="DPI for PDF to image conversion")
    image_format: str = Field(
        default="png", description="Image format for PDF conversion"
    )

    # RAG Integration Options
    create_semantic_chunks: bool = Field(
        default=False, description="Create semantic chunks for RAG applications"
    )
    chunk_size: int = Field(
        default=1000, description="Target size for text chunks (characters)"
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks (characters)"
    )
    preserve_section_boundaries: bool = Field(
        default=True, description="Avoid breaking chunks mid-section"
    )


class VisualTextResult(BaseModel):
    """Result of visual text extraction"""

    text: str
    source_type: str  # "pdf", "image", "base64_image"
    metadata: Dict[str, Any]

    # PDF-specific results
    page_count: Optional[int] = None
    pages: Optional[List[Dict[str, Any]]] = None

    # Vision processing results
    vision_processing_used: bool = False
    extracted_tables: Optional[List[Dict[str, Any]]] = None
    processing_method: str = "traditional"  # "traditional", "vision"
    bounding_boxes: Optional[List[Dict[str, Any]]] = None

    # Image conversion results
    converted_images: Optional[List[str]] = None  # base64 encoded images

    # RAG Integration Results
    semantic_chunks: Optional[List[Dict[str, Any]]] = None


class VisualToTextService:
    """
    A service for extracting text from visual content including PDFs and images
    """

    def __init__(self):
        """Initialize the visual-to-text service"""
        self.supported_pdf_formats = [".pdf"]
        self.supported_image_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".webp",
        ]
        self._vision_client = None

    def extract_text_from_file(
        self, file_path: str, options: Optional[VisualExtractOptions] = None
    ) -> VisualTextResult:
        """
        Extract text from a visual file (PDF or image)

        Args:
            file_path: Path to the visual file
            options: Extraction options

        Returns:
            VisualTextResult with extracted text and metadata
        """
        if options is None:
            options = VisualExtractOptions()

        # Validate file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file extension
        file_ext = pathlib.Path(file_path).suffix.lower()

        if file_ext in self.supported_pdf_formats:
            return self._extract_from_pdf(file_path, options)
        elif file_ext in self.supported_image_formats:
            return self._extract_from_image(file_path, options)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: {self.supported_pdf_formats + self.supported_image_formats}"
            )

    def extract_text_from_pdf_bytes(
        self, pdf_bytes: bytes, options: Optional[VisualExtractOptions] = None
    ) -> VisualTextResult:
        """
        Extract text from PDF bytes

        Args:
            pdf_bytes: PDF file content as bytes
            options: Extraction options

        Returns:
            VisualTextResult with extracted text and metadata
        """
        if options is None:
            options = VisualExtractOptions()

        try:
            with pymupdf.open("pdf", pdf_bytes) as doc:
                return self._extract_text_from_pdf_document(doc, options)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF bytes: {str(e)}")

    def extract_text_from_base64_image(
        self, base64_image: str, options: Optional[VisualExtractOptions] = None
    ) -> VisualTextResult:
        """
        Extract text from base64 encoded image

        Args:
            base64_image: Base64 encoded image string
            options: Extraction options

        Returns:
            VisualTextResult with extracted text and metadata
        """
        if options is None:
            options = VisualExtractOptions()

        if not VISION_AVAILABLE:
            raise RuntimeError(
                "Vision processing not available. Please install required dependencies."
            )

        try:
            # Get vision client
            vision_client = self._get_vision_client()

            # Extract text using vision model
            text = self._extract_text_from_image_with_client(
                base64_image, options, vision_client
            )

            # Get image metadata
            try:
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                metadata = {
                    "format": image.format,
                    "mode": image.mode,
                    "size": image.size,
                    "width": image.width,
                    "height": image.height,
                }
            except Exception as e:
                metadata = {"error": f"Could not extract image metadata: {str(e)}"}

            result = VisualTextResult(
                text=text or "",
                source_type="base64_image",
                metadata=metadata,
                vision_processing_used=True,
                processing_method="vision",
            )

            # Create semantic chunks if requested
            if options.create_semantic_chunks and CHUNKING_AVAILABLE:
                result.semantic_chunks = self._create_semantic_chunks_with_service(
                    result.text, options
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Error processing base64 image: {str(e)}")

    def convert_pdf_to_base64_images(self, file_path: str, dpi: int = 150) -> List[str]:
        """
        Convert PDF pages to base64 encoded images

        Args:
            file_path: Path to PDF file
            dpi: DPI for image conversion

        Returns:
            List of base64 encoded image strings
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = pathlib.Path(file_path).suffix.lower()
        if file_ext not in self.supported_pdf_formats:
            raise ValueError(f"File must be a PDF. Got: {file_ext}")

        try:
            with pymupdf.open(file_path) as doc:
                base64_images = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=dpi)
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode()
                    base64_images.append(img_base64)
                return base64_images
        except Exception as e:
            raise RuntimeError(f"Error converting PDF to images: {str(e)}")

    def _extract_from_pdf(
        self, file_path: str, options: VisualExtractOptions
    ) -> VisualTextResult:
        """Extract text from PDF file"""
        try:
            with pymupdf.open(file_path) as doc:
                return self._extract_text_from_pdf_document(doc, options)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")

    def _extract_from_image(
        self, file_path: str, options: VisualExtractOptions
    ) -> VisualTextResult:
        """Extract text from image file"""
        if not VISION_AVAILABLE:
            raise RuntimeError(
                "Vision processing not available. Please install required dependencies."
            )

        try:
            # Convert image to base64
            with open(file_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()

            # Get image metadata
            try:
                image = Image.open(file_path)
                metadata = {
                    "format": image.format,
                    "mode": image.mode,
                    "size": image.size,
                    "width": image.width,
                    "height": image.height,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                }
            except Exception as e:
                metadata = {"error": f"Could not extract image metadata: {str(e)}"}

            # Get vision client
            vision_client = self._get_vision_client()

            # Extract text using vision model
            text = self._extract_text_from_image_with_client(
                img_base64, options, vision_client
            )

            result = VisualTextResult(
                text=text or "",
                source_type="image",
                metadata=metadata,
                vision_processing_used=True,
                processing_method="vision",
            )

            # Create semantic chunks if requested
            if options.create_semantic_chunks and CHUNKING_AVAILABLE:
                result.semantic_chunks = self._create_semantic_chunks_with_service(
                    result.text, options
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Error processing image: {str(e)}")

    def _extract_text_from_pdf_document(
        self, doc: pymupdf.Document, options: VisualExtractOptions
    ) -> VisualTextResult:
        """
        Internal method to extract text from a PyMuPDF document
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
        converted_images = []

        for page_num in range(start_page, end_page):
            page = doc[page_num]

            # Extract text based on format
            if options.output_format == "text":
                page_text = page.get_text(flags=options.pdf_flags)
            elif options.output_format == "dict":
                page_text = str(page.get_text("dict", flags=options.pdf_flags))
            elif options.output_format == "json":
                page_text = str(page.get_text("json", flags=options.pdf_flags))
            elif options.output_format == "html":
                page_text = page.get_text("html", flags=options.pdf_flags)
            elif options.output_format == "xml":
                page_text = page.get_text("xml", flags=options.pdf_flags)
            else:
                page_text = page.get_text(flags=options.pdf_flags)

            text_parts.append(page_text)

            # Convert page to base64 image if needed
            if VISION_AVAILABLE and (
                len(page_text.strip()) < 50 or options.include_bounding_boxes
            ):
                pix = page.get_pixmap(dpi=options.image_dpi)
                img_data = pix.tobytes(options.image_format)
                img_base64 = base64.b64encode(img_data).decode()
                converted_images.append(img_base64)

            # Store page data if requested
            if options.preserve_layout:
                pages_data.append(
                    {
                        "page_number": page_num + 1,
                        "text": page_text,
                        "bbox": list(page.rect),
                        "rotation": page.rotation,
                    }
                )

        # Join text with page separators
        if options.preserve_layout:
            combined_text = chr(12).join(text_parts)  # Form feed character as separator
        else:
            combined_text = "\n".join(text_parts)

        result = VisualTextResult(
            text=combined_text,
            source_type="pdf",
            page_count=page_count,
            metadata=metadata,
            pages=pages_data if options.preserve_layout else None,
            converted_images=converted_images if converted_images else None,
        )

        # Use vision processing if text extraction failed or was minimal
        if VISION_AVAILABLE and (
            len(combined_text.strip()) < 50 or options.include_bounding_boxes
        ):
            try:
                vision_result = self._process_pdf_with_vision(doc, options)
                if vision_result:
                    result.vision_processing_used = True
                    result.processing_method = "vision"
                    # Use vision text as primary if traditional extraction failed
                    if len(combined_text.strip()) < 50:
                        result.text = vision_result
            except Exception as e:
                print(f"Warning: Vision processing failed: {e}")

        # Create semantic chunks if requested
        if options.create_semantic_chunks and CHUNKING_AVAILABLE:
            result.semantic_chunks = self._create_semantic_chunks_with_service(
                result.text, options
            )

        return result

    def _process_pdf_with_vision(
        self, doc: pymupdf.Document, options: VisualExtractOptions
    ) -> Optional[str]:
        """Process PDF using vision models"""
        try:
            # Get vision client
            vision_client = self._get_vision_client()

            # Convert pages to images and process
            page_texts = []
            max_pages = min(doc.page_count, 5)  # Limit for API cost

            for page_num in range(max_pages):
                page = doc[page_num]
                # Convert page to image
                pix = page.get_pixmap(dpi=options.image_dpi)
                img_data = pix.tobytes(options.image_format)
                img_base64 = base64.b64encode(img_data).decode()

                # Process with vision model
                page_text = self._extract_text_from_image_with_client(
                    img_base64, options, vision_client
                )
                if page_text:
                    page_texts.append(page_text)

            return "\n\n".join(page_texts) if page_texts else None

        except Exception as e:
            print(f"Vision processing error: {e}")
            return None

    def _get_vision_client(self):
        """Get or create vision client instance"""
        if not VISION_AVAILABLE:
            raise RuntimeError(
                "Vision processing not available. Please install required dependencies."
            )

        if self._vision_client is None:
            self._vision_client = GeminiVisionClient()

        return self._vision_client

    def _extract_text_from_image_with_client(
        self, img_base64: str, options: VisualExtractOptions, vision_client
    ) -> Optional[str]:
        """Extract text from image using vision client"""
        try:
            # Build prompt based on options
            if options.vision_prompt:
                prompt = options.vision_prompt
            elif options.extract_tables:
                prompt = "Extract all table data from this image. Format as structured text with clear column separators."
            elif options.extract_images_text:
                prompt = "Extract all text visible in this image, including text within any embedded images or diagrams."
            elif options.include_bounding_boxes:
                prompt = "Extract all text from this image and provide bounding box coordinates for each text element. Format as: TEXT: [x1, y1, x2, y2]"
            else:
                prompt = "Extract all text from this image. Maintain the original structure and formatting as much as possible."

            # Use the vision client to process the image
            return vision_client.analyze_image(img_base64, prompt)

        except Exception as e:
            print(f"Image text extraction error: {e}")
            return None

    def _create_semantic_chunks_with_service(
        self, text: str, options: VisualExtractOptions
    ) -> List[Dict[str, Any]]:
        """Create semantic chunks using the existing chunking service"""
        try:
            # Configure the existing chunking service
            chunk_config = ChunkingConfig(
                target_size=options.chunk_size,
                tolerance=options.chunk_overlap,
                preserve_paragraphs=options.preserve_section_boundaries,
                preserve_sentences=True,
                preserve_words=True,
            )

            chunking_service = ChunkingService(chunk_config)
            chunk_texts = chunking_service.chunk_text(text)

            # Convert to the expected format with metadata
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunks.append(
                    {
                        "chunk_id": i,
                        "text": chunk_text,
                        "character_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "start_position": (
                            text.find(chunk_text[:50])
                            if len(chunk_text) > 50
                            else text.find(chunk_text)
                        ),
                    }
                )

            return chunks

        except Exception as e:
            print(f"Warning: Chunking with service failed: {e}")
            # Fallback to basic list if available
            if text:
                return [
                    {
                        "chunk_id": 0,
                        "text": text,
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "start_position": 0,
                    }
                ]
            return []


# Convenience functions for quick usage
def extract_text_from_visual(file_path: str, preserve_layout: bool = False) -> str:
    """
    Quick function to extract text from a visual file (PDF or image)

    Args:
        file_path: Path to visual file
        preserve_layout: Whether to preserve text layout

    Returns:
        Extracted text as string
    """
    service = VisualToTextService()
    options = VisualExtractOptions(preserve_layout=preserve_layout)
    result = service.extract_text_from_file(file_path, options)
    return result.text


def extract_text_from_base64_image(
    base64_image: str, include_bounding_boxes: bool = False
) -> str:
    """
    Quick function to extract text from base64 encoded image

    Args:
        base64_image: Base64 encoded image string
        include_bounding_boxes: Whether to include bounding box information

    Returns:
        Extracted text as string
    """
    service = VisualToTextService()
    options = VisualExtractOptions(include_bounding_boxes=include_bounding_boxes)
    result = service.extract_text_from_base64_image(base64_image, options)
    return result.text


def convert_pdf_to_images(file_path: str, dpi: int = 150) -> List[str]:
    """
    Convert PDF to base64 encoded images

    Args:
        file_path: Path to PDF file
        dpi: DPI for image conversion

    Returns:
        List of base64 encoded image strings
    """
    service = VisualToTextService()
    return service.convert_pdf_to_base64_images(file_path, dpi)


def extract_with_bounding_boxes(
    file_path_or_base64: str, is_base64: bool = False
) -> VisualTextResult:
    """
    Extract text with bounding box information

    Args:
        file_path_or_base64: File path or base64 encoded image
        is_base64: Whether input is base64 encoded

    Returns:
        VisualTextResult with bounding box information
    """
    service = VisualToTextService()
    options = VisualExtractOptions(include_bounding_boxes=True)

    if is_base64:
        return service.extract_text_from_base64_image(file_path_or_base64, options)
    else:
        return service.extract_text_from_file(file_path_or_base64, options)


def extract_tables_from_visual(
    file_path_or_base64: str, is_base64: bool = False
) -> VisualTextResult:
    """
    Extract table data from visual content

    Args:
        file_path_or_base64: File path or base64 encoded image
        is_base64: Whether input is base64 encoded

    Returns:
        VisualTextResult with table-focused extraction
    """
    service = VisualToTextService()
    options = VisualExtractOptions(extract_tables=True)

    if is_base64:
        return service.extract_text_from_base64_image(file_path_or_base64, options)
    else:
        return service.extract_text_from_file(file_path_or_base64, options)
