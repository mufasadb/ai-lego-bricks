"""
PDF to base64 conversion tool implementation.
"""
import os
from typing import Dict, Any, List
from .tool_types import ToolSchema, ToolParameter, ParameterType, ToolExecutor, ToolCall, ToolResult, Tool


class PDFToBase64Executor(ToolExecutor):
    """PDF to base64 conversion tool executor."""
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute PDF to base64 conversion."""
        try:
            # Get parameters
            file_path = tool_call.parameters.get("file_path")
            dpi = tool_call.parameters.get("dpi", 150)
            
            # Validate input
            if not file_path:
                raise ValueError("file_path is required")
            
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Import the conversion function
            from pdf_to_text.visual_to_text_service import convert_pdf_to_base64_images
            
            # Convert PDF to base64 images
            base64_images = convert_pdf_to_base64_images(file_path, dpi=dpi)
            
            if not base64_images:
                raise ValueError("No images were extracted from the PDF")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={
                    "success": True,
                    "file_path": file_path,
                    "base64_images": base64_images,
                    "page_count": len(base64_images),
                    "dpi": dpi
                }
            )
                
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )


def create_pdf_to_base64_tool():
    """Create PDF to base64 conversion tool."""
    
    schema = ToolSchema(
        name="pdf_to_base64",
        description="Convert PDF document to base64 encoded images",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "file_path": ToolParameter(
                    type=ParameterType.STRING,
                    description="Path to the PDF file to convert"
                ),
                "dpi": ToolParameter(
                    type=ParameterType.INTEGER,
                    description="DPI resolution for conversion (default: 150)"
                )
            },
            required=["file_path"]
        )
    )
    
    return Tool(schema=schema, executor=PDFToBase64Executor())