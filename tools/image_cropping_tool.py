"""
Image cropping tool implementation.
"""

import os
import base64
from PIL import Image
import io
from .tool_types import (
    ToolSchema,
    ToolParameter,
    ParameterType,
    ToolExecutor,
    ToolCall,
    ToolResult,
    Tool,
)


class ImageCroppingExecutor(ToolExecutor):
    """Image cropping tool executor."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute image cropping based on bounding box coordinates."""
        try:
            # Get parameters
            image_path = tool_call.parameters.get("image_path")
            bounding_box = tool_call.parameters.get("bounding_box")
            output_path = tool_call.parameters.get("output_path")
            base64_image = tool_call.parameters.get("base64_image")

            # Validate input
            if not bounding_box or len(bounding_box) != 4:
                raise ValueError(
                    "bounding_box must be a list of 4 coordinates [x1, y1, x2, y2]"
                )

            if not image_path and not base64_image:
                raise ValueError("Either image_path or base64_image must be provided")

            # Load image
            if base64_image:
                # Decode base64 image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
            else:
                # Load from file path
                if not os.path.exists(image_path):
                    raise ValueError(f"Image file not found: {image_path}")
                image = Image.open(image_path)

            # Extract bounding box coordinates
            x1, y1, x2, y2 = bounding_box

            # Ensure coordinates are within image bounds
            width, height = image.size
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ensure proper ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Crop the image
            cropped_image = image.crop((x1, y1, x2, y2))

            # Save or return as base64
            if output_path:
                # Save to file
                cropped_image.save(output_path)

                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={
                        "success": True,
                        "output_path": output_path,
                        "cropped_dimensions": {
                            "width": cropped_image.width,
                            "height": cropped_image.height,
                        },
                        "bounding_box": [x1, y1, x2, y2],
                    },
                )
            else:
                # Return as base64
                buffer = io.BytesIO()
                cropped_image.save(buffer, format="PNG")
                cropped_base64 = base64.b64encode(buffer.getvalue()).decode()

                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={
                        "success": True,
                        "cropped_image_base64": cropped_base64,
                        "cropped_dimensions": {
                            "width": cropped_image.width,
                            "height": cropped_image.height,
                        },
                        "bounding_box": [x1, y1, x2, y2],
                    },
                )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e),
            )


def create_image_cropping_tool():
    """Create image cropping tool."""

    schema = ToolSchema(
        name="crop_image",
        description="Crop an image based on bounding box coordinates",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "image_path": ToolParameter(
                    type=ParameterType.STRING,
                    description="Path to the image file to crop",
                ),
                "base64_image": ToolParameter(
                    type=ParameterType.STRING,
                    description="Base64 encoded image data (alternative to image_path)",
                ),
                "bounding_box": ToolParameter(
                    type=ParameterType.ARRAY,
                    description="Bounding box coordinates as [x1, y1, x2, y2]",
                    items=ToolParameter(type=ParameterType.NUMBER),
                ),
                "output_path": ToolParameter(
                    type=ParameterType.STRING,
                    description="Output file path (optional - if not provided, returns base64)",
                ),
            },
            required=["bounding_box"],
        ),
    )

    return Tool(schema=schema, executor=ImageCroppingExecutor())
