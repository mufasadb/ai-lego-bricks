from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, TypeVar, Type, Protocol, Generator
from pydantic import BaseModel
from enum import Enum
import json


class LLMProvider(str, Enum):
    """Supported LLM providers"""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"  # Future support
    ANTHROPIC = "anthropic"  # Future support


class VisionProvider(str, Enum):
    """Supported vision model providers"""

    GEMINI_VISION = "gemini_vision"
    LLAVA = "llava"
    OPENAI_VISION = "openai_vision"  # Future support


class ChatMessage(BaseModel):
    """Standard chat message format"""

    role: str  # 'user', 'assistant', 'system'
    content: str


class LLMConfig(BaseModel):
    """Configuration for LLM clients"""

    provider: LLMProvider
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 60.0
    extra_params: Dict[str, Any] = {}


class GenerationConfig(BaseModel):
    """Configuration for text generation"""

    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class VisionConfig(BaseModel):
    """Configuration for vision models"""

    provider: VisionProvider
    model: Optional[str] = None
    temperature: float = 0.1  # Lower for accuracy
    max_tokens: int = 2048
    timeout: float = 60.0
    extra_params: Dict[str, Any] = {}


class TextLLMClient(ABC):
    """Abstract base class for text-based LLM clients"""

    @abstractmethod
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Send a chat message and get response"""
        pass

    @abstractmethod
    def chat_with_messages(self, messages: List[ChatMessage]) -> str:
        """Send multiple messages and get response"""
        pass

    def chat_stream(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Generator[str, None, str]:
        """
        Send a chat message and get streaming response

        Args:
            message: The user message
            chat_history: Optional chat history

        Yields:
            str: Partial response chunks as they arrive

        Returns:
            str: Complete response when streaming is done
        """
        # Default implementation for clients that don't support streaming
        full_response = self.chat(message, chat_history)
        yield full_response
        return full_response

    def chat_with_messages_stream(
        self, messages: List[ChatMessage]
    ) -> Generator[str, None, str]:
        """
        Send multiple messages and get streaming response

        Args:
            messages: List of chat messages

        Yields:
            str: Partial response chunks as they arrive

        Returns:
            str: Complete response when streaming is done
        """
        # Default implementation for clients that don't support streaming
        full_response = self.chat_with_messages(messages)
        yield full_response
        return full_response


class VisionLLMClient(ABC):
    """Abstract base class for vision-capable LLM clients"""

    @abstractmethod
    def process_image(
        self, image_data: str, prompt: str, mime_type: str = "image/png"
    ) -> str:
        """Process an image with a text prompt"""
        pass

    @abstractmethod
    def process_image_with_messages(
        self, image_data: str, messages: List[ChatMessage], mime_type: str = "image/png"
    ) -> str:
        """Process an image with chat history"""
        pass


class EmbeddingClient(ABC):
    """Abstract base class for embedding generation"""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this client"""
        pass


# Structured Response Types
T = TypeVar("T", bound=BaseModel)


class StructuredResponseConfig(BaseModel):
    """Configuration for structured responses"""

    response_schema: Optional[Dict[str, Any]] = None
    retry_attempts: int = 3
    validation_strict: bool = True
    fallback_to_text: bool = True


class StructuredLLMClient(Protocol):
    """Protocol for LLM clients that support structured responses"""

    def chat_structured(
        self,
        message: str,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Chat with structured output based on schema"""
        ...

    def with_structured_output(self, schema: Type[T]) -> "StructuredLLMWrapper[T]":
        """Return a wrapper that ensures structured output of specified type"""
        ...


class StructuredLLMWrapper:
    """Wrapper that enforces structured output from any TextLLMClient"""

    def __init__(
        self,
        client: TextLLMClient,
        schema: Type[T],
        config: Optional[StructuredResponseConfig] = None,
    ):
        self.client = client
        self.schema = schema
        self.config = config or StructuredResponseConfig()

    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> T:
        """Chat with guaranteed structured response"""
        return self._enforce_structured_response(message, chat_history)

    def _enforce_structured_response(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> T:
        """Enforce structured response using schema validation"""
        schema_json = self.schema.model_json_schema()

        enhanced_prompt = self._build_structured_prompt(message, schema_json)

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat(enhanced_prompt, chat_history)

                # Try to extract JSON from response
                parsed_response = self._extract_json_from_response(response)

                # Validate against schema
                validated_response = self.schema.model_validate(parsed_response)
                return validated_response

            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    if self.config.fallback_to_text:
                        # Return a best-effort structured response
                        return self._create_fallback_response(
                            response if "response" in locals() else str(e)
                        )
                    else:
                        raise ValueError(
                            f"Failed to get structured response after {self.config.retry_attempts} attempts: {e}"
                        )

                # Modify prompt for retry
                enhanced_prompt = self._build_retry_prompt(
                    message, schema_json, attempt + 1
                )

    def _build_structured_prompt(self, message: str, schema: Dict[str, Any]) -> str:
        """Build prompt that encourages structured response"""
        return f"""
{message}

IMPORTANT: Respond with valid JSON that exactly matches this schema:
{json.dumps(schema, indent=2)}

Ensure your response is properly formatted JSON that can be parsed. Do not include any text before or after the JSON.

JSON Response:
"""

    def _build_retry_prompt(
        self, message: str, schema: Dict[str, Any], attempt: int
    ) -> str:
        """Build retry prompt with more emphasis on format"""
        return f"""
{message}

CRITICAL - ATTEMPT {attempt}: Your previous response was not valid JSON. 
You MUST respond with ONLY valid JSON that matches this exact schema:
{json.dumps(schema, indent=2)}

Requirements:
1. Use only valid JSON syntax
2. Include all required fields from the schema
3. Match the exact data types specified
4. Do not include any explanatory text outside the JSON

JSON Response:
"""

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling common formatting issues"""
        response = response.strip()

        # Try to find JSON within the response
        start_markers = ["{", "["]
        end_markers = ["}", "]"]

        # Find the first occurrence of a JSON start marker
        start_idx = -1
        for marker in start_markers:
            idx = response.find(marker)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx

        if start_idx == -1:
            raise ValueError("No JSON found in response")

        # Find the last occurrence of the corresponding end marker
        end_idx = -1
        for marker in end_markers:
            idx = response.rfind(marker)
            if idx != -1 and idx > end_idx:
                end_idx = idx

        if end_idx == -1:
            raise ValueError("Incomplete JSON in response")

        json_str = response[start_idx:end_idx + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try some common fixes
            json_str = self._fix_common_json_issues(json_str)
            return json.loads(json_str)

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Apply common fixes for malformed JSON"""
        # Remove trailing commas
        json_str = json_str.replace(",}", "}").replace(",]", "]")

        # Fix unescaped quotes in strings (basic attempt)
        # This is a simple heuristic and may not work for all cases
        lines = json_str.split("\n")
        for i, line in enumerate(lines):
            if (
                ":" in line
                and not line.strip().endswith(",")
                and not line.strip().endswith("{")
                and not line.strip().endswith("}")
            ):
                # This might be a string value, escape internal quotes
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1].strip()
                    if value_part.startswith('"') and value_part.endswith('"'):
                        # Escape internal quotes
                        value_content = value_part[1:-1]
                        value_content = value_content.replace('"', '\\"')
                        lines[i] = f'{key_part}: "{value_content}"'

        return "\n".join(lines)

    def _create_fallback_response(self, failed_response: str) -> T:
        """Create a fallback structured response when parsing fails"""
        # Create an instance with default values where possible
        try:
            # Try to create with minimal required fields
            default_dict = {}
            schema = self.schema.model_json_schema()

            if "properties" in schema:
                for field_name, field_info in schema["properties"].items():
                    if field_name in schema.get("required", []):
                        # Set default values based on type
                        if field_info.get("type") == "string":
                            default_dict[field_name] = (
                                failed_response[:100]
                                if failed_response
                                else "Error: Could not parse response"
                            )
                        elif field_info.get("type") == "number":
                            default_dict[field_name] = 0
                        elif field_info.get("type") == "boolean":
                            default_dict[field_name] = False
                        elif field_info.get("type") == "array":
                            default_dict[field_name] = []
                        elif field_info.get("type") == "object":
                            default_dict[field_name] = {}

            return self.schema.model_validate(default_dict)
        except Exception:
            # If even fallback fails, raise the original error
            raise ValueError(
                f"Could not create structured response from: {failed_response}"
            )


def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON schema"""
    return model.model_json_schema()
