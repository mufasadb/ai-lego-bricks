"""
Service for handling thinking tokens in LLM responses
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from agent_orchestration.models import ThinkingTokensMode


@dataclass
class ThinkingTokensResult:
    """Result of thinking tokens processing"""

    original_response: str
    clean_response: str
    thinking_content: Optional[str] = None
    has_thinking_tokens: bool = False
    processing_mode: ThinkingTokensMode = ThinkingTokensMode.AUTO
    delimiter_used: Optional[str] = None


class ThinkingTokensService:
    """Service for detecting and processing thinking tokens in LLM responses"""

    # Default delimiter patterns - ordered by priority
    DEFAULT_DELIMITERS = [
        "<thinking>",
        "<think>",
        "<reasoning>",
        "<reflection>",
        "**Thinking:**",
        "**Reasoning:**",
        "**Reflection:**",
    ]

    # Closing patterns for each delimiter
    CLOSING_PATTERNS = {
        "<thinking>": "</thinking>",
        "<think>": "</think>",
        "<reasoning>": "</reasoning>",
        "<reflection>": "</reflection>",
        "**Thinking:**": "\n\n",  # Ends at double newline
        "**Reasoning:**": "\n\n",
        "**Reflection:**": "\n\n",
    }

    def __init__(self, delimiters: Optional[List[str]] = None):
        """
        Initialize the thinking tokens service

        Args:
            delimiters: Custom delimiter patterns to use for detection
        """
        self.delimiters = delimiters or self.DEFAULT_DELIMITERS

    def detect_thinking_tokens(self, response: str) -> bool:
        """
        Detect if response contains thinking tokens

        Args:
            response: The LLM response to analyze

        Returns:
            True if thinking tokens are detected, False otherwise
        """
        response_lower = response.lower()

        for delimiter in self.delimiters:
            if delimiter.lower() in response_lower:
                return True

        return False

    def extract_thinking_tokens(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract thinking tokens from response

        Args:
            response: The LLM response to process

        Returns:
            Tuple of (thinking_content, clean_response)
        """
        thinking_content = None
        clean_response = response

        for delimiter in self.delimiters:
            delimiter_lower = delimiter.lower()
            response_lower = response.lower()

            # Find delimiter position
            start_pos = response_lower.find(delimiter_lower)
            if start_pos == -1:
                continue

            # Get closing pattern
            closing_pattern = self.CLOSING_PATTERNS.get(delimiter, "\n\n")

            if closing_pattern == "\n\n":
                # For markdown-style thinking blocks, find end at double newline
                thinking_start = start_pos + len(delimiter)
                thinking_end = response.find("\n\n", thinking_start)
                if thinking_end == -1:
                    thinking_end = len(response)

                # Extract thinking content
                thinking_content = response[thinking_start:thinking_end].strip()

                # Remove entire thinking block
                clean_response = (
                    response[:start_pos] + response[thinking_end:].lstrip()
                ).strip()

            else:
                # For XML-style thinking blocks
                end_pos = response_lower.find(closing_pattern.lower(), start_pos)
                if end_pos == -1:
                    # No closing tag found, treat rest as thinking
                    thinking_content = response[start_pos + len(delimiter):].strip()
                    clean_response = response[:start_pos].strip()
                else:
                    # Extract thinking content between tags
                    thinking_content = response[
                        start_pos + len(delimiter):end_pos
                    ].strip()

                    # Remove entire thinking block
                    clean_response = (
                        response[:start_pos]
                        + response[end_pos + len(closing_pattern):].lstrip()
                    ).strip()

            break  # Use first match

        return thinking_content, clean_response

    def process_response(
        self,
        response: str,
        mode: ThinkingTokensMode = ThinkingTokensMode.AUTO,
        preserve_for_structured: bool = False,
    ) -> ThinkingTokensResult:
        """
        Process LLM response according to thinking tokens mode

        Args:
            response: The LLM response to process
            mode: Processing mode for thinking tokens
            preserve_for_structured: Whether to preserve clean response for structured parsing

        Returns:
            ThinkingTokensResult with processed response
        """
        has_thinking = self.detect_thinking_tokens(response)

        if not has_thinking:
            return ThinkingTokensResult(
                original_response=response,
                clean_response=response,
                has_thinking_tokens=False,
                processing_mode=mode,
            )

        thinking_content, clean_response = self.extract_thinking_tokens(response)

        # Determine delimiter used
        delimiter_used = None
        for delimiter in self.delimiters:
            if delimiter.lower() in response.lower():
                delimiter_used = delimiter
                break

        # Auto mode logic
        if mode == ThinkingTokensMode.AUTO:
            # If preserve_for_structured is True, default to HIDE
            # Otherwise, if response is very long, default to HIDE
            # For short responses, default to SHOW
            if preserve_for_structured:
                mode = ThinkingTokensMode.HIDE
            elif len(response) > 2000:  # Long response
                mode = ThinkingTokensMode.HIDE
            else:
                mode = ThinkingTokensMode.SHOW

        # Process according to mode
        if mode == ThinkingTokensMode.SHOW:
            pass
        elif mode == ThinkingTokensMode.HIDE:
            pass
        elif mode == ThinkingTokensMode.EXTRACT:
            pass  # Return clean by default
        else:
            pass  # Fallback

        return ThinkingTokensResult(
            original_response=response,
            clean_response=clean_response,
            thinking_content=thinking_content,
            has_thinking_tokens=True,
            processing_mode=mode,
            delimiter_used=delimiter_used,
        )

    def format_extracted_response(self, result: ThinkingTokensResult) -> Dict[str, Any]:
        """
        Format thinking tokens result for consumption

        Args:
            result: ThinkingTokensResult to format

        Returns:
            Dictionary with formatted response data
        """
        formatted = {
            "response": result.clean_response,
            "has_thinking_tokens": result.has_thinking_tokens,
            "processing_mode": result.processing_mode.value,
        }

        if (
            result.processing_mode == ThinkingTokensMode.EXTRACT
            and result.thinking_content
        ):
            formatted["thinking_content"] = result.thinking_content
            formatted["original_response"] = result.original_response

        if result.processing_mode == ThinkingTokensMode.SHOW:
            formatted["response"] = result.original_response

        return formatted
