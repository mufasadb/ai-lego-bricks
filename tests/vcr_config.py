"""
VCR Configuration for AI Lego Bricks Testing

This module provides shared VCR configuration to ensure consistent
recording and security filtering across all tests.
"""

import os
from typing import Dict, Any
import vcr


def get_vcr_config() -> Dict[str, Any]:
    """
    Get standardized VCR configuration with security filtering.

    Returns:
        Dict containing VCR configuration options
    """
    return {
        # Recording mode - can be overridden by pytest --record-mode
        "record_mode": "once",
        # Filter sensitive headers
        "filter_headers": [
            "authorization",
            "x-api-key",
            "openai-api-key",
            "anthropic-api-key",
            "x-openai-api-key",
            "x-anthropic-api-key",
            "x-goog-api-key",  # Google API key header
            "google-api-key",
            "bearer",
            "token",
            "api-key",
            "client-secret",
            "secret",
            "password",
            "auth",
        ],
        # Filter sensitive query parameters
        "filter_query_parameters": [
            "api_key",
            "token",
            "key",
            "secret",
            "password",
            "auth",
            "client_secret",
        ],
        # Filter sensitive POST data
        "filter_post_data_parameters": [
            "api_key",
            "token",
            "key",
            "secret",
            "password",
            "client_secret",
            "auth",
        ],
        # Cassette library directory
        "cassette_library_dir": os.path.join(os.path.dirname(__file__), "cassettes"),
        # Decode compressed responses for easier inspection
        "decode_compressed_response": True,
        # Match on method, scheme, port, path, and query (exclude host for flexibility)
        "match_on": ["method", "scheme", "port", "path", "query"],
        # Custom serializer for better readability
        "serializer": "yaml",
        # Ignore certain hosts that shouldn't be recorded
        "ignore_hosts": ["localhost", "127.0.0.1"],
    }


def get_pytest_vcr_config() -> Dict[str, Any]:
    """
    Get pytest-specific VCR configuration for use with @pytest.mark.vcr.

    Returns:
        Dict containing pytest-recording configuration
    """
    base_config = get_vcr_config()

    # Additional configuration for pytest-recording
    pytest_config = {
        **base_config,
        # Custom before_record hook to sanitize sensitive data
        "before_record_request": sanitize_request,
        "before_record_response": sanitize_response,
    }

    return pytest_config


def get_unit_test_vcr_config() -> Dict[str, Any]:
    """
    Get unit test specific VCR configuration with relaxed host matching.

    For unit tests, we don't care about exact host matching since we're testing
    with recorded cassettes and want them to work regardless of local environment.

    Returns:
        Dict containing unit test VCR configuration
    """
    base_config = get_vcr_config()

    # Unit test specific configuration - exclude host from matching
    unit_config = {
        **base_config,
        # Match on everything except host, scheme, and port (allows localhost vs IP flexibility)
        "match_on": ["method", "path", "query"],
        # Add ignore_hosts for additional flexibility
        "ignore_hosts": ["localhost", "127.0.0.1", "100.83.40.11"],
        # Custom before_record hook to sanitize sensitive data
        "before_record_request": sanitize_request,
        "before_record_response": sanitize_response,
    }

    return unit_config


def sanitize_request(request):
    """
    Sanitize request before recording to cassette.

    Args:
        request: VCR request object

    Returns:
        Modified request object with sensitive data removed
    """
    import re

    # Sanitize URI - replace all IP addresses with localhost (except 127.0.0.1)
    if hasattr(request, "uri"):
        # Replace all IPv4 addresses except localhost
        ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

        def replace_ip(match):
            ip = match.group()
            # Don't replace localhost or 0.0.0.0
            if ip in ("127.0.0.1", "0.0.0.0"):
                return ip
            return "localhost"

        request.uri = re.sub(ip_pattern, replace_ip, request.uri)

    # Sanitize request body
    if hasattr(request, "body") and request.body:
        if isinstance(request.body, str):
            # Replace all IP addresses in body (except localhost)
            ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

            def replace_ip(match):
                ip = match.group()
                if ip in ("127.0.0.1", "0.0.0.0"):
                    return ip
                return "localhost"

            request.body = re.sub(ip_pattern, replace_ip, request.body)

    # Sanitize headers - replace IP addresses in host headers
    if hasattr(request, "headers"):
        for header_name, header_values in request.headers.items():
            if header_name.lower() == "host":
                # Clean list of header values
                cleaned_values = []
                for value in header_values:
                    ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

                    def replace_ip(match):
                        ip = match.group()
                        if ip in ("127.0.0.1", "0.0.0.0"):
                            return ip
                        return "localhost"

                    cleaned_value = re.sub(ip_pattern, replace_ip, value)
                    cleaned_values.append(cleaned_value)
                request.headers[header_name] = cleaned_values

    return request


def sanitize_response(response):
    """
    Sanitize response before recording to cassette.

    Args:
        response: VCR response object (can be dict or object)

    Returns:
        Modified response object with sensitive data removed
    """
    import re

    # Define IP pattern and replacement function
    ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

    def replace_ip(match):
        ip = match.group()
        if ip in ("127.0.0.1", "0.0.0.0"):
            return ip
        return "XXX.XXX.XXX.XXX"  # More obvious placeholder for IPs in responses

    # Define API key patterns for sanitization
    api_key_patterns = [
        (
            r"AIza[0-9A-Za-z_-]{35}",
            "AIzaSy**REDACTED_GOOGLE_API_KEY**",
        ),  # Google API keys
        (r"sk-[a-zA-Z0-9]{48}", "sk-**REDACTED_OPENAI_API_KEY**"),  # OpenAI API keys
        (
            r"anthropic-[a-zA-Z0-9-]{50,}",
            "anthropic-**REDACTED_ANTHROPIC_API_KEY**",
        ),  # Anthropic API keys
        (
            r"[a-zA-Z0-9]{32,}",
            lambda m: (
                "**REDACTED_API_KEY**"
                if len(m.group()) > 20
                and any(c.isdigit() for c in m.group())
                and any(c.isalpha() for c in m.group())
                else m.group()
            ),
        ),  # Generic long alphanumeric strings
    ]

    def sanitize_content(content_str):
        """Apply all sanitization patterns to content string."""
        # Replace IP addresses
        content_str = re.sub(ip_pattern, replace_ip, content_str)

        # Replace API keys
        for pattern, replacement in api_key_patterns:
            if callable(replacement):
                content_str = re.sub(pattern, replacement, content_str)
            else:
                content_str = re.sub(pattern, replacement, content_str)

        return content_str

    # Handle dict-style response (VCR internal format)
    if isinstance(response, dict) and "body" in response:
        body = response["body"]
        if isinstance(body, dict) and "string" in body:
            content = body["string"]
            if isinstance(content, bytes):
                # Decode bytes, sanitize, then re-encode
                content_str = content.decode("utf-8")
                sanitized_str = sanitize_content(content_str)
                response["body"]["string"] = sanitized_str.encode("utf-8")
            elif isinstance(content, str):
                # Sanitize string directly
                response["body"]["string"] = sanitize_content(content)

    # Handle object-style response (fallback)
    elif hasattr(response, "body") and response.body:
        if isinstance(response.body, dict) and "string" in response.body:
            content = response.body["string"]
            if isinstance(content, (str, bytes)):
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                response.body["string"] = sanitize_content(content)
        elif isinstance(response.body, str):
            response.body = sanitize_content(response.body)

    return response


def create_vcr_instance(**kwargs) -> vcr.VCR:
    """
    Create a VCR instance with default configuration.

    Args:
        **kwargs: Additional configuration to override defaults

    Returns:
        Configured VCR instance
    """
    config = get_vcr_config()
    config.update(kwargs)

    return vcr.VCR(**config)


# Common VCR decorators for different scenarios
def vcr_integration_test(**kwargs):
    """Decorator for integration tests that record real API calls."""
    config = {"record_mode": "new_episodes"}
    config.update(kwargs)
    return create_vcr_instance(**config).use_cassette


def vcr_unit_test(**kwargs):
    """Decorator for unit tests that only replay from cassettes."""
    config = {"record_mode": "none"}
    config.update(kwargs)
    return create_vcr_instance(**config).use_cassette
