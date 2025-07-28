"""
HTTP Request Service for AI Lego Bricks

This service provides a standardized interface for making HTTP requests across the system.
Supports async operations, credential integration, and comprehensive error handling.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import httpx
from urllib.parse import urlparse

try:
    from credentials.credential_manager import CredentialManager
except ImportError:
    CredentialManager = None

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    """Supported HTTP methods"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class HttpRequestConfig:
    """Configuration for HTTP requests"""

    url: str
    method: HttpMethod = HttpMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    json_data: Optional[Dict[str, Any]] = None
    form_data: Optional[Dict[str, Any]] = None
    data: Optional[Union[str, bytes]] = None
    timeout: float = 30.0
    follow_redirects: bool = True
    verify_ssl: bool = True
    auth_type: Optional[str] = None  # 'bearer', 'basic', 'api_key'
    auth_credentials: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.url:
            raise ValueError("URL is required")

        # Validate URL format
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {self.url}")

        # Ensure mutually exclusive data fields
        data_fields = [self.json_data, self.form_data, self.data]
        if sum(x is not None for x in data_fields) > 1:
            raise ValueError(
                "Only one of json_data, form_data, or data can be specified"
            )


@dataclass
class HttpResponse:
    """HTTP response wrapper"""

    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    url: str
    method: str
    elapsed_time: float

    @property
    def is_success(self) -> bool:
        """Check if response indicates success (2xx status code)"""
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error (4xx status code)"""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error (5xx status code)"""
        return 500 <= self.status_code < 600

    def json(self) -> Dict[str, Any]:
        """Parse response as JSON"""
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}")

    def raise_for_status(self):
        """Raise an exception for HTTP error status codes"""
        if not self.is_success:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} error for {self.method} {self.url}",
                request=None,
                response=None,
            )


class HttpRequestService:
    """
    Service for making HTTP requests with credential integration and standardized error handling.
    """

    def __init__(
        self,
        credential_manager: Optional[CredentialManager] = None,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        """
        Initialize the HTTP request service.

        Args:
            credential_manager: Optional credential manager for authentication
            default_timeout: Default timeout for requests
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for retry delays
        """
        self.credential_manager = credential_manager
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(default_timeout), follow_redirects=True, verify=True
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    def _prepare_auth_headers(self, config: HttpRequestConfig) -> Dict[str, str]:
        """Prepare authentication headers based on configuration"""
        headers = config.headers.copy()

        if not config.auth_type or not config.auth_credentials:
            return headers

        if config.auth_type == "bearer":
            token = config.auth_credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif config.auth_type == "basic":
            username = config.auth_credentials.get("username")
            password = config.auth_credentials.get("password")
            if username and password:
                import base64

                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

        elif config.auth_type == "api_key":
            api_key = config.auth_credentials.get("api_key")
            key_header = config.auth_credentials.get("header_name", "X-API-Key")
            if api_key:
                headers[key_header] = api_key

        return headers

    def _resolve_credentials(self, config: HttpRequestConfig) -> HttpRequestConfig:
        """Resolve credentials from credential manager if available"""
        if not self.credential_manager or not config.auth_type:
            return config

        # Create a copy to avoid modifying original
        auth_credentials = config.auth_credentials or {}

        if config.auth_type == "bearer":
            # Look for common token credential names
            token_keys = ["token", "access_token", "bearer_token", "auth_token"]
            for key in token_keys:
                if key not in auth_credentials:
                    token = self.credential_manager.get_credential(key.upper())
                    if token:
                        auth_credentials["token"] = token
                        break

        elif config.auth_type == "basic":
            # Look for basic auth credentials
            if "username" not in auth_credentials:
                username = self.credential_manager.get_credential("USERNAME")
                if username:
                    auth_credentials["username"] = username

            if "password" not in auth_credentials:
                password = self.credential_manager.get_credential("PASSWORD")
                if password:
                    auth_credentials["password"] = password

        elif config.auth_type == "api_key":
            # Look for API key
            if "api_key" not in auth_credentials:
                api_key = self.credential_manager.get_credential("API_KEY")
                if api_key:
                    auth_credentials["api_key"] = api_key

        # Return updated config
        return HttpRequestConfig(
            url=config.url,
            method=config.method,
            headers=config.headers,
            params=config.params,
            json_data=config.json_data,
            form_data=config.form_data,
            data=config.data,
            timeout=config.timeout,
            follow_redirects=config.follow_redirects,
            verify_ssl=config.verify_ssl,
            auth_type=config.auth_type,
            auth_credentials=auth_credentials,
        )

    async def request(self, config: HttpRequestConfig) -> HttpResponse:
        """
        Make an HTTP request with retry logic and error handling.

        Args:
            config: HTTP request configuration

        Returns:
            HttpResponse object with response data

        Raises:
            ValueError: For invalid configuration
            httpx.HTTPStatusError: For HTTP error responses
            httpx.RequestError: For network/connection errors
        """
        # Resolve credentials
        config = self._resolve_credentials(config)

        # Prepare headers with authentication
        headers = self._prepare_auth_headers(config)

        # Set content type for JSON data
        if config.json_data:
            headers["Content-Type"] = "application/json"

        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                import time

                start_time = time.time()

                # Make the request
                response = await self.client.request(
                    method=config.method.value,
                    url=config.url,
                    headers=headers,
                    params=config.params,
                    json=config.json_data,
                    data=config.form_data or config.data,
                    timeout=config.timeout,
                    follow_redirects=config.follow_redirects,
                )

                elapsed_time = time.time() - start_time

                # Create response object
                http_response = HttpResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                    text=response.text,
                    url=str(response.url),
                    method=config.method.value,
                    elapsed_time=elapsed_time,
                )

                logger.info(
                    f"HTTP {config.method.value} {config.url} -> {response.status_code} ({elapsed_time:.2f}s)"
                )

                return http_response

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2**attempt)
                    logger.warning(
                        f"HTTP request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"HTTP request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in HTTP request: {e}")
                raise

        # This should never be reached, but just in case
        raise last_exception

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> HttpResponse:
        """Convenience method for GET requests"""
        config = HttpRequestConfig(
            url=url,
            method=HttpMethod.GET,
            params=params or {},
            headers=headers or {},
            **kwargs,
        )
        return await self.request(config)

    async def post(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        form_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> HttpResponse:
        """Convenience method for POST requests"""
        config = HttpRequestConfig(
            url=url,
            method=HttpMethod.POST,
            json_data=json_data,
            form_data=form_data,
            data=data,
            headers=headers or {},
            **kwargs,
        )
        return await self.request(config)

    async def put(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        form_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> HttpResponse:
        """Convenience method for PUT requests"""
        config = HttpRequestConfig(
            url=url,
            method=HttpMethod.PUT,
            json_data=json_data,
            form_data=form_data,
            data=data,
            headers=headers or {},
            **kwargs,
        )
        return await self.request(config)

    async def delete(
        self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> HttpResponse:
        """Convenience method for DELETE requests"""
        config = HttpRequestConfig(
            url=url, method=HttpMethod.DELETE, headers=headers or {}, **kwargs
        )
        return await self.request(config)

    async def patch(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        form_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> HttpResponse:
        """Convenience method for PATCH requests"""
        config = HttpRequestConfig(
            url=url,
            method=HttpMethod.PATCH,
            json_data=json_data,
            form_data=form_data,
            data=data,
            headers=headers or {},
            **kwargs,
        )
        return await self.request(config)

    async def head(
        self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> HttpResponse:
        """Convenience method for HEAD requests"""
        config = HttpRequestConfig(
            url=url, method=HttpMethod.HEAD, headers=headers or {}, **kwargs
        )
        return await self.request(config)

    async def options(
        self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> HttpResponse:
        """Convenience method for OPTIONS requests"""
        config = HttpRequestConfig(
            url=url, method=HttpMethod.OPTIONS, headers=headers or {}, **kwargs
        )
        return await self.request(config)


# Factory function for creating HTTP request service instances
def create_http_request_service(
    credential_manager: Optional[CredentialManager] = None, **kwargs
) -> HttpRequestService:
    """
    Factory function to create an HTTP request service instance.

    Args:
        credential_manager: Optional credential manager for authentication
        **kwargs: Additional configuration parameters

    Returns:
        HttpRequestService instance
    """
    return HttpRequestService(credential_manager=credential_manager, **kwargs)


# Convenience functions for quick HTTP requests
async def quick_get(url: str, **kwargs) -> HttpResponse:
    """Quick GET request without creating a service instance"""
    async with HttpRequestService() as service:
        return await service.get(url, **kwargs)


async def quick_post(url: str, **kwargs) -> HttpResponse:
    """Quick POST request without creating a service instance"""
    async with HttpRequestService() as service:
        return await service.post(url, **kwargs)


async def quick_request(config: HttpRequestConfig) -> HttpResponse:
    """Quick request with full configuration without creating a service instance"""
    async with HttpRequestService() as service:
        return await service.request(config)


# Example usage
if __name__ == "__main__":

    async def main():
        # Example 1: Basic GET request
        print("=== Basic GET Request ===")
        async with HttpRequestService() as service:
            response = await service.get("https://httpbin.org/get")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

        # Example 2: POST request with JSON data
        print("\n=== POST Request with JSON ===")
        async with HttpRequestService() as service:
            response = await service.post(
                "https://httpbin.org/post",
                json_data={"key": "value", "message": "Hello World"},
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

        # Example 3: Request with authentication
        print("\n=== Request with Authentication ===")
        config = HttpRequestConfig(
            url="https://httpbin.org/bearer",
            method=HttpMethod.GET,
            auth_type="bearer",
            auth_credentials={"token": "my-secret-token"},
        )
        async with HttpRequestService() as service:
            response = await service.request(config)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

        # Example 4: Quick convenience function
        print("\n=== Quick GET Function ===")
        response = await quick_get("https://httpbin.org/get", params={"test": "value"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

    asyncio.run(main())
