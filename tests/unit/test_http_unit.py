"""
Unit tests for HTTP Request Service using VCR cassettes.

These tests use recorded HTTP interactions and run without real network calls.
Run with: pytest tests/unit/ --record-mode=none
"""

import os
import pytest
import asyncio

# Import HTTP request service
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from services.http_request_service import (
    HttpRequestService,
    HttpRequestConfig,
    HttpResponse,
    HttpMethod,
    create_http_request_service,
    quick_get,
    quick_post,
    quick_request,
)


class TestHttpRequestServiceUnit:
    """Unit tests for HTTP Request Service using recorded cassettes."""

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_basic_get_request(self, mock_credentials):
        """Test basic GET request using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/get",
            method=HttpMethod.GET,
            params={"test": "integration"},
        )

        response = asyncio.run(service.request(config))

        assert isinstance(response, HttpResponse)
        assert response.is_success
        assert response.status_code == 200
        assert "httpbin.org" in response.url

        # Parse JSON response
        json_data = response.json()
        assert "args" in json_data
        assert json_data["args"]["test"] == "integration"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_post_request_with_json(self, mock_credentials):
        """Test POST request with JSON data using recorded response."""
        service = create_http_request_service()

        test_data = {
            "name": "Integration Test",
            "type": "HTTP POST",
            "timestamp": "2025-01-01T00:00:00Z",
        }

        config = HttpRequestConfig(
            url="https://httpbin.org/post",
            method=HttpMethod.POST,
            json_data=test_data,
            headers={"Content-Type": "application/json"},
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

        json_data = response.json()
        assert "json" in json_data
        assert json_data["json"]["name"] == "Integration Test"
        assert json_data["json"]["type"] == "HTTP POST"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_post_request_with_form_data(self, mock_credentials):
        """Test POST request with form data using recorded response."""
        service = create_http_request_service()

        form_data = {
            "field1": "value1",
            "field2": "value2",
            "field3": "special chars: !@#$%^&*()",
        }

        config = HttpRequestConfig(
            url="https://httpbin.org/post", method=HttpMethod.POST, form_data=form_data
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

        json_data = response.json()
        assert "form" in json_data
        assert json_data["form"]["field1"] == "value1"
        assert json_data["form"]["field2"] == "value2"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_put_request(self, mock_credentials):
        """Test PUT request using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/put",
            method=HttpMethod.PUT,
            json_data={"action": "update", "id": 123},
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

        json_data = response.json()
        assert json_data["json"]["action"] == "update"
        assert json_data["json"]["id"] == 123

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_delete_request(self, mock_credentials):
        """Test DELETE request using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/delete", method=HttpMethod.DELETE
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_request_with_custom_headers(self, mock_credentials):
        """Test request with custom headers using recorded response."""
        service = create_http_request_service()

        custom_headers = {
            "User-Agent": "AI-Lego-Bricks/1.0",
            "X-Custom-Header": "Integration-Test",
            "Accept": "application/json",
        }

        config = HttpRequestConfig(
            url="https://httpbin.org/headers",
            method=HttpMethod.GET,
            headers=custom_headers,
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        json_data = response.json()

        # Check that our custom headers were sent
        assert "User-Agent" in json_data["headers"]
        assert "X-Custom-Header" in json_data["headers"]
        assert json_data["headers"]["X-Custom-Header"] == "Integration-Test"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_basic_authentication(self, mock_credentials):
        """Test basic authentication using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/basic-auth/testuser/testpass",
            method=HttpMethod.GET,
            auth_type="basic",
            auth_credentials={"username": "testuser", "password": "testpass"},
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

        json_data = response.json()
        assert json_data["authenticated"] is True
        assert json_data["user"] == "testuser"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_bearer_token_authentication(self, mock_credentials):
        """Test bearer token authentication using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/bearer",
            method=HttpMethod.GET,
            auth_type="bearer",
            auth_credentials={"token": "test-bearer-token-12345"},
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

        json_data = response.json()
        assert json_data["authenticated"] is True
        assert json_data["token"] == "test-bearer-token-12345"

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_error_handling_404(self, mock_credentials):
        """Test handling of 404 Not Found errors using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/404", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))

        assert not response.is_success
        assert response.is_client_error
        assert response.status_code == 404

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_error_handling_500(self, mock_credentials):
        """Test handling of 500 Internal Server Error using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/500", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))

        assert not response.is_success
        assert response.is_server_error
        assert response.status_code == 500

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_redirect_handling(self, mock_credentials):
        """Test automatic redirect following using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/redirect/2",  # 2 redirects
            method=HttpMethod.GET,
            follow_redirects=True,
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200
        # Final URL should be the redirect target
        assert "get" in response.url

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    @pytest.mark.asyncio
    async def test_convenience_methods(self, mock_credentials):
        """Test convenience methods using recorded responses."""
        service = create_http_request_service()

        # Test GET convenience method
        get_response = await service.get(
            "https://httpbin.org/get", params={"method": "get"}
        )
        assert get_response.is_success

        # Test POST convenience method
        post_response = await service.post(
            "https://httpbin.org/post", json_data={"method": "post"}
        )
        assert post_response.is_success

        # Test PUT convenience method
        put_response = await service.put(
            "https://httpbin.org/put", json_data={"method": "put"}
        )
        assert put_response.is_success

        # Test DELETE convenience method
        delete_response = await service.delete("https://httpbin.org/delete")
        assert delete_response.is_success

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_quick_functions(self, mock_credentials):
        """Test quick convenience functions using recorded responses."""
        # Test quick_get
        get_response = asyncio.run(quick_get("https://httpbin.org/get"))
        assert get_response.is_success

        # Test quick_post
        post_response = asyncio.run(
            quick_post("https://httpbin.org/post", json_data={"quick": "post"})
        )
        assert post_response.is_success

        # Test quick_request
        config = HttpRequestConfig(url="https://httpbin.org/get", method=HttpMethod.GET)
        request_response = asyncio.run(quick_request(config))
        assert request_response.is_success

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_complex_json_handling(self, mock_credentials):
        """Test handling of complex JSON data using recorded response."""
        service = create_http_request_service()

        complex_data = {
            "user": {
                "name": "Test User",
                "email": "test@example.com",
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                    "tags": ["developer", "tester", "integration"],
                },
            },
            "metadata": {
                "created_at": "2025-01-01T00:00:00Z",
                "version": 1.0,
                "features": ["auth", "api", "testing"],
            },
            "numbers": [1, 2, 3.14, 42],
            "boolean_flags": {"active": True, "verified": False, "premium": None},
        }

        config = HttpRequestConfig(
            url="https://httpbin.org/post",
            method=HttpMethod.POST,
            json_data=complex_data,
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        json_data = response.json()

        # Verify complex data was preserved
        assert json_data["json"]["user"]["name"] == "Test User"
        assert json_data["json"]["user"]["preferences"]["theme"] == "dark"
        assert "developer" in json_data["json"]["user"]["preferences"]["tags"]
        assert json_data["json"]["metadata"]["version"] == 1.0
        assert json_data["json"]["boolean_flags"]["active"] is True
        assert json_data["json"]["boolean_flags"]["premium"] is None

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_concurrent_requests(self, mock_credentials):
        """Test multiple concurrent HTTP requests using recorded responses."""
        service = create_http_request_service()

        async def make_request(request_id):
            config = HttpRequestConfig(
                url="https://httpbin.org/get",
                method=HttpMethod.GET,
                params={"request_id": str(request_id)},
            )
            return await service.request(config)

        async def run_concurrent_requests():
            # Create multiple concurrent requests
            tasks = [make_request(i) for i in range(3)]
            return await asyncio.gather(*tasks)

        responses = asyncio.run(run_concurrent_requests())

        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.is_success
            json_data = response.json()
            assert json_data["args"]["request_id"] == str(i)

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_large_response_handling(self, mock_credentials):
        """Test handling of large responses using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/base64/SFRUUCBSZXF1ZXN0IFNlcnZpY2UgVGVzdCAtIExhcmdlIFJlc3BvbnNlIERhdGEgd2l0aCBtdWx0aXBsZSBsaW5lcyBhbmQgc3BlY2lhbCBjaGFyYWN0ZXJzIDEyMzQ1Njc4OTAhQCMkJV4mKigp",
            method=HttpMethod.GET,
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert len(response.content) > 0
        assert len(response.text) > 0

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    @pytest.mark.asyncio
    async def test_response_content_types(self, mock_credentials):
        """Test different response content types using recorded responses."""
        service = create_http_request_service()

        # Test JSON response
        json_config = HttpRequestConfig(
            url="https://httpbin.org/json", method=HttpMethod.GET
        )
        json_response = await service.request(json_config)
        assert json_response.is_success
        json_data = json_response.json()
        assert isinstance(json_data, dict)

        # Test HTML response
        html_config = HttpRequestConfig(
            url="https://httpbin.org/html", method=HttpMethod.GET
        )
        html_response = await service.request(html_config)
        assert html_response.is_success
        assert "<html>" in html_response.text.lower()

        # Test XML response
        xml_config = HttpRequestConfig(
            url="https://httpbin.org/xml", method=HttpMethod.GET
        )
        xml_response = await service.request(xml_config)
        assert xml_response.is_success
        assert "<?xml" in xml_response.text

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_authentication_failure(self, mock_credentials):
        """Test authentication failure scenarios using recorded response."""
        service = create_http_request_service()

        # Test wrong basic auth credentials
        config = HttpRequestConfig(
            url="https://httpbin.org/basic-auth/testuser/testpass",
            method=HttpMethod.GET,
            auth_type="basic",
            auth_credentials={"username": "wronguser", "password": "wrongpass"},
        )

        response = asyncio.run(service.request(config))
        assert response.status_code == 401
        assert not response.is_success

    @pytest.mark.vcr
    @pytest.mark.unit
    @pytest.mark.cassette
    def test_rate_limiting_handling(self, mock_credentials):
        """Test handling of rate limiting using recorded response."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/429", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))
        assert response.status_code == 429
        assert response.is_client_error
        assert not response.is_success


class TestHttpRequestServiceMocking:
    """Tests that demonstrate pure mocking without VCR."""

    @pytest.mark.unit
    def test_http_config_validation(self, mock_credentials):
        """Test configuration validation without network calls."""
        # Test valid configuration
        config = HttpRequestConfig(
            url="https://example.com/api",
            method=HttpMethod.POST,
            headers={"Authorization": "Bearer token"},
            json_data={"key": "value"},
            timeout=60.0,
        )

        assert config.url == "https://example.com/api"
        assert config.method == HttpMethod.POST
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.json_data == {"key": "value"}
        assert config.timeout == 60.0

    @pytest.mark.unit
    def test_invalid_configuration_combinations(self, mock_credentials):
        """Test invalid configuration combinations."""
        # Test mutually exclusive data fields
        with pytest.raises(
            ValueError,
            match="Only one of json_data, form_data, or data can be specified",
        ):
            HttpRequestConfig(
                url="https://httpbin.org/post",
                json_data={"key": "value"},
                form_data={"key": "value"},
            )

        # Test empty URL
        with pytest.raises(ValueError, match="URL is required"):
            HttpRequestConfig(url="")

        # Test invalid URL format
        with pytest.raises(ValueError, match="Invalid URL format"):
            HttpRequestConfig(url="not-a-valid-url")

    @pytest.mark.unit
    def test_http_response_properties(self, mock_credentials):
        """Test HTTP response property calculations."""
        # Test success response
        success_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"key": "value"}',
            text='{"key": "value"}',
            url="https://example.com",
            method="GET",
            elapsed_time=0.5,
        )

        assert success_response.is_success is True
        assert success_response.is_client_error is False
        assert success_response.is_server_error is False

        # Test client error response
        client_error_response = HttpResponse(
            status_code=404,
            headers={},
            content=b"Not Found",
            text="Not Found",
            url="https://example.com",
            method="GET",
            elapsed_time=0.1,
        )

        assert client_error_response.is_success is False
        assert client_error_response.is_client_error is True
        assert client_error_response.is_server_error is False

        # Test server error response
        server_error_response = HttpResponse(
            status_code=500,
            headers={},
            content=b"Internal Server Error",
            text="Internal Server Error",
            url="https://example.com",
            method="GET",
            elapsed_time=0.1,
        )

        assert server_error_response.is_success is False
        assert server_error_response.is_client_error is False
        assert server_error_response.is_server_error is True

    @pytest.mark.unit
    def test_json_parsing(self, mock_credentials):
        """Test JSON parsing functionality."""
        # Test valid JSON
        response_with_json = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"key": "value", "number": 42}',
            text='{"key": "value", "number": 42}',
            url="https://example.com",
            method="GET",
            elapsed_time=0.1,
        )

        json_data = response_with_json.json()
        assert json_data == {"key": "value", "number": 42}

        # Test invalid JSON
        response_with_invalid_json = HttpResponse(
            status_code=200,
            headers={},
            content=b"Invalid JSON",
            text="Invalid JSON",
            url="https://example.com",
            method="GET",
            elapsed_time=0.1,
        )

        with pytest.raises(ValueError, match="Response is not valid JSON"):
            response_with_invalid_json.json()

    @pytest.mark.unit
    def test_service_initialization(self, mock_credentials):
        """Test service initialization without network calls."""
        # Test basic service creation
        service = HttpRequestService()
        assert service.default_timeout == 30.0
        assert service.max_retries == 3
        assert service.backoff_factor == 1.0

        # Test service with custom parameters
        custom_service = HttpRequestService(
            default_timeout=60.0, max_retries=5, backoff_factor=2.0
        )
        assert custom_service.default_timeout == 60.0
        assert custom_service.max_retries == 5
        assert custom_service.backoff_factor == 2.0

    @pytest.mark.unit
    def test_auth_header_preparation(self, mock_credentials):
        """Test authentication header preparation without network calls."""
        service = HttpRequestService()

        # Test bearer token
        bearer_config = HttpRequestConfig(
            url="https://example.com",
            auth_type="bearer",
            auth_credentials={"token": "secret-token"},
        )
        bearer_headers = service._prepare_auth_headers(bearer_config)
        assert bearer_headers["Authorization"] == "Bearer secret-token"

        # Test basic auth
        basic_config = HttpRequestConfig(
            url="https://example.com",
            auth_type="basic",
            auth_credentials={"username": "user", "password": "pass"},
        )
        basic_headers = service._prepare_auth_headers(basic_config)
        assert "Authorization" in basic_headers
        assert basic_headers["Authorization"].startswith("Basic ")

        # Test API key
        api_key_config = HttpRequestConfig(
            url="https://example.com",
            auth_type="api_key",
            auth_credentials={"api_key": "secret-key"},
        )
        api_headers = service._prepare_auth_headers(api_key_config)
        assert api_headers["X-API-Key"] == "secret-key"

        # Test API key with custom header
        custom_header_config = HttpRequestConfig(
            url="https://example.com",
            auth_type="api_key",
            auth_credentials={"api_key": "secret-key", "header_name": "X-Custom-Key"},
        )
        custom_headers = service._prepare_auth_headers(custom_header_config)
        assert custom_headers["X-Custom-Key"] == "secret-key"
