"""
Integration tests for HTTP Request Service.

These tests make real HTTP calls to record VCR cassettes.
Run with: pytest tests/integration/ --record-mode=once
"""

import os
import pytest
import asyncio

# Import HTTP request service
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from services.http_request_service import (
    HttpRequestConfig,
    HttpResponse,
    HttpMethod,
    create_http_request_service,
    quick_get,
    quick_post,
    quick_request,
)


class TestHttpRequestServiceIntegration:
    """Integration tests for HTTP Request Service with real HTTP calls."""

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_basic_get_request(self, integration_env_check):
        """Test basic GET request to a public API."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_post_request_with_json(self, integration_env_check):
        """Test POST request with JSON data."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_post_request_with_form_data(self, integration_env_check):
        """Test POST request with form data."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_put_request(self, integration_env_check):
        """Test PUT request."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_delete_request(self, integration_env_check):
        """Test DELETE request."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/delete", method=HttpMethod.DELETE
        )

        response = asyncio.run(service.request(config))

        assert response.is_success
        assert response.status_code == 200

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_request_with_custom_headers(self, integration_env_check):
        """Test request with custom headers."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_basic_authentication(self, integration_env_check):
        """Test basic authentication."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_bearer_token_authentication(self, integration_env_check):
        """Test bearer token authentication."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_error_handling_404(self, integration_env_check):
        """Test handling of 404 Not Found errors."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/404", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))

        assert not response.is_success
        assert response.is_client_error
        assert response.status_code == 404

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_error_handling_500(self, integration_env_check):
        """Test handling of 500 Internal Server Error."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/500", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))

        assert not response.is_success
        assert response.is_server_error
        assert response.status_code == 500

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_timeout_handling(self, integration_env_check):
        """Test timeout handling with slow endpoint."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/delay/2",  # 2 second delay
            method=HttpMethod.GET,
            timeout=1.0,  # 1 second timeout
        )

        # This should timeout and raise an exception
        with pytest.raises(Exception):  # Could be timeout or other network error
            asyncio.run(service.request(config))

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_redirect_handling(self, integration_env_check):
        """Test automatic redirect following."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convenience_methods(self, integration_env_check):
        """Test convenience methods (GET, POST, etc.)."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_quick_functions(self, integration_env_check):
        """Test quick convenience functions."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_complex_json_handling(self, integration_env_check):
        """Test handling of complex JSON data structures."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_concurrent_requests(self, integration_env_check):
        """Test multiple concurrent HTTP requests."""

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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_large_response_handling(self, integration_env_check):
        """Test handling of large responses."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_response_content_types(self, integration_env_check):
        """Test different response content types."""
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


class TestHttpRequestServiceErrorCases:
    """Test error cases and edge conditions."""

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_invalid_url_error(self, integration_env_check):
        """Test handling of invalid URLs."""
        # This should fail during configuration validation
        with pytest.raises(ValueError, match="Invalid URL format"):
            HttpRequestConfig(url="not-a-valid-url")

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_connection_error_handling(self, integration_env_check):
        """Test handling of connection errors."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://nonexistent-domain-12345.com/api",
            method=HttpMethod.GET,
            timeout=5.0,
        )

        # This should raise a connection error
        with pytest.raises(Exception):  # Network-related exception
            asyncio.run(service.request(config))

    @pytest.mark.unit
    def test_invalid_configuration_combinations(self, integration_env_check):
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

    @pytest.mark.vcr
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_authentication_failure(self, integration_env_check):
        """Test authentication failure scenarios."""
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
    @pytest.mark.real_api
    @pytest.mark.integration
    def test_rate_limiting_handling(self, integration_env_check):
        """Test handling of rate limiting (429 status)."""
        service = create_http_request_service()

        config = HttpRequestConfig(
            url="https://httpbin.org/status/429", method=HttpMethod.GET
        )

        response = asyncio.run(service.request(config))
        assert response.status_code == 429
        assert response.is_client_error
        assert not response.is_success
