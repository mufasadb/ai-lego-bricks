"""
Tests for HTTP Step Handler
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

# Import the necessary modules
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agent_orchestration.step_handlers import StepHandlerRegistry
from agent_orchestration.models import StepConfig, StepType, ExecutionContext
from services.http_request_service import HttpResponse


class TestHttpStepHandler:
    """Test HTTP step handler functionality"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator"""
        orchestrator = Mock()
        orchestrator.get_service.return_value = None
        return orchestrator

    @pytest.fixture
    def step_registry(self, mock_orchestrator):
        """Create step registry for testing"""
        return StepHandlerRegistry(mock_orchestrator)

    @pytest.fixture
    def mock_context(self):
        """Mock execution context"""
        context = Mock(spec=ExecutionContext)
        context.credential_manager = None
        return context

    def test_handler_registration(self, step_registry):
        """Test that HTTP handler is registered"""
        handler = step_registry.get_handler(StepType.HTTP_REQUEST)
        assert handler is not None
        assert callable(handler)

    def test_basic_get_request(self, step_registry, mock_context):
        """Test basic GET request"""
        step = StepConfig(
            id="test_http", type=StepType.HTTP_REQUEST, config={"method": "GET"}
        )

        inputs = {"url": "https://httpbin.org/get"}

        # Mock the HTTP response
        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"args": {}, "headers": {}, "origin": "127.0.0.1", "url": "https://httpbin.org/get"}',
            text='{"args": {}, "headers": {}, "origin": "127.0.0.1", "url": "https://httpbin.org/get"}',
            url="https://httpbin.org/get",
            method="GET",
            elapsed_time=0.5,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["method"] == "GET"
            assert result["url"] == "https://httpbin.org/get"
            assert result["elapsed_time"] == 0.5

    def test_post_request_with_json(self, step_registry, mock_context):
        """Test POST request with JSON data"""
        step = StepConfig(
            id="test_http_post", type=StepType.HTTP_REQUEST, config={"method": "POST"}
        )

        inputs = {
            "url": "https://httpbin.org/post",
            "json_data": {"key": "value", "number": 42},
        }

        mock_response = HttpResponse(
            status_code=201,
            headers={"Content-Type": "application/json"},
            content=b'{"json": {"key": "value", "number": 42}}',
            text='{"json": {"key": "value", "number": 42}}',
            url="https://httpbin.org/post",
            method="POST",
            elapsed_time=0.3,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 201
            assert result["method"] == "POST"
            assert "json" in result
            assert result["json"]["json"]["key"] == "value"

    def test_request_with_headers(self, step_registry, mock_context):
        """Test request with custom headers"""
        step = StepConfig(
            id="test_http_headers", type=StepType.HTTP_REQUEST, config={"method": "GET"}
        )

        inputs = {
            "url": "https://httpbin.org/headers",
            "headers": {
                "User-Agent": "TestAgent/1.0",
                "X-Custom-Header": "CustomValue",
            },
        }

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"headers": {"User-Agent": "TestAgent/1.0", "X-Custom-Header": "CustomValue"}}',
            text='{"headers": {"User-Agent": "TestAgent/1.0", "X-Custom-Header": "CustomValue"}}',
            url="https://httpbin.org/headers",
            method="GET",
            elapsed_time=0.2,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert "json" in result
            assert result["json"]["headers"]["User-Agent"] == "TestAgent/1.0"

    def test_request_with_authentication(self, step_registry, mock_context):
        """Test request with authentication"""
        step = StepConfig(
            id="test_http_auth",
            type=StepType.HTTP_REQUEST,
            config={
                "method": "GET",
                "auth_type": "bearer",
                "auth_credentials": {"token": "secret-token"},
            },
        )

        inputs = {"url": "https://httpbin.org/bearer"}

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"authenticated": true, "token": "secret-token"}',
            text='{"authenticated": true, "token": "secret-token"}',
            url="https://httpbin.org/bearer",
            method="GET",
            elapsed_time=0.4,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert "json" in result
            assert result["json"]["authenticated"] is True

    def test_request_with_query_parameters(self, step_registry, mock_context):
        """Test request with query parameters"""
        step = StepConfig(
            id="test_http_params", type=StepType.HTTP_REQUEST, config={"method": "GET"}
        )

        inputs = {
            "url": "https://httpbin.org/get",
            "params": {"param1": "value1", "param2": "value2"},
        }

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"args": {"param1": "value1", "param2": "value2"}}',
            text='{"args": {"param1": "value1", "param2": "value2"}}',
            url="https://httpbin.org/get?param1=value1&param2=value2",
            method="GET",
            elapsed_time=0.3,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert "json" in result
            assert result["json"]["args"]["param1"] == "value1"

    def test_request_with_form_data(self, step_registry, mock_context):
        """Test request with form data"""
        step = StepConfig(
            id="test_http_form", type=StepType.HTTP_REQUEST, config={"method": "POST"}
        )

        inputs = {
            "url": "https://httpbin.org/post",
            "form_data": {"field1": "value1", "field2": "value2"},
        }

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"form": {"field1": "value1", "field2": "value2"}}',
            text='{"form": {"field1": "value1", "field2": "value2"}}',
            url="https://httpbin.org/post",
            method="POST",
            elapsed_time=0.6,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert "json" in result
            assert result["json"]["form"]["field1"] == "value1"

    def test_request_with_timeout_config(self, step_registry, mock_context):
        """Test request with timeout configuration"""
        step = StepConfig(
            id="test_http_timeout",
            type=StepType.HTTP_REQUEST,
            config={
                "method": "GET",
                "timeout": 60.0,
                "max_retries": 5,
                "backoff_factor": 2.0,
            },
        )

        inputs = {"url": "https://httpbin.org/delay/1"}

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"args": {}}',
            text='{"args": {}}',
            url="https://httpbin.org/delay/1",
            method="GET",
            elapsed_time=1.1,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["elapsed_time"] == 1.1

            # Verify service was created with correct configuration
            mock_service_class.assert_called_once()
            call_args = mock_service_class.call_args
            assert call_args[1]["max_retries"] == 5
            assert call_args[1]["backoff_factor"] == 2.0

    def test_request_error_handling(self, step_registry, mock_context):
        """Test error handling in HTTP requests"""
        step = StepConfig(
            id="test_http_error", type=StepType.HTTP_REQUEST, config={"method": "GET"}
        )

        inputs = {"url": "https://httpbin.org/status/404"}

        mock_response = HttpResponse(
            status_code=404,
            headers={"Content-Type": "text/plain"},
            content=b"Not Found",
            text="Not Found",
            url="https://httpbin.org/status/404",
            method="GET",
            elapsed_time=0.1,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is False
            assert result["status_code"] == 404
            assert result["is_client_error"] is True
            assert result["is_server_error"] is False

    def test_request_exception_handling(self, step_registry, mock_context):
        """Test exception handling in HTTP requests"""
        step = StepConfig(
            id="test_http_exception",
            type=StepType.HTTP_REQUEST,
            config={"method": "GET"},
        )

        inputs = {"url": "https://invalid-domain-that-does-not-exist.com"}

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(side_effect=Exception("Connection failed"))
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is False
            assert result["error"] == "Connection failed"
            assert result["error_type"] == "Exception"
            assert result["status_code"] is None

    def test_missing_url_error(self, step_registry, mock_context):
        """Test missing URL error"""
        step = StepConfig(
            id="test_http_no_url", type=StepType.HTTP_REQUEST, config={"method": "GET"}
        )

        inputs = {}  # No URL provided

        handler = step_registry.get_handler(StepType.HTTP_REQUEST)
        result = handler(step, inputs, mock_context)

        assert result["success"] is False
        assert "url is required" in result["error"]
        assert result["error_type"] == "ValueError"

    def test_invalid_method_error(self, step_registry, mock_context):
        """Test invalid HTTP method error"""
        step = StepConfig(
            id="test_http_invalid_method",
            type=StepType.HTTP_REQUEST,
            config={"method": "INVALID"},
        )

        inputs = {"url": "https://httpbin.org/get"}

        handler = step_registry.get_handler(StepType.HTTP_REQUEST)
        result = handler(step, inputs, mock_context)

        assert result["success"] is False
        assert "Invalid HTTP method" in result["error"]
        assert result["error_type"] == "ValueError"

    def test_content_filtering(self, step_registry, mock_context):
        """Test content filtering options"""
        step = StepConfig(
            id="test_http_content_filter",
            type=StepType.HTTP_REQUEST,
            config={"method": "GET", "include_content": False, "include_text": False},
        )

        inputs = {"url": "https://httpbin.org/get"}

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"message": "Hello World"}',
            text='{"message": "Hello World"}',
            url="https://httpbin.org/get",
            method="GET",
            elapsed_time=0.2,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200
            assert "content" not in result
            assert "text" not in result
            assert "metadata" in result

    def test_credential_manager_integration(self, step_registry):
        """Test integration with credential manager"""
        mock_cred_manager = Mock()
        mock_cred_manager.get_credential.return_value = "resolved-token"

        mock_context = Mock(spec=ExecutionContext)
        mock_context.credential_manager = mock_cred_manager

        step = StepConfig(
            id="test_http_credentials",
            type=StepType.HTTP_REQUEST,
            config={"method": "GET", "auth_type": "bearer"},
        )

        inputs = {"url": "https://httpbin.org/bearer"}

        mock_response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"authenticated": true}',
            text='{"authenticated": true}',
            url="https://httpbin.org/bearer",
            method="GET",
            elapsed_time=0.3,
        )

        with patch(
            "services.http_request_service.HttpRequestService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service.request = AsyncMock(return_value=mock_response)
            mock_service_class.return_value = mock_service

            handler = step_registry.get_handler(StepType.HTTP_REQUEST)
            result = handler(step, inputs, mock_context)

            assert result["success"] is True
            assert result["status_code"] == 200

            # Verify credential manager was passed to service
            mock_service_class.assert_called_once()
            call_args = mock_service_class.call_args
            assert call_args[1]["credential_manager"] == mock_cred_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
