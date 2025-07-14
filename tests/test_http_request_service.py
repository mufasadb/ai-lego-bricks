"""
Tests for HTTP Request Service
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the HTTP request service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.http_request_service import (
    HttpRequestService,
    HttpRequestConfig,
    HttpResponse,
    HttpMethod,
    create_http_request_service,
    quick_get,
    quick_post,
    quick_request
)
from credentials.credential_manager import CredentialManager


class TestHttpRequestConfig:
    """Test HttpRequestConfig class"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = HttpRequestConfig(
            url="https://example.com/api",
            method=HttpMethod.POST,
            headers={"Authorization": "Bearer token"},
            json_data={"key": "value"}
        )
        
        assert config.url == "https://example.com/api"
        assert config.method == HttpMethod.POST
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.json_data == {"key": "value"}
    
    def test_invalid_url(self):
        """Test invalid URL validation"""
        with pytest.raises(ValueError, match="Invalid URL format"):
            HttpRequestConfig(url="invalid-url")
    
    def test_empty_url(self):
        """Test empty URL validation"""
        with pytest.raises(ValueError, match="URL is required"):
            HttpRequestConfig(url="")
    
    def test_mutually_exclusive_data(self):
        """Test mutually exclusive data fields"""
        with pytest.raises(ValueError, match="Only one of json_data, form_data, or data can be specified"):
            HttpRequestConfig(
                url="https://example.com",
                json_data={"key": "value"},
                form_data={"key": "value"}
            )
    
    def test_default_values(self):
        """Test default configuration values"""
        config = HttpRequestConfig(url="https://example.com")
        
        assert config.method == HttpMethod.GET
        assert config.headers == {}
        assert config.params == {}
        assert config.timeout == 30.0
        assert config.follow_redirects is True
        assert config.verify_ssl is True


class TestHttpResponse:
    """Test HttpResponse class"""
    
    def test_response_properties(self):
        """Test response properties"""
        response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"key": "value"}',
            text='{"key": "value"}',
            url="https://example.com",
            method="GET",
            elapsed_time=0.5
        )
        
        assert response.is_success is True
        assert response.is_client_error is False
        assert response.is_server_error is False
    
    def test_client_error_response(self):
        """Test client error response"""
        response = HttpResponse(
            status_code=404,
            headers={},
            content=b'Not Found',
            text='Not Found',
            url="https://example.com",
            method="GET",
            elapsed_time=0.1
        )
        
        assert response.is_success is False
        assert response.is_client_error is True
        assert response.is_server_error is False
    
    def test_server_error_response(self):
        """Test server error response"""
        response = HttpResponse(
            status_code=500,
            headers={},
            content=b'Internal Server Error',
            text='Internal Server Error',
            url="https://example.com",
            method="GET",
            elapsed_time=0.1
        )
        
        assert response.is_success is False
        assert response.is_client_error is False
        assert response.is_server_error is True
    
    def test_json_parsing(self):
        """Test JSON parsing"""
        response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=b'{"key": "value", "number": 42}',
            text='{"key": "value", "number": 42}',
            url="https://example.com",
            method="GET",
            elapsed_time=0.1
        )
        
        json_data = response.json()
        assert json_data == {"key": "value", "number": 42}
    
    def test_invalid_json_parsing(self):
        """Test invalid JSON parsing"""
        response = HttpResponse(
            status_code=200,
            headers={},
            content=b'Invalid JSON',
            text='Invalid JSON',
            url="https://example.com",
            method="GET",
            elapsed_time=0.1
        )
        
        with pytest.raises(ValueError, match="Response is not valid JSON"):
            response.json()


class TestHttpRequestService:
    """Test HttpRequestService class"""
    
    @pytest.fixture
    def mock_credential_manager(self):
        """Mock credential manager"""
        manager = Mock(spec=CredentialManager)
        manager.get_credential.return_value = None
        return manager
    
    @pytest.fixture
    def service(self, mock_credential_manager):
        """Create service instance for testing"""
        return HttpRequestService(credential_manager=mock_credential_manager)
    
    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service.default_timeout == 30.0
        assert service.max_retries == 3
        assert service.backoff_factor == 1.0
        assert service.credential_manager is not None
    
    def test_prepare_auth_headers_bearer(self, service):
        """Test bearer token authentication"""
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="bearer",
            auth_credentials={"token": "secret-token"}
        )
        
        headers = service._prepare_auth_headers(config)
        assert headers["Authorization"] == "Bearer secret-token"
    
    def test_prepare_auth_headers_basic(self, service):
        """Test basic authentication"""
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="basic",
            auth_credentials={"username": "user", "password": "pass"}
        )
        
        headers = service._prepare_auth_headers(config)
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")
    
    def test_prepare_auth_headers_api_key(self, service):
        """Test API key authentication"""
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="api_key",
            auth_credentials={"api_key": "secret-key"}
        )
        
        headers = service._prepare_auth_headers(config)
        assert headers["X-API-Key"] == "secret-key"
    
    def test_prepare_auth_headers_custom_header(self, service):
        """Test API key with custom header"""
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="api_key",
            auth_credentials={"api_key": "secret-key", "header_name": "X-Custom-Key"}
        )
        
        headers = service._prepare_auth_headers(config)
        assert headers["X-Custom-Key"] == "secret-key"
    
    def test_resolve_credentials_bearer(self, service):
        """Test credential resolution for bearer tokens"""
        service.credential_manager.get_credential.side_effect = lambda key: {
            "TOKEN": "resolved-token"
        }.get(key)
        
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="bearer"
        )
        
        resolved_config = service._resolve_credentials(config)
        assert resolved_config.auth_credentials["token"] == "resolved-token"
    
    def test_resolve_credentials_basic(self, service):
        """Test credential resolution for basic auth"""
        service.credential_manager.get_credential.side_effect = lambda key: {
            "USERNAME": "resolved-user",
            "PASSWORD": "resolved-pass"
        }.get(key)
        
        config = HttpRequestConfig(
            url="https://example.com",
            auth_type="basic"
        )
        
        resolved_config = service._resolve_credentials(config)
        assert resolved_config.auth_credentials["username"] == "resolved-user"
        assert resolved_config.auth_credentials["password"] == "resolved-pass"
    
    @pytest.mark.asyncio
    async def test_request_success(self, service):
        """Test successful HTTP request"""
        # Mock httpx client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.content = b'{"success": true}'
        mock_response.text = '{"success": true}'
        mock_response.url = "https://example.com"
        
        with patch.object(service.client, 'request', return_value=mock_response) as mock_request:
            config = HttpRequestConfig(url="https://example.com")
            response = await service.request(config)
            
            assert response.status_code == 200
            assert response.is_success is True
            assert response.text == '{"success": true}'
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_with_retry(self, service):
        """Test HTTP request with retry logic"""
        import httpx
        
        # Mock to fail first time, succeed second time
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'Success'
        mock_response.text = 'Success'
        mock_response.url = "https://example.com"
        
        with patch.object(service.client, 'request', side_effect=[
            httpx.RequestError("Connection failed"),
            mock_response
        ]) as mock_request:
            config = HttpRequestConfig(url="https://example.com")
            response = await service.request(config)
            
            assert response.status_code == 200
            assert mock_request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self, service):
        """Test convenience methods"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'OK'
        mock_response.text = 'OK'
        mock_response.url = "https://example.com"
        
        with patch.object(service.client, 'request', return_value=mock_response):
            # Test GET
            response = await service.get("https://example.com")
            assert response.status_code == 200
            
            # Test POST
            response = await service.post("https://example.com", json_data={"key": "value"})
            assert response.status_code == 200
            
            # Test PUT
            response = await service.put("https://example.com", json_data={"key": "value"})
            assert response.status_code == 200
            
            # Test DELETE
            response = await service.delete("https://example.com")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_credential_manager):
        """Test async context manager"""
        async with HttpRequestService(credential_manager=mock_credential_manager) as service:
            assert service is not None
            assert hasattr(service, 'client')


class TestFactoryFunctions:
    """Test factory and convenience functions"""
    
    def test_create_http_request_service(self):
        """Test factory function"""
        service = create_http_request_service()
        assert isinstance(service, HttpRequestService)
        assert service.credential_manager is None
    
    def test_create_http_request_service_with_credentials(self):
        """Test factory function with credentials"""
        cred_manager = Mock(spec=CredentialManager)
        service = create_http_request_service(credential_manager=cred_manager)
        assert service.credential_manager == cred_manager
    
    @pytest.mark.asyncio
    async def test_quick_get(self):
        """Test quick GET function"""
        with patch('services.http_request_service.HttpRequestService') as mock_service_class:
            mock_service = Mock()
            mock_service.__aenter__ = AsyncMock(return_value=mock_service)
            mock_service.__aexit__ = AsyncMock()
            mock_service.get = AsyncMock(return_value=Mock(status_code=200))
            mock_service_class.return_value = mock_service
            
            response = await quick_get("https://example.com")
            assert response.status_code == 200
            mock_service.get.assert_called_once_with("https://example.com")
    
    @pytest.mark.asyncio
    async def test_quick_post(self):
        """Test quick POST function"""
        with patch('services.http_request_service.HttpRequestService') as mock_service_class:
            mock_service = Mock()
            mock_service.__aenter__ = AsyncMock(return_value=mock_service)
            mock_service.__aexit__ = AsyncMock()
            mock_service.post = AsyncMock(return_value=Mock(status_code=201))
            mock_service_class.return_value = mock_service
            
            response = await quick_post("https://example.com", json_data={"key": "value"})
            assert response.status_code == 201
            mock_service.post.assert_called_once_with("https://example.com", json_data={"key": "value"})
    
    @pytest.mark.asyncio
    async def test_quick_request(self):
        """Test quick request function"""
        with patch('services.http_request_service.HttpRequestService') as mock_service_class:
            mock_service = Mock()
            mock_service.__aenter__ = AsyncMock(return_value=mock_service)
            mock_service.__aexit__ = AsyncMock()
            mock_service.request = AsyncMock(return_value=Mock(status_code=200))
            mock_service_class.return_value = mock_service
            
            config = HttpRequestConfig(url="https://example.com")
            response = await quick_request(config)
            assert response.status_code == 200
            mock_service.request.assert_called_once_with(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])