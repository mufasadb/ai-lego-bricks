from __future__ import annotations
import os
import time
import random
import json
from typing import List, Optional, Union, Type, TypeVar, Dict, Any, Generator
import httpx
from pydantic import BaseModel
from .llm_types import (
    TextLLMClient, ChatMessage, LLMConfig, LLMProvider, 
    StructuredLLMWrapper, StructuredResponseConfig,
    create_function_calling_schema
)
from credentials import CredentialManager, default_credential_manager

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

T = TypeVar('T', bound=BaseModel)


class OllamaTextClient(TextLLMClient):
    """Ollama text client implementation"""
    
    def __init__(self, config: LLMConfig, credential_manager: Optional[CredentialManager] = None):
        self.config = config
        self.credential_manager = credential_manager or default_credential_manager
        self.base_url = self.credential_manager.get_credential("OLLAMA_URL", "http://localhost:11434")
        self.model = config.model or self.credential_manager.get_credential("OLLAMA_DEFAULT_MODEL", "llama2")
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """Send a chat message and get response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        return self.chat_with_messages(messages)
    
    def chat_with_messages(self, messages: List[ChatMessage]) -> str:
        """Send multiple messages and get response"""
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different model
        
        Args:
            new_model: The new model name to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if not self._validate_model(new_model):
            raise ValueError(f"Model '{new_model}' is not available in Ollama")
        
        old_model = self.model
        self.model = new_model
        print(f"Switched from '{old_model}' to '{new_model}'")
        return True
    
    def get_current_model(self) -> str:
        """
        Get the current model name
        
        Returns:
            Current model name
        """
        return self.model
    
    def _validate_model(self, model_name: str) -> bool:
        """
        Check if a model exists in Ollama
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return model_name in available_models
        except Exception:
            return False
    
    def chat_structured(self, message: str, schema: Union[Type[BaseModel], Dict[str, Any]], 
                       chat_history: Optional[List[ChatMessage]] = None) -> Union[BaseModel, Dict[str, Any]]:
        """Chat with structured output using JSON schema prompting"""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic model
            wrapper = StructuredLLMWrapper(self, schema)
            return wrapper.chat(message, chat_history)
        else:
            # Raw JSON schema - use basic prompting
            schema_str = json.dumps(schema, indent=2)
            enhanced_message = f"""
{message}

IMPORTANT: Respond with valid JSON that exactly matches this schema:
{schema_str}

JSON Response:
"""
            response = self.chat(enhanced_message, chat_history)
            # Try to parse and return
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback - try to extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    return json.loads(json_str)
                raise ValueError(f"Could not parse JSON from response: {response}")
    
    def with_structured_output(self, schema: Type[T]) -> StructuredLLMWrapper[T]:
        """Return a wrapper that ensures structured output of specified type"""
        return StructuredLLMWrapper(self, schema)
    
    def chat_stream(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Generator[str, None, str]:
        """Send a chat message and get streaming response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        return self.chat_with_messages_stream(messages)
    
    def chat_with_messages_stream(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
        """Send multiple messages and get streaming response"""
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        full_response = ""
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    content = chunk["message"]["content"]
                                    if content:
                                        full_response += content
                                        yield content
                                
                                # Check if this is the final chunk
                                if chunk.get("done", False):
                                    break
                                    
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
                                
        except Exception as e:
            # If streaming fails, fall back to regular response
            full_response = self.chat_with_messages(messages)
            yield full_response
        
        return full_response


class GeminiTextClient(TextLLMClient):
    """Gemini text client implementation"""
    
    def __init__(self, config: LLMConfig, credential_manager: Optional[CredentialManager] = None):
        self.config = config
        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential("GOOGLE_AI_STUDIO_KEY", "Gemini")
        
        self.api_base_url = self.credential_manager.get_credential("GEMINI_API_BASE_URL", 
                                                                  "https://generativelanguage.googleapis.com/v1beta")
        self.model = config.model or self.credential_manager.get_credential("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """Send a chat message and get response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        return self.chat_with_messages(messages)
    
    def chat_with_messages(self, messages: List[ChatMessage]) -> str:
        """Send multiple messages and get response"""
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(
                        f"{self.api_base_url}/models/{self.model}:generateContent",
                        json=payload,
                        headers={"x-goog-api-key": self.api_key}
                    )
                    
                    # Handle specific error codes
                    if response.status_code == 503:
                        print(f"503 Service Unavailable error on attempt {attempt + 1}/{max_retries}")
                        print(f"Response: {response.text}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"Retrying in {delay:.2f} seconds...")
                            time.sleep(delay)
                            continue
                    
                    response.raise_for_status()
                    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"503 error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                elif e.response.status_code == 429:
                    # Handle quota/billing limits with loud warnings
                    error_text = e.response.text
                    print("=" * 80)
                    print("🚨 CRITICAL: GEMINI API QUOTA/BILLING LIMIT EXCEEDED! 🚨")
                    print("=" * 80)
                    print(f"❌ HTTP 429 Error: {error_text}")
                    print()
                    print("💰 BILLING ALERT:")
                    print("   • You have exceeded your Gemini API quota limits")
                    print("   • This may result in unexpected charges if on paid tier")
                    print("   • Consider implementing rate limiting or usage monitoring")
                    print()
                    print("🤖 AGENT WARNING:")
                    print("   • IF THIS IS BEING CONSUMED BY AN AGENT, PLEASE INFORM THE USER")
                    print("   • Agents can rapidly consume API quotas without user awareness")
                    print("   • Consider pausing automated workflows until limits reset")
                    print()
                    print("📊 Next Steps:")
                    print("   1. Check your Google AI Studio billing dashboard")
                    print("   2. Implement request throttling in agent workflows")
                    print("   3. Consider upgrading quota limits if needed")
                    print("   4. Monitor usage patterns to prevent future overages")
                    print("=" * 80)
                    raise
                else:
                    print(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded for Gemini API request")
    
    def chat_structured(self, message: str, schema: Union[Type[BaseModel], Dict[str, Any]], 
                       chat_history: Optional[List[ChatMessage]] = None) -> Union[BaseModel, Dict[str, Any]]:
        """Chat with structured output using Gemini function calling"""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Use function calling for Pydantic models
            return self._chat_with_function_calling(message, schema, chat_history)
        else:
            # Fallback to prompting for raw JSON schema
            wrapper = StructuredLLMWrapper(self, schema if isinstance(schema, type) else type('DynamicSchema', (BaseModel,), {}))
            return wrapper.chat(message, chat_history)
    
    def _chat_with_function_calling(self, message: str, schema: Type[BaseModel], 
                                   chat_history: Optional[List[ChatMessage]] = None) -> BaseModel:
        """Use Gemini's function calling for structured responses"""
        # Create function declaration from Pydantic schema
        function_schema = create_function_calling_schema(schema, "structured_response", 
                                                        "Provide a structured response")
        
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        
        # Convert to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        payload = {
            "contents": gemini_messages,
            "tools": [{
                "function_declarations": [function_schema]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(
                        f"{self.api_base_url}/models/{self.model}:generateContent",
                        json=payload,
                        headers={"x-goog-api-key": self.api_key}
                    )
                    
                    if response.status_code == 503 and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"503 error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    
                    response.raise_for_status()
                    response_data = response.json()
                    
                    # Extract function call from response
                    candidates = response_data.get("candidates", [])
                    if not candidates:
                        raise ValueError("No candidates in Gemini response")
                    
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    # Look for function call
                    for part in parts:
                        if "functionCall" in part:
                            function_call = part["functionCall"]
                            if function_call.get("name") == "structured_response":
                                args = function_call.get("args", {})
                                # Validate and return as Pydantic model
                                return schema.model_validate(args)
                    
                    # Fallback: if no function call, try to parse text response
                    for part in parts:
                        if "text" in part:
                            text_response = part["text"]
                            # Try to extract JSON and validate
                            wrapper = StructuredLLMWrapper(self, schema)
                            return wrapper._enforce_structured_response(message, chat_history)
                    
                    raise ValueError("No function call or text response found in Gemini response")
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    # Handle other HTTP errors as before
                    if e.response.status_code == 429:
                        # Re-use existing 429 error handling
                        error_text = e.response.text
                        print("=" * 80)
                        print("🚨 CRITICAL: GEMINI API QUOTA/BILLING LIMIT EXCEEDED! 🚨")
                        print("=" * 80)
                        print(f"❌ HTTP 429 Error: {error_text}")
                        # ... rest of existing error handling
                        raise
                    else:
                        print(f"HTTP error {e.response.status_code}: {e.response.text}")
                        raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded for Gemini structured API request")
    
    def with_structured_output(self, schema: Type[T]) -> StructuredLLMWrapper[T]:
        """Return a wrapper that ensures structured output of specified type"""
        return StructuredLLMWrapper(self, schema)
    
    def chat_stream(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Generator[str, None, str]:
        """Send a chat message and get streaming response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        return self.chat_with_messages_stream(messages)
    
    def chat_with_messages_stream(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
        """Send multiple messages and get streaming response"""
        # Gemini doesn't have a simple streaming API like Ollama
        # So we'll simulate streaming by chunking the response
        try:
            # Get the full response first
            full_response = self.chat_with_messages(messages)
            
            # Simulate streaming by yielding words in chunks
            words = full_response.split()
            current_chunk = ""
            for i, word in enumerate(words):
                current_chunk += word + " "
                if i % 4 == 0 or i == len(words) - 1:  # Yield every 4 words
                    yield current_chunk.strip()
                    current_chunk = ""
                    time.sleep(0.03)  # Small delay to simulate streaming
            
            return full_response
            
        except Exception as e:
            # Fall back to regular response
            full_response = self.chat_with_messages(messages)
            yield full_response
            return full_response


class AnthropicTextClient(TextLLMClient):
    """Anthropic Claude text client implementation"""
    
    def __init__(self, config: LLMConfig, credential_manager: Optional[CredentialManager] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic package not available. Install with: pip install anthropic")
        
        self.config = config
        self.credential_manager = credential_manager or default_credential_manager
        self.api_key = self.credential_manager.require_credential("ANTHROPIC_API_KEY", "Anthropic")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = config.model or self.credential_manager.get_credential("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20241022")
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """Send a chat message and get response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role='user', content=message))
        return self.chat_with_messages(messages)
    
    def chat_with_messages(self, messages: List[ChatMessage]) -> str:
        """Send multiple messages and get response"""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": anthropic_messages,
                **self.config.extra_params
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
            
        except anthropic.APIError as e:
            if e.status_code == 429:
                print("=" * 80)
                print("🚨 CRITICAL: ANTHROPIC API RATE LIMIT EXCEEDED! 🚨")
                print("=" * 80)
                print(f"❌ HTTP 429 Error: {e}")
                print()
                print("💰 BILLING ALERT:")
                print("   • You have exceeded your Anthropic API rate limits")
                print("   • This may result in unexpected charges or service interruption")
                print("   • Consider implementing rate limiting or usage monitoring")
                print()
                print("🤖 AGENT WARNING:")
                print("   • IF THIS IS BEING CONSUMED BY AN AGENT, PLEASE INFORM THE USER")
                print("   • Agents can rapidly consume API quotas without user awareness")
                print("   • Consider pausing automated workflows until limits reset")
                print()
                print("📊 Next Steps:")
                print("   1. Check your Anthropic Console billing dashboard")
                print("   2. Implement request throttling in agent workflows")
                print("   3. Consider upgrading rate limits if needed")
                print("   4. Monitor usage patterns to prevent future overages")
                print("=" * 80)
                raise
            else:
                print(f"Anthropic API error: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error with Anthropic API: {e}")
            raise
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different Anthropic model
        
        Args:
            new_model: The new model name to switch to
            
        Returns:
            True if successful, False otherwise
        """
        old_model = self.model
        self.model = new_model
        print(f"Switched from '{old_model}' to '{new_model}'")
        return True
    
    def get_current_model(self) -> str:
        """
        Get the current model name
        
        Returns:
            Current model name
        """
        return self.model
    
    def chat_structured(self, message: str, schema: Union[Type[BaseModel], Dict[str, Any]], 
                       chat_history: Optional[List[ChatMessage]] = None) -> Union[BaseModel, Dict[str, Any]]:
        """Chat with structured output using JSON schema prompting"""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic model - use wrapper
            wrapper = StructuredLLMWrapper(self, schema)
            return wrapper.chat(message, chat_history)
        else:
            # Raw JSON schema - use basic prompting
            schema_str = json.dumps(schema, indent=2)
            enhanced_message = f"""
{message}

IMPORTANT: Respond with valid JSON that exactly matches this schema:
{schema_str}

JSON Response:
"""
            response = self.chat(enhanced_message, chat_history)
            # Try to parse and return
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback - try to extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    return json.loads(json_str)
                raise ValueError(f"Could not parse JSON from response: {response}")
    
    def with_structured_output(self, schema: Type[T]) -> StructuredLLMWrapper[T]:
        """Return a wrapper that ensures structured output of specified type"""
        return StructuredLLMWrapper(self, schema)
    def chat_stream(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Generator[str, None, str]:
        """Send a chat message and get streaming response"""
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(ChatMessage(role="user", content=message))
        return self.chat_with_messages_stream(messages)
    
    def chat_with_messages_stream(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
        """Send multiple messages and get streaming response"""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": anthropic_messages,
                "stream": True,  # Enable streaming
                **self.config.extra_params
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            full_response = ""
            with self.client.messages.stream(**kwargs) as stream:
                for chunk in stream:
                    if chunk.type == "content_block_delta":
                        if hasattr(chunk.delta, "text"):
                            text_chunk = chunk.delta.text
                            full_response += text_chunk
                            yield text_chunk
            
            return full_response
            
        except anthropic.APIError as e:
            if e.status_code == 429:
                print("🚨 CRITICAL: ANTHROPIC API RATE LIMIT EXCEEDED\! 🚨")
                raise
            else:
                # Fall back to regular API
                full_response = self.chat_with_messages(messages)
                yield full_response
                return full_response
        except Exception as e:
            # Fall back to regular API
            full_response = self.chat_with_messages(messages)
            yield full_response
            return full_response
