import os
from typing import List, Optional, Dict, Any, Generator
from dotenv import load_dotenv
from pydantic import BaseModel
import httpx
import json

# Load environment variables
load_dotenv()

# Import LLM abstraction layer for new functionality
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))
    from llm_factory import LLMClientFactory, get_chat_service
    from llm_types import LLMProvider, TextLLMClient, ChatMessage as LLMChatMessage
    LLM_ABSTRACTION_AVAILABLE = True
except ImportError:
    LLM_ABSTRACTION_AVAILABLE = False

class ChatMessage(BaseModel):
    """Pydantic model for chat messages"""
    role: str  # 'user' or 'assistant' or 'system'
    content: str

class ChatService:
    """
    A unified chat service that can work with different LLM providers
    """
    
    def __init__(self, service: str = "ollama", model: str = None, **kwargs):
        """
        Initialize the chat service
        
        Args:
            service: Either "ollama" or "gemini"
            model: Model name to use
            **kwargs: Additional parameters for the specific service
        """
        self.service = service.lower()
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        if self.service == "ollama":
            self._init_ollama(**kwargs)
        elif self.service == "gemini":
            self._init_gemini(**kwargs)
        else:
            raise ValueError(f"Unsupported service: {service}. Use 'ollama' or 'gemini'")
    
    def _init_ollama(self, **kwargs):
        """Initialize Ollama configuration"""
        default_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
        self.model = self.model or default_model
        
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        print(f"Initialized Ollama with model: {self.model}")
        print(f"Using URL: {self.base_url}")
    
    def _init_gemini(self, **kwargs):
        """Initialize Gemini configuration"""
        default_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
        self.model = self.model or default_model
        
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_AI_STUDIO_KEY not found in environment variables")
        
        self.api_base_url = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        
        print(f"Initialized Gemini with model: {self.model}")
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """
        Send a message and get a response
        
        Args:
            message: The user's message
            chat_history: Optional list of previous messages for context
            
        Returns:
            The AI's response as a string
        """
        # Build the full conversation
        messages = []
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add the current message
        messages.append(ChatMessage(role='user', content=message))
        
        # Get response based on service
        try:
            if self.service == "ollama":
                return self._call_ollama(messages)
            elif self.service == "gemini":
                return self._call_gemini(messages)
            else:
                raise ValueError(f"Unsupported service: {self.service}")
        except Exception as e:
            raise RuntimeError(f"Error getting response from {self.service}: {str(e)}")
    
    def chat_stream(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Generator[str, None, str]:
        """
        Send a message and get a streaming response
        
        Args:
            message: The user's message
            chat_history: Optional list of previous messages for context
            
        Yields:
            str: Partial response chunks as they arrive
            
        Returns:
            str: Complete response when streaming is done
        """
        # Build the full conversation
        messages = []
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add the current message
        messages.append(ChatMessage(role='user', content=message))
        
        # Get streaming response based on service
        try:
            if self.service == "ollama":
                return self._call_ollama_stream(messages)
            elif self.service == "gemini":
                return self._call_gemini_stream(messages)
            else:
                raise ValueError(f"Unsupported service: {self.service}")
        except Exception as e:
            # Fall back to regular chat if streaming fails
            full_response = self.chat(message, chat_history)
            yield full_response
            return full_response
    
    def _call_ollama(self, messages: List[ChatMessage]) -> str:
        """Make API call to Ollama"""
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
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def _call_gemini(self, messages: List[ChatMessage]) -> str:
        """Make API call to Google Gemini"""
        # Convert messages to Gemini format
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
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.api_base_url}/models/{self.model}:generateContent",
                json=payload,
                headers={"x-goog-api-key": self.api_key}
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    def _call_ollama_stream(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
        """Make streaming API call to Ollama"""
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
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        full_response = ""
        try:
            with httpx.Client(timeout=60.0) as client:
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
            full_response = self._call_ollama(messages)
            yield full_response
        
        return full_response
    
    def _call_gemini_stream(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
        """Make streaming API call to Google Gemini"""
        # Gemini streaming is more complex, so we'll simulate it for now
        try:
            # Get the full response first
            full_response = self._call_gemini(messages)
            
            # Simulate streaming by yielding words in chunks
            words = full_response.split()
            current_chunk = ""
            for i, word in enumerate(words):
                current_chunk += word + " "
                if i % 3 == 0 or i == len(words) - 1:  # Yield every 3 words
                    yield current_chunk.strip()
                    current_chunk = ""
                    import time
                    time.sleep(0.02)  # Small delay to simulate streaming
            
            return full_response
            
        except Exception as e:
            # Fall back to regular response
            full_response = self._call_gemini(messages)
            yield full_response
            return full_response
    
    def chat_with_history(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> tuple[str, List[Dict[str, str]]]:
        """
        Chat with automatic history management using simple dictionaries
        
        Args:
            message: The user's message
            chat_history: List of dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (response, updated_chat_history)
        """
        if chat_history is None:
            chat_history = []
        
        # Convert dict history to ChatMessage objects
        chat_messages = []
        for msg in chat_history:
            chat_messages.append(ChatMessage(role=msg['role'], content=msg['content']))
        
        # Get response
        response = self.chat(message, chat_messages)
        
        # Update history
        updated_history = chat_history.copy()
        updated_history.append({'role': 'user', 'content': message})
        updated_history.append({'role': 'assistant', 'content': response})
        
        return response, updated_history

# Modern factory functions using the new LLM abstraction
def create_chat_service(provider: str, model: str = None, **kwargs):
    """
    Create a chat service using the new LLM abstraction layer (recommended)
    
    Args:
        provider: LLM provider ('gemini', 'ollama')
        model: Model name (optional)
        **kwargs: Additional configuration
        
    Returns:
        ChatService-compatible adapter or legacy ChatService
    """
    if LLM_ABSTRACTION_AVAILABLE:
        return get_chat_service(provider, model, **kwargs)
    else:
        # Fallback to legacy implementation
        return ChatService(provider, model, **kwargs)

# Convenience functions for quick usage
def quick_chat_ollama(message: str, model: str = None) -> str:
    """Quick one-off chat with Ollama"""
    chat_service = create_chat_service("ollama", model)
    return chat_service.chat(message)

def quick_chat_gemini(message: str, model: str = None) -> str:
    """Quick one-off chat with Gemini"""
    chat_service = create_chat_service("gemini", model)
    return chat_service.chat(message)

# Streaming convenience functions
def quick_chat_ollama_stream(message: str, model: str = None) -> Generator[str, None, str]:
    """Quick one-off streaming chat with Ollama"""
    chat_service = create_chat_service("ollama", model)
    return chat_service.chat_stream(message)

def quick_chat_gemini_stream(message: str, model: str = None) -> Generator[str, None, str]:
    """Quick one-off streaming chat with Gemini"""
    chat_service = create_chat_service("gemini", model)
    return chat_service.chat_stream(message)

# Example usage
if __name__ == "__main__":
    # Test with Ollama
    print("=== Testing Ollama ===")
    ollama_chat = ChatService("ollama")
    response1 = ollama_chat.chat("Tell me a short joke")
    print(f"Ollama: {response1}")
    
    print("\n=== Testing Gemini ===")
    gemini_chat = ChatService("gemini")
    response2 = gemini_chat.chat("Tell me a short joke")
    print(f"Gemini: {response2}")
    
    print("\n=== Testing with History ===")
    history = []
    response3, history = gemini_chat.chat_with_history("What's 2+2?", history)
    print(f"Gemini: {response3}")
    
    response4, history = gemini_chat.chat_with_history("What about 3+3?", history)
    print(f"Gemini: {response4}")
    
    print(f"\nFinal history: {history}")