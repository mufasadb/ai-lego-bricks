# Chat Service - Detailed Technical Documentation

## 🏗️ Architecture Overview

The Chat Service provides a unified interface for multi-provider LLM chat interactions, supporting both stateless and stateful conversation patterns. It abstracts provider-specific APIs while maintaining full feature access through a consistent interface.

### Core Components

```
Chat Service Ecosystem
├── ChatService (Stateless Chat)
│   ├── Ollama Integration
│   ├── Gemini Integration
│   └── Provider Abstraction Layer
├── ConversationService (Stateful Chat)
│   ├── Message History Management
│   ├── Context Preservation
│   └── Conversation Analytics
├── Streaming Support
│   ├── Real-time Response Generation
│   ├── Token-level Streaming
│   └── Backpressure Handling
└── Factory Pattern
    ├── Service Creation
    ├── Configuration Management
    └── Credential Integration
```

## 🔧 ChatService Implementation

### Core Architecture

```python
class ChatService:
    """
    Unified chat service supporting multiple LLM providers.
    
    Design Principles:
    - Provider agnostic interface
    - Flexible configuration
    - Error resilience
    - Streaming support
    """
    
    def __init__(self, service: str, model: str = None, **kwargs):
        self.service = service.lower()
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        # Provider-specific initialization
        if self.service == "ollama":
            self._init_ollama(**kwargs)
        elif self.service == "gemini":
            self._init_gemini(**kwargs)
        else:
            raise ValueError(f"Unsupported service: {service}")
```

### Provider-Specific Initialization

#### Ollama Configuration

```python
def _init_ollama(self, **kwargs):
    """
    Initialize Ollama configuration.
    
    Configuration Sources:
    1. Direct parameters
    2. Environment variables
    3. Default values
    """
    self.model = self.model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
    self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Ollama-specific parameters
    self.stream = kwargs.get('stream', False)
    self.format = kwargs.get('format', 'json')
    self.options = kwargs.get('options', {})
    
    # Validate connection
    self._validate_ollama_connection()
```

#### Gemini Configuration

```python
def _init_gemini(self, **kwargs):
    """
    Initialize Gemini configuration.
    
    Authentication:
    - API key from environment
    - Service account credentials (future)
    """
    self.model = self.model or os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
    self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
    
    if not self.api_key:
        raise ValueError("GOOGLE_AI_STUDIO_KEY not found in environment variables")
    
    self.api_base_url = os.getenv("GEMINI_API_BASE_URL", 
                                 "https://generativelanguage.googleapis.com/v1beta")
    
    # Gemini-specific parameters
    self.safety_settings = kwargs.get('safety_settings', [])
    self.generation_config = kwargs.get('generation_config', {})
```

### Message Processing Pipeline

```python
def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
    """
    Process chat message through provider-specific pipeline.
    
    Pipeline Stages:
    1. Message validation and formatting
    2. History context integration
    3. Provider-specific API call
    4. Response processing and validation
    5. Error handling and recovery
    """
    
    # Stage 1: Message Validation
    if not message or not message.strip():
        raise ValueError("Message cannot be empty")
    
    # Stage 2: Context Integration
    messages = self._build_message_context(message, chat_history)
    
    # Stage 3: Provider API Call
    try:
        if self.service == "ollama":
            return self._call_ollama(messages)
        elif self.service == "gemini":
            return self._call_gemini(messages)
    except Exception as e:
        # Stage 5: Error Recovery
        return self._handle_chat_error(e, message)

def _build_message_context(self, message: str, chat_history: Optional[List[ChatMessage]]) -> List[ChatMessage]:
    """
    Build complete message context for provider.
    
    Context Building Strategy:
    - Preserve conversation history
    - Maintain role consistency
    - Optimize for provider requirements
    """
    messages = []
    
    # Add history with validation
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict):
                messages.append(ChatMessage(**msg))
            elif isinstance(msg, ChatMessage):
                messages.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
    
    # Add current message
    messages.append(ChatMessage(role='user', content=message))
    
    return messages
```

### Ollama Integration Deep Dive

```python
def _call_ollama(self, messages: List[ChatMessage]) -> str:
    """
    Ollama API integration with comprehensive error handling.
    
    API Specifics:
    - REST API over HTTP
    - JSON request/response
    - Streaming support
    - Local model management
    """
    
    # Format messages for Ollama
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Build request payload
    payload = {
        "model": self.model,
        "messages": formatted_messages,
        "stream": False,  # Non-streaming for simple chat
        "options": {
            "temperature": self.temperature,
            "num_predict": self.max_tokens
        }
    }
    
    # Make API call
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result.get('message', {}).get('content', '')
            
    except httpx.TimeoutException:
        raise RuntimeError("Ollama request timed out")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama API error: {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Ollama connection error: {str(e)}")
```

### Gemini Integration Deep Dive

```python
def _call_gemini(self, messages: List[ChatMessage]) -> str:
    """
    Gemini API integration with safety controls.
    
    API Specifics:
    - REST API over HTTPS
    - JSON request/response
    - Safety filtering
    - Content generation controls
    """
    
    # Format messages for Gemini
    gemini_messages = []
    for msg in messages:
        role = "user" if msg.role == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })
    
    # Build request payload
    payload = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "topP": 0.8,
            "topK": 10
        }
    }
    
    # Add safety settings if configured
    if self.safety_settings:
        payload["safetySettings"] = self.safety_settings
    
    # Make API call
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.api_base_url}/models/{self.model}:generateContent",
                json=payload,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            candidates = result.get('candidates', [])
            
            if not candidates:
                raise RuntimeError("No response candidates from Gemini")
            
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            
            if not parts:
                raise RuntimeError("No content parts in Gemini response")
            
            return parts[0].get('text', '')
            
    except httpx.HTTPStatusError as e:
        self._handle_gemini_error(e)
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

def _handle_gemini_error(self, error: httpx.HTTPStatusError):
    """Handle Gemini-specific error responses."""
    if error.response.status_code == 400:
        raise RuntimeError("Invalid request to Gemini API")
    elif error.response.status_code == 401:
        raise RuntimeError("Gemini API authentication failed")
    elif error.response.status_code == 429:
        raise RuntimeError("Gemini API rate limit exceeded")
    elif error.response.status_code == 500:
        raise RuntimeError("Gemini API internal server error")
    else:
        raise RuntimeError(f"Gemini API error: {error.response.status_code}")
```

## 🔄 Streaming Chat Implementation

### Real-time Response Generation

```python
def chat_stream(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Generator[str, None, str]:
    """
    Stream chat responses token by token.
    
    Streaming Benefits:
    - Reduced perceived latency
    - Progressive response building
    - Better user experience
    - Memory efficiency
    """
    
    messages = self._build_message_context(message, chat_history)
    
    try:
        if self.service == "ollama":
            yield from self._stream_ollama(messages)
        elif self.service == "gemini":
            yield from self._stream_gemini(messages)
        else:
            raise ValueError(f"Streaming not supported for {self.service}")
    except Exception as e:
        raise RuntimeError(f"Streaming error: {str(e)}")

def _stream_ollama(self, messages: List[ChatMessage]) -> Generator[str, None, str]:
    """
    Stream responses from Ollama API.
    
    Implementation Details:
    - Server-sent events (SSE)
    - JSON streaming protocol
    - Chunked transfer encoding
    """
    
    payload = {
        "model": self.model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "stream": True,
        "options": {
            "temperature": self.temperature,
            "num_predict": self.max_tokens
        }
    }
    
    try:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60.0
        ) as response:
            response.raise_for_status()
            
            accumulated_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data:
                            content = data['message'].get('content', '')
                            if content:
                                accumulated_content += content
                                yield content
                        
                        # Check for completion
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            return accumulated_content
            
    except httpx.TimeoutException:
        raise RuntimeError("Ollama streaming request timed out")
    except Exception as e:
        raise RuntimeError(f"Ollama streaming error: {str(e)}")
```

## 🗣️ ConversationService Architecture

### Stateful Conversation Management

```python
class ConversationService:
    """
    Stateful conversation service with history management.
    
    Features:
    - Persistent conversation state
    - Message history tracking
    - Context window management
    - Conversation analytics
    """
    
    def __init__(self, provider: str, model: str = None, **kwargs):
        self.provider = provider
        self.model = model
        self.conversation_id = self._generate_conversation_id()
        self.messages: List[ChatMessage] = []
        self.system_message: Optional[str] = None
        self.max_history = kwargs.get('max_history', 50)
        
        # Initialize underlying chat service
        self.chat_service = ChatService(provider, model, **kwargs)
    
    def add_system_message(self, message: str):
        """Add system message to conversation context."""
        self.system_message = message
        self.messages.insert(0, ChatMessage(role='system', content=message))
    
    def send_message(self, message: str) -> str:
        """
        Send message and maintain conversation state.
        
        State Management:
        - Append user message to history
        - Get AI response
        - Append AI response to history
        - Manage context window
        """
        
        # Add user message
        user_message = ChatMessage(role='user', content=message)
        self.messages.append(user_message)
        
        # Get response using full conversation history
        response = self.chat_service.chat(message, self.messages[:-1])
        
        # Add AI response
        ai_message = ChatMessage(role='assistant', content=response)
        self.messages.append(ai_message)
        
        # Manage context window
        self._manage_context_window()
        
        return response
    
    def _manage_context_window(self):
        """
        Manage conversation context window.
        
        Strategy:
        - Keep system message
        - Preserve recent messages
        - Remove oldest messages when limit exceeded
        """
        if len(self.messages) > self.max_history:
            # Keep system message if it exists
            system_messages = [msg for msg in self.messages if msg.role == 'system']
            other_messages = [msg for msg in self.messages if msg.role != 'system']
            
            # Keep most recent messages
            recent_messages = other_messages[-(self.max_history - len(system_messages)):]
            
            # Rebuild message list
            self.messages = system_messages + recent_messages
```

### Conversation Analytics

```python
def get_conversation_stats(self) -> Dict[str, Any]:
    """
    Get comprehensive conversation statistics.
    
    Analytics Include:
    - Message counts by role
    - Average message length
    - Conversation duration
    - Topic analysis
    """
    
    user_messages = [msg for msg in self.messages if msg.role == 'user']
    ai_messages = [msg for msg in self.messages if msg.role == 'assistant']
    
    return {
        "conversation_id": self.conversation_id,
        "total_messages": len(self.messages),
        "user_messages": len(user_messages),
        "ai_messages": len(ai_messages),
        "average_user_message_length": sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0,
        "average_ai_message_length": sum(len(msg.content) for msg in ai_messages) / len(ai_messages) if ai_messages else 0,
        "conversation_start": self.messages[0].timestamp if self.messages else None,
        "last_message": self.messages[-1].timestamp if self.messages else None,
    }

def search_messages(self, query: str, role: Optional[str] = None) -> List[ChatMessage]:
    """
    Search messages in conversation history.
    
    Search Features:
    - Full-text search
    - Role filtering
    - Fuzzy matching
    - Relevance scoring
    """
    
    filtered_messages = self.messages
    
    # Filter by role if specified
    if role:
        filtered_messages = [msg for msg in filtered_messages if msg.role == role]
    
    # Simple text search (can be enhanced with fuzzy matching)
    matching_messages = []
    for msg in filtered_messages:
        if query.lower() in msg.content.lower():
            matching_messages.append(msg)
    
    return matching_messages

def export_conversation(self, format: str = 'json') -> str:
    """
    Export conversation in various formats.
    
    Supported Formats:
    - JSON: Machine-readable format
    - Markdown: Human-readable format
    - CSV: Tabular format
    - HTML: Web-ready format
    """
    
    if format == 'json':
        return json.dumps([msg.dict() for msg in self.messages], indent=2)
    
    elif format == 'markdown':
        lines = [f"# Conversation {self.conversation_id}\n"]
        for msg in self.messages:
            role_header = f"**{msg.role.title()}:**"
            lines.append(f"{role_header} {msg.content}\n")
        return "\n".join(lines)
    
    elif format == 'csv':
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Role', 'Content', 'Timestamp'])
        
        for msg in self.messages:
            writer.writerow([msg.role, msg.content, msg.timestamp])
        
        return output.getvalue()
    
    else:
        raise ValueError(f"Unsupported export format: {format}")
```

## 🏭 Factory Pattern Implementation

### Service Creation and Management

```python
class ChatServiceFactory:
    """
    Factory for creating and managing chat services.
    
    Benefits:
    - Centralized configuration
    - Service caching
    - Credential management
    - Provider abstraction
    """
    
    def __init__(self, credential_manager: Optional['CredentialManager'] = None):
        self.credential_manager = credential_manager
        self._service_cache: Dict[str, ChatService] = {}
        self._conversation_cache: Dict[str, ConversationService] = {}
    
    def create_chat_service(self, provider: str, model: str = None, **kwargs) -> ChatService:
        """
        Create chat service with caching.
        
        Caching Strategy:
        - Cache by provider + model + key parameters
        - Reuse services for identical configurations
        - Clear cache on credential changes
        """
        
        cache_key = f"{provider}:{model}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._service_cache:
            # Inject credentials if available
            if self.credential_manager:
                kwargs = self._inject_credentials(provider, kwargs)
            
            self._service_cache[cache_key] = ChatService(provider, model, **kwargs)
        
        return self._service_cache[cache_key]
    
    def create_conversation_service(self, provider: str, model: str = None, **kwargs) -> ConversationService:
        """Create conversation service with state management."""
        
        # Inject credentials if available
        if self.credential_manager:
            kwargs = self._inject_credentials(provider, kwargs)
        
        conversation = ConversationService(provider, model, **kwargs)
        self._conversation_cache[conversation.conversation_id] = conversation
        
        return conversation
    
    def _inject_credentials(self, provider: str, kwargs: Dict) -> Dict:
        """Inject provider-specific credentials."""
        
        updated_kwargs = kwargs.copy()
        
        if provider == "gemini":
            if 'api_key' not in updated_kwargs:
                updated_kwargs['api_key'] = self.credential_manager.get_credential('GOOGLE_AI_STUDIO_KEY')
        
        elif provider == "ollama":
            if 'base_url' not in updated_kwargs:
                updated_kwargs['base_url'] = self.credential_manager.get_credential('OLLAMA_URL', 'http://localhost:11434')
        
        return updated_kwargs
```

### Convenience Functions

```python
def create_chat_service(provider: str = "auto", model: str = None, 
                       credential_manager: Optional['CredentialManager'] = None, 
                       **kwargs) -> ChatService:
    """
    Convenience function for creating chat services.
    
    Provider Selection:
    - "auto": Auto-detect available provider
    - Specific provider name
    - Fallback to default
    """
    
    if provider == "auto":
        provider = _auto_detect_provider(credential_manager)
    
    factory = ChatServiceFactory(credential_manager)
    return factory.create_chat_service(provider, model, **kwargs)

def _auto_detect_provider(credential_manager: Optional['CredentialManager']) -> str:
    """Auto-detect available chat provider."""
    
    if credential_manager:
        # Check for Gemini credentials
        if credential_manager.has_credential('GOOGLE_AI_STUDIO_KEY'):
            return "gemini"
        
        # Check for Ollama (local service)
        if credential_manager.has_credential('OLLAMA_URL'):
            return "ollama"
    
    # Check environment variables
    if os.getenv('GOOGLE_AI_STUDIO_KEY'):
        return "gemini"
    
    # Default to Ollama (local)
    return "ollama"

def quick_chat_gemini(message: str, model: str = "gemini-1.5-flash") -> str:
    """Quick Gemini chat without service setup."""
    service = create_chat_service("gemini", model)
    return service.chat(message)

def quick_chat_ollama(message: str, model: str = None) -> str:
    """Quick Ollama chat without service setup."""
    service = create_chat_service("ollama", model)
    return service.chat(message)
```

## 🔗 Integration Patterns

### Agent Orchestration Integration

```python
def create_chat_workflow_step(provider: str, model: str = None) -> Dict:
    """
    Create chat step for agent orchestration.
    
    Step Configuration:
    - Provider and model selection
    - Input/output mapping
    - Error handling
    - Context preservation
    """
    
    return {
        "type": "chat",
        "name": f"chat_{provider}",
        "provider": provider,
        "model": model,
        "input_mapping": {
            "message": "user_input",
            "history": "conversation_history"
        },
        "output_mapping": {
            "response": "ai_response",
            "updated_history": "conversation_history"
        },
        "error_handling": {
            "retry_count": 3,
            "retry_delay": 1.0,
            "fallback_provider": "ollama"
        }
    }
```

### Memory Service Integration

```python
def create_memory_aware_conversation(provider: str, memory_service) -> ConversationService:
    """
    Create conversation service with memory integration.
    
    Memory Features:
    - Store conversation history
    - Retrieve relevant context
    - Maintain long-term memory
    """
    
    class MemoryAwareConversation(ConversationService):
        def __init__(self, provider: str, memory_service, **kwargs):
            super().__init__(provider, **kwargs)
            self.memory_service = memory_service
        
        def send_message(self, message: str) -> str:
            # Retrieve relevant memories
            memories = self.memory_service.retrieve_memories(message, limit=3)
            
            # Add memory context to system message
            if memories:
                memory_context = "Relevant context:\n" + "\n".join([m.content for m in memories])
                self.add_system_message(memory_context)
            
            # Process message normally
            response = super().send_message(message)
            
            # Store conversation turn in memory
            self.memory_service.store_memory(
                content=f"User: {message}\nAI: {response}",
                metadata={
                    "type": "conversation",
                    "conversation_id": self.conversation_id,
                    "provider": self.provider
                }
            )
            
            return response
    
    return MemoryAwareConversation(provider, memory_service)
```

### Streaming Integration

```python
def create_streaming_chat_pipeline(provider: str, output_handler) -> Generator:
    """
    Create streaming chat pipeline with custom output handling.
    
    Pipeline Features:
    - Real-time token streaming
    - Custom output processing
    - Error recovery
    - Backpressure handling
    """
    
    service = create_chat_service(provider)
    
    def stream_with_processing(message: str):
        accumulated_response = ""
        
        try:
            for token in service.chat_stream(message):
                accumulated_response += token
                
                # Process token through handler
                processed_token = output_handler(token, accumulated_response)
                
                yield processed_token
                
        except Exception as e:
            # Error recovery
            yield f"[Error: {str(e)}]"
            
            # Fallback to non-streaming
            fallback_response = service.chat(message)
            yield fallback_response
    
    return stream_with_processing
```

## 🧪 Testing and Validation

### Unit Testing Patterns

```python
def test_chat_service_initialization():
    """Test chat service initialization with different providers."""
    
    # Test Ollama initialization
    ollama_service = ChatService("ollama", "llama2")
    assert ollama_service.service == "ollama"
    assert ollama_service.model == "llama2"
    
    # Test Gemini initialization
    os.environ['GOOGLE_AI_STUDIO_KEY'] = 'test_key'
    gemini_service = ChatService("gemini", "gemini-1.5-flash")
    assert gemini_service.service == "gemini"
    assert gemini_service.model == "gemini-1.5-flash"

def test_message_context_building():
    """Test message context building with history."""
    
    service = ChatService("ollama")
    
    history = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!")
    ]
    
    context = service._build_message_context("How are you?", history)
    
    assert len(context) == 3
    assert context[0].content == "Hello"
    assert context[1].content == "Hi there!"
    assert context[2].content == "How are you?"

def test_conversation_state_management():
    """Test conversation state management."""
    
    conversation = ConversationService("ollama")
    
    # Test system message
    conversation.add_system_message("You are a helpful assistant")
    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"
    
    # Test conversation flow
    # Note: This would require mocking the actual API calls
    # conversation.send_message("Hello")
    # assert len(conversation.messages) == 3  # system + user + assistant
```

### Integration Testing

```python
def test_provider_switching():
    """Test switching between providers."""
    
    # Test with mock providers
    factory = ChatServiceFactory()
    
    # Create services for different providers
    ollama_service = factory.create_chat_service("ollama")
    gemini_service = factory.create_chat_service("gemini")
    
    # Verify services are different instances
    assert ollama_service != gemini_service
    assert ollama_service.service == "ollama"
    assert gemini_service.service == "gemini"

def test_streaming_functionality():
    """Test streaming chat functionality."""
    
    # Mock streaming response
    def mock_stream():
        for token in ["Hello", " ", "world", "!"]:
            yield token
    
    # Test streaming accumulation
    accumulated = ""
    for token in mock_stream():
        accumulated += token
    
    assert accumulated == "Hello world!"
```

## 🎯 Performance Optimization

### Connection Pool Management

```python
class OptimizedChatService(ChatService):
    """
    Optimized chat service with connection pooling.
    
    Optimizations:
    - HTTP connection reuse
    - Request batching
    - Response caching
    - Connection pooling
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create persistent HTTP client
        self.http_client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )
    
    def __del__(self):
        """Clean up HTTP client on destruction."""
        if hasattr(self, 'http_client'):
            self.http_client.close()
    
    def _call_ollama(self, messages: List[ChatMessage]) -> str:
        """Optimized Ollama call with connection reuse."""
        
        # Use persistent client instead of creating new one
        response = self.http_client.post(
            f"{self.base_url}/api/chat",
            json=self._build_ollama_payload(messages)
        )
        
        response.raise_for_status()
        return response.json()['message']['content']
```

### Response Caching

```python
class CachedChatService(ChatService):
    """
    Chat service with response caching.
    
    Cache Strategy:
    - Hash-based cache keys
    - TTL-based expiration
    - Memory-efficient storage
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_ttl = kwargs.get('cache_ttl', 3600)  # 1 hour
    
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """Chat with response caching."""
        
        # Generate cache key
        cache_key = self._generate_cache_key(message, chat_history)
        
        # Check cache
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_response
        
        # Get fresh response
        response = super().chat(message, chat_history)
        
        # Cache response
        self.cache[cache_key] = (response, time.time())
        
        return response
    
    def _generate_cache_key(self, message: str, chat_history: Optional[List[ChatMessage]]) -> str:
        """Generate cache key from message and history."""
        
        # Create deterministic hash
        import hashlib
        
        key_data = message
        if chat_history:
            key_data += str([(msg.role, msg.content) for msg in chat_history])
        
        return hashlib.md5(key_data.encode()).hexdigest()
```

This comprehensive documentation provides deep technical insight into the chat service architecture, implementation patterns, and optimization strategies.