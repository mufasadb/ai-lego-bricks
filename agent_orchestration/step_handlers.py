"""
Step handlers for different workflow step types
"""

from typing import Any, Dict, Callable, Type, List
try:
    from .models import StepConfig, StepType, ExecutionContext, ConversationThread, ConversationMessage
except ImportError:
    from models import StepConfig, StepType, ExecutionContext, ConversationThread, ConversationMessage
from pydantic import BaseModel
import importlib
import json

# Import LLM providers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_types import LLMProvider, VisionProvider
from chunking.chunking_service import ChunkingConfig
from pdf_to_text.pdf_to_text_service import PDFExtractOptions


class StepHandlerRegistry:
    """Registry of step handlers for different step types"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.handlers: Dict[StepType, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register all default step handlers"""
        self.handlers[StepType.INPUT] = self._handle_input
        self.handlers[StepType.DOCUMENT_PROCESSING] = self._handle_document_processing
        self.handlers[StepType.MEMORY_STORE] = self._handle_memory_store
        self.handlers[StepType.MEMORY_RETRIEVE] = self._handle_memory_retrieve
        self.handlers[StepType.LLM_CHAT] = self._handle_llm_chat
        self.handlers[StepType.LLM_STRUCTURED] = self._handle_llm_structured
        self.handlers[StepType.LLM_VISION] = self._handle_llm_vision
        self.handlers[StepType.CHUNK_TEXT] = self._handle_chunk_text
        self.handlers[StepType.CONDITION] = self._handle_condition
        self.handlers[StepType.OUTPUT] = self._handle_output
        self.handlers[StepType.FILE_OUTPUT] = self._handle_file_output
        self.handlers[StepType.HUMAN_APPROVAL] = self._handle_human_approval
        # Conversation management handlers
        self.handlers[StepType.START_CONVERSATION] = self._handle_start_conversation
        self.handlers[StepType.ADD_TO_CONVERSATION] = self._handle_add_to_conversation
        self.handlers[StepType.CONTINUE_CONVERSATION] = self._handle_continue_conversation
    
    def get_handler(self, step_type: StepType) -> Callable:
        """Get handler for a specific step type"""
        return self.handlers.get(step_type)
    
    def register_handler(self, step_type: StepType, handler: Callable):
        """Register a custom handler for a step type"""
        self.handlers[step_type] = handler
    
    def _handle_input(self, step: StepConfig, inputs: Dict[str, Any], 
                     context: ExecutionContext) -> Any:
        """Handle input step - typically collects user input or external data"""
        # For now, just return the configured input or prompt for it
        if "value" in step.config:
            return step.config["value"]
        elif "prompt" in step.config:
            # In a real implementation, this would prompt the user
            return input(step.config["prompt"])
        else:
            return inputs
    
    def _handle_start_conversation(self, step: StepConfig, inputs: Dict[str, Any], 
                                  context: ExecutionContext) -> Any:
        """Handle starting a new conversation thread"""
        conversation_id = step.config.get("conversation_id") or inputs.get("conversation_id")
        if not conversation_id:
            # Generate a default conversation ID
            conversation_id = f"conversation_{len(context.conversations) + 1}"
        
        conversation_name = step.config.get("name") or inputs.get("name")
        
        # Create the conversation
        conversation = context.create_conversation(conversation_id, conversation_name)
        
        # Set as active conversation
        context.set_active_conversation(conversation_id)
        
        # Add initial system message if provided
        system_message = step.config.get("system_message") or inputs.get("system_message")
        if system_message:
            conversation.add_message("system", system_message)
        
        return {
            "conversation_id": conversation_id,
            "conversation_name": conversation_name,
            "system_message": system_message,
            "message_count": len(conversation.messages)
        }
    
    def _handle_add_to_conversation(self, step: StepConfig, inputs: Dict[str, Any], 
                                   context: ExecutionContext) -> Any:
        """Handle adding a message to a conversation thread"""
        conversation_id = step.config.get("conversation_id") or inputs.get("conversation_id")
        role = step.config.get("role") or inputs.get("role")
        content = inputs.get("content") or inputs.get("message")
        
        if not content:
            raise ValueError("content or message required for add_to_conversation")
        
        if not role:
            raise ValueError("role required for add_to_conversation")
        
        # Get the conversation (use active if not specified)
        if conversation_id:
            if conversation_id not in context.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = context.conversations[conversation_id]
        else:
            conversation = context.get_active_conversation()
            if not conversation:
                raise ValueError("No active conversation found and no conversation_id specified")
            conversation_id = conversation.id
        
        # Add the message
        metadata = inputs.get("metadata", {})
        message = conversation.add_message(role, content, metadata)
        
        return {
            "conversation_id": conversation_id,
            "message_role": role,
            "message_content": content,
            "message_timestamp": message.timestamp.isoformat(),
            "total_messages": len(conversation.messages)
        }
    
    def _handle_continue_conversation(self, step: StepConfig, inputs: Dict[str, Any], 
                                     context: ExecutionContext) -> Any:
        """Handle continuing a conversation with an LLM response"""
        conversation_id = step.config.get("conversation_id") or inputs.get("conversation_id")
        user_message = inputs.get("message") or inputs.get("user_message")
        
        if not user_message:
            raise ValueError("message or user_message required for continue_conversation")
        
        # Get the conversation (use active if not specified)
        if conversation_id:
            if conversation_id not in context.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = context.conversations[conversation_id]
        else:
            conversation = context.get_active_conversation()
            if not conversation:
                raise ValueError("No active conversation found and no conversation_id specified")
            conversation_id = conversation.id
        
        # Add user message to conversation
        conversation.add_message("user", user_message)
        
        # Get LLM configuration
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")
        
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        
        # Create LLM client
        if provider == "gemini":
            client = llm_factory.create_text_client(LLMProvider.GEMINI, model)
        elif provider == "ollama":
            client = llm_factory.create_text_client(LLMProvider.OLLAMA, model)
        elif provider == "anthropic":
            client = llm_factory.create_text_client(LLMProvider.ANTHROPIC, model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Get conversation history for context
        include_system = step.config.get("include_system_messages", True)
        history_limit = step.config.get("history_limit", 0)  # 0 = all messages
        
        if history_limit > 0:
            recent_messages = conversation.get_last_messages(history_limit, include_system)
            llm_messages = [{
                'role': msg.role,
                'content': msg.content
            } for msg in recent_messages]
        else:
            llm_messages = conversation.get_messages_for_llm(include_system)
        
        # Generate response using conversation history
        # For now, we'll use a simple approach - create a prompt with history
        if len(llm_messages) > 1:
            # Build conversation context
            context_parts = []
            for msg in llm_messages[:-1]:  # All but the last message (current user message)
                context_parts.append(f"{msg['role']}: {msg['content']}")
            
            conversation_context = "\n".join(context_parts)
            full_prompt = f"Previous conversation:\n{conversation_context}\n\nCurrent message: {user_message}"
        else:
            full_prompt = user_message
        
        # Generate response
        response = client.chat(full_prompt)
        
        # Add assistant response to conversation
        conversation.add_message("assistant", response)
        
        return {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "assistant_response": response,
            "total_messages": len(conversation.messages),
            "provider": provider,
            "model": model,
            "history_used": len(llm_messages) > 1
        }
    
    def _handle_human_approval(self, step: StepConfig, inputs: Dict[str, Any], 
                              context: ExecutionContext) -> Any:
        """Handle human approval step - presents context and collects human input"""
        import signal
        import time
        
        # Configuration
        approval_type = step.config.get("approval_type", "approve_reject")
        prompt = step.config.get("prompt", "Please review and provide your decision:")
        options = step.config.get("options", ["approve", "reject"])
        timeout_seconds = step.config.get("timeout_seconds", 300)  # 5 minute default
        default_action = step.config.get("default_action", "reject")
        show_context = step.config.get("show_context", True)
        context_fields = step.config.get("context_fields", [])
        
        # Display context to human
        if show_context:
            self._display_approval_context(inputs, context, context_fields)
        
        # Present the prompt and options
        print(f"\n{'='*60}")
        print("HUMAN APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"{prompt}\n")
        
        if approval_type == "approve_reject":
            print("Options: [a]pprove, [r]eject")
            valid_inputs = {"a": "approve", "approve": "approve", "r": "reject", "reject": "reject"}
            
        elif approval_type == "multiple_choice":
            print(f"Options: {', '.join(f'[{i+1}] {opt}' for i, opt in enumerate(options))}")
            valid_inputs = {}
            for i, option in enumerate(options):
                valid_inputs[str(i+1)] = option
                valid_inputs[option.lower()] = option
                
        elif approval_type == "custom_input":
            print("Please provide your input:")
            valid_inputs = None  # Any input is valid
            
        else:
            raise ValueError(f"Unknown approval_type: {approval_type}")
        
        # Setup timeout handling
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Human approval timed out")
        
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            # Get human input
            while True:
                try:
                    if timeout_seconds > 0:
                        print(f"\n(Timeout in {timeout_seconds} seconds, default: {default_action})")
                    
                    user_input = input("\nYour response: ").strip().lower()
                    
                    if approval_type == "custom_input":
                        # For custom input, any non-empty response is valid
                        if user_input:
                            decision = user_input
                            feedback = user_input
                            break
                        else:
                            print("Please provide a non-empty response.")
                            continue
                    
                    elif user_input in valid_inputs:
                        decision = valid_inputs[user_input]
                        feedback = f"Selected: {decision}"
                        break
                    
                    else:
                        print(f"Invalid input. Please choose from: {list(valid_inputs.keys())}")
                        continue
                        
                except EOFError:
                    # Handle Ctrl+D
                    print(f"\nInput interrupted. Using default action: {default_action}")
                    decision = default_action
                    feedback = f"Default action used due to input interruption"
                    break
                    
        except TimeoutError:
            print(f"\nTimeout reached. Using default action: {default_action}")
            decision = default_action
            feedback = f"Default action used due to timeout ({timeout_seconds}s)"
            
        finally:
            # Restore original signal handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        # Return results that can be used by subsequent steps
        result = {
            "decision": decision,
            "feedback": feedback,
            "approval_type": approval_type,
            "prompt": prompt,
            "timestamp": time.time(),
            "timeout_seconds": timeout_seconds,
            "context_shown": show_context
        }
        
        # For custom input, also include the raw input
        if approval_type == "custom_input":
            result["user_input"] = user_input
        
        print(f"\nDecision recorded: {decision}")
        print(f"{'='*60}\n")
        
        return result
    
    def _display_approval_context(self, inputs: Dict[str, Any], 
                                 context: ExecutionContext, 
                                 context_fields: List[str]) -> None:
        """Display relevant context information to help human make decision"""
        print(f"\n{'='*60}")
        print("CONTEXT INFORMATION")
        print(f"{'='*60}")
        
        # Show step inputs
        if inputs:
            print("\nCurrent Step Inputs:")
            for key, value in inputs.items():
                if not context_fields or key in context_fields:
                    formatted_value = self._format_context_value(value)
                    print(f"  {key}: {formatted_value}")
        
        # Show relevant step outputs from workflow
        if context.step_outputs:
            print("\nPrevious Step Results:")
            for key, value in context.step_outputs.items():
                # If context_fields specified, only show those
                if context_fields:
                    # Check if this output matches any requested field
                    field_name = key.split('.')[-1] if '.' in key else key
                    if field_name not in context_fields:
                        continue
                
                formatted_value = self._format_context_value(value)
                print(f"  {key}: {formatted_value}")
        
        # Show global variables if relevant
        if context.global_variables:
            print("\nGlobal Variables:")
            for key, value in context.global_variables.items():
                if not context_fields or key in context_fields:
                    formatted_value = self._format_context_value(value)
                    print(f"  {key}: {formatted_value}")
    
    def _format_context_value(self, value: Any, max_length: int = 200) -> str:
        """Format a context value for display, truncating if too long"""
        if isinstance(value, (str, int, float, bool)):
            str_value = str(value)
        elif isinstance(value, dict):
            # For dicts, show structure but limit content
            if len(str(value)) <= max_length:
                str_value = str(value)
            else:
                # Show just the keys for large dicts
                str_value = f"{{dict with keys: {list(value.keys())}}}"
        elif isinstance(value, list):
            if len(str(value)) <= max_length:
                str_value = str(value)
            else:
                str_value = f"[list with {len(value)} items]"
        else:
            str_value = str(value)
        
        # Truncate if still too long
        if len(str_value) > max_length:
            str_value = str_value[:max_length-3] + "..."
            
        return str_value
    
    def _handle_document_processing(self, step: StepConfig, inputs: Dict[str, Any], 
                                   context: ExecutionContext) -> Any:
        """Handle document processing step"""
        pdf_service = self.orchestrator.get_service("pdf_processor")
        if not pdf_service:
            raise RuntimeError("PDF processor service not available")
        
        file_path = inputs.get("file_path")
        if not file_path:
            raise ValueError("file_path required for document processing")
        
        # Configure extraction options
        options = PDFExtractOptions()
        if "enhance_with_llm" in step.config:
            options.use_llm_enhancement = step.config["enhance_with_llm"]
        if "semantic_analysis" in step.config:
            options.extract_key_points = step.config["semantic_analysis"]
        
        # Extract text from PDF
        result = pdf_service.extract_text_from_file(file_path, options)
        
        return {
            "text": result.enhanced_text if result.enhanced_text else result.text,
            "original_text": result.text,
            "enhanced_text": result.enhanced_text,
            "key_points": result.key_points,
            "page_count": result.page_count
        }
    
    def _handle_memory_store(self, step: StepConfig, inputs: Dict[str, Any], 
                            context: ExecutionContext) -> Any:
        """Handle memory storage step"""
        memory_service = self.orchestrator.get_service("memory")
        if not memory_service:
            raise RuntimeError("Memory service not available")
        
        content = inputs.get("content")
        if not content:
            raise ValueError("content required for memory storage")
        
        metadata = inputs.get("metadata", {})
        metadata.update(step.config.get("metadata", {}))
        
        # Store memory
        memory_id = memory_service.store_memory(content, metadata)
        
        return {
            "memory_id": memory_id,
            "stored_content": content,
            "metadata": metadata
        }
    
    def _handle_memory_retrieve(self, step: StepConfig, inputs: Dict[str, Any], 
                               context: ExecutionContext) -> Any:
        """Handle memory retrieval step"""
        memory_service = self.orchestrator.get_service("memory")
        if not memory_service:
            raise RuntimeError("Memory service not available")
        
        query = inputs.get("query")
        if not query:
            raise ValueError("query required for memory retrieval")
        
        # Configure retrieval
        limit = step.config.get("limit", 5)
        threshold = step.config.get("threshold", 0.7)
        
        # Retrieve memories
        memories = memory_service.retrieve_memories(query, limit=limit, threshold=threshold)
        
        return {
            "memories": [
                {
                    "content": m.content,
                    "metadata": m.metadata,
                    "similarity": getattr(m, "similarity", None)
                }
                for m in memories
            ],
            "count": len(memories)
        }
    
    def _handle_llm_chat(self, step: StepConfig, inputs: Dict[str, Any], 
                        context: ExecutionContext) -> Any:
        """Handle LLM chat step - uses generation service for one-shot, conversation service for multi-turn"""
        message = inputs.get("message")
        if not message:
            raise ValueError("message required for LLM chat")
        
        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        temperature = step.config.get("temperature", 0.7)
        max_tokens = step.config.get("max_tokens", 1000)
        
        # Check if we should use conversation context
        use_conversation = step.config.get("use_conversation", False)
        conversation_id = step.config.get("conversation_id")
        
        if use_conversation:
            # Use conversation service for multi-turn conversations
            conversation_service = self.orchestrator.get_service("conversation")
            if not conversation_service:
                raise RuntimeError("Conversation service not available")
            
            # Check if this is a new conversation or continuing existing one
            if conversation_id:
                # Try to load existing conversation state if available
                # For now, create a new conversation service instance
                # In a full implementation, you might want to persist/restore conversation state
                from llm.llm_types import LLMProvider
                provider_enum = LLMProvider(provider)
                conversation_service = type(conversation_service)(
                    provider_enum, model, temperature, max_tokens, conversation_id
                )
            
            # Add system message if provided
            system_message = step.config.get("system_message")
            if system_message:
                conversation_service.add_system_message(system_message)
            
            # Send message and get response
            response = conversation_service.send_message(message)
            
            return {
                "response": response,
                "message": message,
                "provider": provider,
                "model": model,
                "conversation_id": conversation_service.conversation.id,
                "total_messages": conversation_service.get_conversation_length(),
                "first_prompt": conversation_service.get_first_prompt(),
                "last_response": conversation_service.get_last_response(),
                "service_type": "conversation"
            }
        
        else:
            # Use generation service for one-shot interactions
            generation_service = self.orchestrator.get_service("generation")
            if not generation_service:
                raise RuntimeError("Generation service not available")
            
            # Create generation service with specific config for this call
            from llm.llm_types import LLMProvider
            provider_enum = LLMProvider(provider)
            gen_service = type(generation_service)(provider_enum, model, temperature, max_tokens)
            
            # Check for system prompt
            system_prompt = step.config.get("system_message")
            if system_prompt:
                response = gen_service.generate_with_system_prompt(message, system_prompt)
            else:
                response = gen_service.generate(message)
            
            return {
                "response": response,
                "message": message,
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt,
                "service_type": "generation"
            }
    
    def _handle_llm_structured(self, step: StepConfig, inputs: Dict[str, Any], 
                              context: ExecutionContext) -> Any:
        """Handle structured LLM response step"""
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")
        
        message = inputs.get("message")
        if not message:
            raise ValueError("message required for LLM structured response")
        
        # Get schema configuration
        schema_config = step.config.get("response_schema")
        if not schema_config:
            raise ValueError("response_schema required for structured LLM step")
        
        # Parse schema - can be either a class reference or a direct Pydantic model
        schema_class = self._parse_schema_config(schema_config)
        
        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        
        # Create structured client
        structured_client = llm_factory.create_structured_client(
            provider=provider,
            schema=schema_class,
            model=model,
            temperature=step.config.get("temperature", 0.7),
            max_tokens=step.config.get("max_tokens", 1000)
        )
        
        # Generate structured response
        try:
            structured_response = structured_client.chat(message)
            
            # Convert to dict for easier access in workflows
            response_dict = structured_response.model_dump()
            
            return {
                "structured_response": structured_response,
                "response_dict": response_dict,
                "message": message,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                **response_dict  # Include all fields at top level for easy access
            }
            
        except Exception as e:
            # Handle structured response failures gracefully
            return {
                "error": str(e),
                "message": message,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                "structured_response": None,
                "response_dict": {}
            }
    
    def _parse_schema_config(self, schema_config: Any) -> Type[BaseModel]:
        """Parse schema configuration into a Pydantic model class"""
        if isinstance(schema_config, str):
            # String reference - try to import as module.ClassName
            if "." in schema_config:
                module_path, class_name = schema_config.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    schema_class = getattr(module, class_name)
                    if not (isinstance(schema_class, type) and issubclass(schema_class, BaseModel)):
                        raise ValueError(f"Schema class {schema_config} is not a Pydantic BaseModel")
                    return schema_class
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not import schema class {schema_config}: {e}")
            else:
                # Try to create a dynamic schema from predefined common schemas
                return self._create_dynamic_schema(schema_config)
        
        elif isinstance(schema_config, dict):
            # Dictionary schema definition - create dynamic model
            return self._create_schema_from_dict(schema_config)
        
        elif isinstance(schema_config, type) and issubclass(schema_config, BaseModel):
            # Direct Pydantic model class
            return schema_config
        
        else:
            raise ValueError(f"Invalid schema configuration: {schema_config}")
    
    def _create_dynamic_schema(self, schema_name: str) -> Type[BaseModel]:
        """Create a dynamic schema for common use cases"""
        common_schemas = {
            "simple_response": self._create_simple_response_schema,
            "classification": self._create_classification_schema,
            "extraction": self._create_extraction_schema,
            "decision": self._create_decision_schema,
            "summary": self._create_summary_schema
        }
        
        if schema_name in common_schemas:
            return common_schemas[schema_name]()
        else:
            raise ValueError(f"Unknown predefined schema: {schema_name}")
    
    def _create_schema_from_dict(self, schema_dict: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from a dictionary definition"""
        from pydantic import create_model
        from typing import get_type_hints
        
        # Extract fields from dictionary
        fields = {}
        for field_name, field_config in schema_dict.get("fields", {}).items():
            if isinstance(field_config, dict):
                field_type = field_config.get("type", str)
                field_default = field_config.get("default", ...)
                field_description = field_config.get("description", "")
                
                # Convert string type names to actual types
                if isinstance(field_type, str):
                    type_mapping = {
                        "str": str,
                        "string": str,
                        "int": int,
                        "integer": int,
                        "float": float,
                        "bool": bool,
                        "boolean": bool,
                        "list": list,
                        "dict": dict
                    }
                    field_type = type_mapping.get(field_type, str)
                
                if field_default == ...:
                    fields[field_name] = (field_type, ...)  # Required field
                else:
                    fields[field_name] = (field_type, field_default)
            else:
                # Simple field definition
                fields[field_name] = (str, ...)
        
        # Create dynamic model
        model_name = schema_dict.get("name", "DynamicSchema")
        return create_model(model_name, **fields)
    
    def _create_simple_response_schema(self) -> Type[BaseModel]:
        """Create a simple response schema"""
        class SimpleResponse(BaseModel):
            response: str
            confidence: float = 1.0
        return SimpleResponse
    
    def _create_classification_schema(self) -> Type[BaseModel]:
        """Create a classification schema"""
        class Classification(BaseModel):
            category: str
            confidence: float
            reasoning: str = ""
        return Classification
    
    def _create_extraction_schema(self) -> Type[BaseModel]:
        """Create an extraction schema"""
        class Extraction(BaseModel):
            extracted_data: Dict[str, Any]
            confidence: float
            source_text: str = ""
        return Extraction
    
    def _create_decision_schema(self) -> Type[BaseModel]:
        """Create a decision schema"""
        class Decision(BaseModel):
            decision: str
            reasoning: str
            confidence: float
            alternatives: List[str] = []
        return Decision
    
    def _create_summary_schema(self) -> Type[BaseModel]:
        """Create a summary schema"""
        class Summary(BaseModel):
            summary: str
            key_points: List[str]
            word_count: int = 0
        return Summary
    
    def _handle_llm_vision(self, step: StepConfig, inputs: Dict[str, Any], 
                          context: ExecutionContext) -> Any:
        """Handle LLM vision step"""
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")
        
        image_path = inputs.get("image_path")
        prompt = inputs.get("prompt", "Describe this image")
        
        if not image_path:
            raise ValueError("image_path required for LLM vision")
        
        # Configure vision model
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        
        # Create vision client
        if provider == "gemini":
            client = llm_factory.create_vision_client(VisionProvider.GEMINI_VISION, model)
        else:
            raise ValueError(f"Unsupported vision provider: {provider}")
        
        # Analyze image
        response = client.analyze_image(image_path, prompt)
        
        return {
            "analysis": response,
            "image_path": image_path,
            "prompt": prompt,
            "provider": provider,
            "model": model
        }
    
    def _handle_chunk_text(self, step: StepConfig, inputs: Dict[str, Any], 
                          context: ExecutionContext) -> Any:
        """Handle text chunking step"""
        chunking_service = self.orchestrator.get_service("chunking")
        if not chunking_service:
            raise RuntimeError("Chunking service not available")
        
        text = inputs.get("text")
        if not text:
            raise ValueError("text required for chunking")
        
        # Configure chunking with required parameters
        target_size = step.config.get("target_size", 1000)
        tolerance = step.config.get("tolerance", 200)
        
        config = ChunkingConfig(
            target_size=target_size,
            tolerance=tolerance,
            preserve_paragraphs=step.config.get("preserve_paragraphs", True),
            preserve_sentences=step.config.get("preserve_sentences", True),
            preserve_words=step.config.get("preserve_words", True),
            paragraph_separator=step.config.get("paragraph_separator", "\n\n"),
            sentence_pattern=step.config.get("sentence_pattern", r'[.!?]+\s+')
        )
        
        # Update service config
        chunking_service.config = config
        
        # Chunk the text
        chunks = chunking_service.chunk_text(text)
        
        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "total_length": len(text)
        }
    
    def _handle_condition(self, step: StepConfig, inputs: Dict[str, Any], 
                         context: ExecutionContext) -> Any:
        """Handle conditional step - evaluates condition and returns routing decision"""
        condition_type = step.config.get("condition_type", "llm_decision")
        
        if condition_type == "llm_decision":
            return self._handle_llm_condition(step, inputs, context)
        elif condition_type == "simple_comparison":
            return self._handle_simple_condition(step, inputs, context)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
    
    def _handle_llm_condition(self, step: StepConfig, inputs: Dict[str, Any], 
                             context: ExecutionContext) -> Any:
        """Handle LLM-based conditional evaluation"""
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")
        
        # Get the condition prompt
        condition_prompt = step.config.get("condition_prompt")
        if not condition_prompt:
            raise ValueError("condition_prompt required for LLM conditional step")
        
        # Get route options
        route_options = step.config.get("route_options", ["true", "false"])
        
        # Create a decision schema with the available routes
        decision_schema = {
            "name": "ConditionalDecision",
            "fields": {
                "decision": {
                    "type": "string", 
                    "description": f"Choose one of: {', '.join(route_options)}"
                },
                "reasoning": {
                    "type": "string", 
                    "description": "Explain why this route was chosen"
                },
                "confidence": {
                    "type": "float", 
                    "description": "Confidence in the decision (0.0 to 1.0)"
                }
            }
        }
        
        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        
        # Create structured client
        structured_client = llm_factory.create_structured_client(
            provider=provider,
            schema=self._create_schema_from_dict(decision_schema),
            model=model,
            temperature=step.config.get("temperature", 0.3)  # Lower temp for more consistent decisions
        )
        
        # Build the full prompt with context
        full_prompt = self._build_condition_prompt(condition_prompt, inputs, context)
        
        try:
            # Get structured decision
            decision_response = structured_client.chat(full_prompt)
            
            # Validate the decision is one of the allowed options
            chosen_route = decision_response.decision
            if chosen_route not in route_options:
                # Fallback to first option if invalid
                chosen_route = route_options[0]
            
            return {
                "condition_result": chosen_route,
                "reasoning": decision_response.reasoning,
                "confidence": decision_response.confidence,
                "route_options": route_options,
                "prompt": full_prompt,
                "provider": provider,
                "model": model
            }
            
        except Exception as e:
            # Fallback to default route on error
            default_route = step.config.get("default_route", route_options[0])
            return {
                "condition_result": default_route,
                "reasoning": f"Error in condition evaluation: {str(e)}",
                "confidence": 0.0,
                "route_options": route_options,
                "error": str(e)
            }
    
    def _handle_simple_condition(self, step: StepConfig, inputs: Dict[str, Any], 
                                context: ExecutionContext) -> Any:
        """Handle simple comparison-based conditional evaluation"""
        # Get comparison parameters
        left_value = step.config.get("left_value")
        operator = step.config.get("operator", "==")
        right_value = step.config.get("right_value")
        
        if left_value is None or right_value is None:
            raise ValueError("left_value and right_value required for simple condition")
        
        # Resolve values if they're references
        left_value = self._resolve_condition_value(left_value, context, inputs)
        right_value = self._resolve_condition_value(right_value, context, inputs)
        
        # Evaluate condition
        result = False
        if operator == "==":
            result = left_value == right_value
        elif operator == "!=":
            result = left_value != right_value
        elif operator == ">":
            result = left_value > right_value
        elif operator == "<":
            result = left_value < right_value
        elif operator == ">=":
            result = left_value >= right_value
        elif operator == "<=":
            result = left_value <= right_value
        elif operator == "contains":
            result = str(right_value) in str(left_value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
        return {
            "condition_result": "true" if result else "false",
            "left_value": left_value,
            "operator": operator,
            "right_value": right_value,
            "reasoning": f"{left_value} {operator} {right_value} = {result}"
        }
    
    def _build_condition_prompt(self, base_prompt: str, inputs: Dict[str, Any], 
                               context: ExecutionContext) -> str:
        """Build a complete condition prompt with context"""
        prompt_parts = [base_prompt]
        
        # Add input context
        if inputs:
            prompt_parts.append("\nInput data:")
            for key, value in inputs.items():
                prompt_parts.append(f"- {key}: {value}")
        
        # Add relevant step outputs from context
        if context.step_outputs:
            prompt_parts.append("\nPrevious step results:")
            for key, value in context.step_outputs.items():
                # Only include recent/relevant outputs to avoid overwhelming the prompt
                if isinstance(value, (str, int, float, bool)):
                    prompt_parts.append(f"- {key}: {value}")
                elif isinstance(value, dict) and len(str(value)) < 200:
                    prompt_parts.append(f"- {key}: {value}")
        
        return "\n".join(prompt_parts)
    
    def _resolve_condition_value(self, value: Any, context: ExecutionContext, inputs: Dict[str, Any]) -> Any:
        """Resolve a condition value that might be a reference"""
        if isinstance(value, dict) and "from_step" in value:
            # This is a reference to another step's output
            step_id = value["from_step"]
            field = value["field"]
            
            # Look for the output in step_outputs
            full_key = f"{step_id}.{field}"
            if full_key in context.step_outputs:
                return context.step_outputs[full_key]
            elif step_id in context.step_outputs:
                # Try to get field from the step's result
                step_result = context.step_outputs[step_id]
                if isinstance(step_result, dict) and field in step_result:
                    return step_result[field]
                else:
                    return step_result
            else:
                raise ValueError(f"Could not resolve condition reference: {step_id}.{field}")
        elif isinstance(value, str) and value.startswith("$"):
            # Global variable reference
            var_name = value[1:]
            if var_name in context.global_variables:
                return context.global_variables[var_name]
            elif var_name in inputs:
                return inputs[var_name]
            else:
                raise ValueError(f"Global variable not found: {var_name}")
        else:
            # Direct value
            return value
    
    def _handle_file_output(self, step: StepConfig, inputs: Dict[str, Any], 
                           context: ExecutionContext) -> Any:
        """Handle file output step - writes data to a file"""
        import json
        import os
        from datetime import datetime
        from pathlib import Path
        
        # Get configuration
        file_path = step.config.get("file_path")
        format_type = step.config.get("format", "json")
        create_dirs = step.config.get("create_dirs", True)
        append_timestamp = step.config.get("append_timestamp", False)
        
        # Get content to write
        content = inputs.get("content")
        if content is None:
            # If no specific content field, use all inputs
            content = inputs
        
        # Handle file path
        if not file_path:
            # Generate default filename based on workflow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"workflow_output_{timestamp}.{format_type}"
            file_path = os.path.join("output", default_name)
        
        # Add timestamp to filename if requested
        if append_timestamp:
            file_path = Path(file_path)
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
            file_path = file_path.parent / f"{file_path.stem}{timestamp}{file_path.suffix}"
            file_path = str(file_path)
        
        # Create directories if needed
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Format content based on format type
        try:
            if format_type == "json":
                if isinstance(content, (dict, list)):
                    formatted_content = json.dumps(content, indent=2, ensure_ascii=False)
                else:
                    # Wrap non-dict/list content in a structure
                    formatted_content = json.dumps({"content": content}, indent=2, ensure_ascii=False)
            
            elif format_type == "text" or format_type == "txt":
                if isinstance(content, dict):
                    # Convert dict to readable text format
                    lines = []
                    for key, value in content.items():
                        if isinstance(value, (dict, list)):
                            lines.append(f"{key}:")
                            lines.append(f"  {json.dumps(value, indent=2)}")
                        else:
                            lines.append(f"{key}: {value}")
                    formatted_content = "\n".join(lines)
                elif isinstance(content, list):
                    formatted_content = "\n".join(str(item) for item in content)
                else:
                    formatted_content = str(content)
            
            elif format_type == "markdown" or format_type == "md":
                if isinstance(content, dict):
                    lines = ["# Workflow Output", ""]
                    for key, value in content.items():
                        lines.append(f"## {key.replace('_', ' ').title()}")
                        if isinstance(value, (dict, list)):
                            lines.append("```json")
                            lines.append(json.dumps(value, indent=2))
                            lines.append("```")
                        else:
                            lines.append(str(value))
                        lines.append("")
                    formatted_content = "\n".join(lines)
                else:
                    formatted_content = f"# Workflow Output\n\n{str(content)}"
            
            elif format_type == "csv":
                import csv
                import io
                
                if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                    # List of dictionaries - convert to CSV
                    output = io.StringIO()
                    if content:
                        writer = csv.DictWriter(output, fieldnames=content[0].keys())
                        writer.writeheader()
                        writer.writerows(content)
                    formatted_content = output.getvalue()
                else:
                    raise ValueError("CSV format requires a list of dictionaries")
            
            else:
                # Default to string representation
                formatted_content = str(content)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            
            return {
                "file_path": os.path.abspath(file_path),
                "format": format_type,
                "file_size": file_size,
                "success": True,
                "content_written": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "file_path": file_path,
                "format": format_type,
                "success": False,
                "error": str(e),
                "content_written": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_output(self, step: StepConfig, inputs: Dict[str, Any], 
                      context: ExecutionContext) -> Any:
        """Handle output step - formats and returns final output"""
        # If format is specified in config, format the output
        if "format" in step.config:
            format_type = step.config["format"]
            if format_type == "json":
                import json
                return json.dumps(inputs, indent=2)
            elif format_type == "text":
                if isinstance(inputs, dict):
                    return "\n".join(f"{k}: {v}" for k, v in inputs.items())
                else:
                    return str(inputs)
        
        return inputs