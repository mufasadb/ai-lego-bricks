"""
Core agent orchestration classes for executing workflows
"""

import json
import time
from typing import Any, Dict, Optional

from .models import WorkflowConfig, ExecutionContext, WorkflowResult, StepConfig
from .step_handlers import StepHandlerRegistry

# Import existing services
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memory.memory_service import create_memory_service
except ImportError:
    create_memory_service = None

try:
    from llm.llm_factory import LLMClientFactory
except ImportError:
    LLMClientFactory = None

try:
    from pdf_to_text.pdf_to_text_service import PDFToTextService
except ImportError:
    PDFToTextService = None

try:
    from chunking.chunking_factory import ChunkingServiceFactory
except ImportError:
    ChunkingServiceFactory = None

try:
    from chat.conversation_service import ConversationService
except ImportError:
    ConversationService = None

try:
    from llm.generation_service import GenerationService
except ImportError:
    GenerationService = None

try:
    from prompt.prompt_service import create_prompt_service
except ImportError:
    create_prompt_service = None

try:
    from tts.tts_factory import create_tts_service
except ImportError:
    create_tts_service = None


class AgentOrchestrator:
    """
    Main orchestrator class that manages all the building block services
    and provides a unified interface for creating and executing workflows.
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize the orchestrator with optional global configuration"""
        self.config = config
        self._services = {}
        self._initialize_services()
        self.step_registry = StepHandlerRegistry(self)
    
    def _initialize_services(self):
        """Initialize all the available services"""
        # Memory service
        if create_memory_service:
            try:
                memory_backend = "auto"
                if self.config and self.config.config:
                    memory_backend = self.config.config.memory_backend
                self._services["memory"] = create_memory_service(memory_backend)
            except Exception as e:
                print(f"Warning: Memory service not available: {e}")
                self._services["memory"] = None
        else:
            self._services["memory"] = None
        
        # LLM Factory
        if LLMClientFactory:
            try:
                self._services["llm_factory"] = LLMClientFactory()
            except Exception as e:
                print(f"Warning: LLM factory not available: {e}")
                self._services["llm_factory"] = None
        else:
            self._services["llm_factory"] = None
        
        # PDF processing
        if PDFToTextService:
            try:
                self._services["pdf_processor"] = PDFToTextService()
            except Exception as e:
                print(f"Warning: PDF processor not available: {e}")
                self._services["pdf_processor"] = None
        else:
            self._services["pdf_processor"] = None
        
        # Chunking service factory
        if ChunkingServiceFactory:
            try:
                self._services["chunking_factory"] = ChunkingServiceFactory()
            except Exception as e:
                print(f"Warning: Chunking service factory not available: {e}")
                self._services["chunking_factory"] = None
        else:
            self._services["chunking_factory"] = None
        
        # Conversation service (for multi-turn conversations)
        if ConversationService:
            try:
                default_provider = "gemini"
                if self.config and self.config.config:
                    default_provider = self.config.config.default_llm_provider
                # Import LLMProvider for proper conversion
                from llm.llm_types import LLMProvider
                provider_enum = LLMProvider(default_provider)
                self._services["conversation"] = ConversationService(provider_enum)
            except Exception as e:
                print(f"Warning: Conversation service not available: {e}")
                self._services["conversation"] = None
        else:
            self._services["conversation"] = None
        
        # Generation service (for one-shot LLM calls)
        if GenerationService:
            try:
                default_provider = "gemini"
                if self.config and self.config.config:
                    default_provider = self.config.config.default_llm_provider
                # Import LLMProvider for proper conversion
                from llm.llm_types import LLMProvider
                provider_enum = LLMProvider(default_provider)
                self._services["generation"] = GenerationService(provider_enum)
            except Exception as e:
                print(f"Warning: Generation service not available: {e}")
                self._services["generation"] = None
        else:
            self._services["generation"] = None
        
        # Prompt service
        if create_prompt_service:
            try:
                # Use same backend as memory service if available, otherwise auto-detect
                prompt_backend = "auto"
                if self.config and self.config.config:
                    prompt_backend = self.config.config.memory_backend
                self._services["prompt"] = create_prompt_service(prompt_backend)
            except Exception as e:
                print(f"Warning: Prompt service not available: {e}")
                self._services["prompt"] = None
        else:
            self._services["prompt"] = None
        
        # TTS service
        if create_tts_service:
            try:
                # Auto-detect available TTS provider
                tts_provider = "auto"
                if self.config and self.config.config:
                    # Check if TTS provider is specified in config
                    tts_provider = getattr(self.config.config, 'default_tts_provider', "auto")
                self._services["tts"] = create_tts_service(tts_provider)
            except Exception as e:
                print(f"Warning: TTS service not available: {e}")
                self._services["tts"] = None
        else:
            self._services["tts"] = None
    
    def get_service(self, service_name: str):
        """Get a service by name"""
        return self._services.get(service_name)
    
    def load_workflow_from_file(self, file_path: str) -> WorkflowConfig:
        """Load a workflow configuration from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return WorkflowConfig(**data)
    
    def load_workflow_from_dict(self, config_dict: Dict[str, Any]) -> WorkflowConfig:
        """Load a workflow configuration from a dictionary"""
        return WorkflowConfig(**config_dict)
    
    def execute_workflow(self, workflow: WorkflowConfig, 
                        initial_inputs: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute a complete workflow"""
        executor = WorkflowExecutor(self)
        return executor.execute(workflow, initial_inputs or {})


class WorkflowExecutor:
    """
    Handles the execution of individual workflow steps and manages
    the execution context and data flow between steps.
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.context = ExecutionContext()
    
    def execute(self, workflow: WorkflowConfig, 
                initial_inputs: Dict[str, Any]) -> WorkflowResult:
        """Execute a complete workflow with conditional branching support"""
        start_time = time.time()
        
        try:
            # Initialize context with initial inputs
            self.context.global_variables.update(initial_inputs)
            
            # Create a mapping of step IDs to steps for easy lookup
            step_map = {step.id: step for step in workflow.steps}
            
            # Execute workflow with conditional branching
            current_step_index = 0
            default_max_iterations = workflow.config.max_iterations
            
            while current_step_index < len(workflow.steps):
                step = workflow.steps[current_step_index]
                
                # Increment iteration count for this step
                iteration_count = self.context.increment_step_iteration(step.id)
                
                # Check for infinite loop protection with configurable max iterations
                max_iterations = step.max_iterations if step.max_iterations is not None else default_max_iterations
                if iteration_count > max_iterations:
                    raise RuntimeError(f"Maximum iterations ({max_iterations}) exceeded for step: {step.id}")
                
                self.context.current_step_index = current_step_index
                
                # Execute the step
                result = self._execute_step(step)
                
                # Store step output
                if step.outputs:
                    for output_name in step.outputs:
                        if output_name in result:
                            self.context.step_outputs[f"{step.id}.{output_name}"] = result[output_name]
                        else:
                            # If specific output not found, store the entire result
                            self.context.step_outputs[f"{step.id}.{output_name}"] = result
                else:
                    # Store entire result under step id
                    self.context.step_outputs[step.id] = result
                
                # Store iteration history if configured
                if step.preserve_previous_results:
                    self.context.add_step_iteration_result(step.id, result)
                
                # Handle conditional routing
                next_step_id = self._get_next_step(step, result, workflow)
                
                if next_step_id:
                    # Find the next step by ID
                    next_step_index = None
                    for i, s in enumerate(workflow.steps):
                        if s.id == next_step_id:
                            next_step_index = i
                            break
                    
                    if next_step_index is not None:
                        current_step_index = next_step_index
                    else:
                        # If target step not found, end execution
                        break
                else:
                    # No conditional routing, proceed to next step
                    current_step_index += 1
            
            # Get final output (from last step or specific output step)
            final_output = self._get_final_output(workflow)
            
            execution_time = time.time() - start_time
            
            return WorkflowResult(
                success=True,
                final_output=final_output,
                step_outputs=dict(self.context.step_outputs),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowResult(
                success=False,
                error=str(e),
                step_outputs=dict(self.context.step_outputs),
                execution_time=execution_time
            )
    
    def _execute_step(self, step: StepConfig) -> Any:
        """Execute a single workflow step"""
        # Track current step for iteration context resolution
        self._current_step_id = step.id
        
        # Resolve inputs
        resolved_inputs = self._resolve_inputs(step.inputs)
        
        # Get the appropriate handler for this step type
        handler = self.orchestrator.step_registry.get_handler(step.type)
        if not handler:
            raise ValueError(f"No handler found for step type: {step.type}")
        
        # Execute the step
        result = handler(step, resolved_inputs, self.context)
        
        # Clear current step tracking
        self._current_step_id = None
        
        return result
    
    def _resolve_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input references to actual values"""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, dict) and "from_step" in value:
                # This is a reference to another step's output
                step_id = value["from_step"]
                field = value["field"]
                
                # Look for the output in step_outputs
                full_key = f"{step_id}.{field}"
                if full_key in self.context.step_outputs:
                    resolved[key] = self.context.step_outputs[full_key]
                elif step_id in self.context.step_outputs:
                    # Try to get field from the step's result
                    step_result = self.context.step_outputs[step_id]
                    if isinstance(step_result, dict) and field in step_result:
                        resolved[key] = step_result[field]
                    else:
                        resolved[key] = step_result
                else:
                    raise ValueError(f"Could not resolve input reference: {step_id}.{field}")
            else:
                # Direct value or variable reference
                resolved[key] = self._resolve_variable_reference(value)
        
        return resolved
    
    def _resolve_variable_reference(self, value: Any) -> Any:
        """Resolve variable references including iteration context"""
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            
            # Check for iteration context variables
            if var_name.startswith("iteration_context."):
                return self._resolve_iteration_context(var_name[18:])  # Remove "iteration_context." prefix
            
            # Check global variables
            if var_name in self.context.global_variables:
                return self.context.global_variables[var_name]
            else:
                raise ValueError(f"Variable not found: {var_name}")
        else:
            return value
    
    def _resolve_iteration_context(self, context_key: str) -> Any:
        """Resolve iteration context variables"""
        current_step_id = None
        if hasattr(self, '_current_step_id'):
            current_step_id = self._current_step_id
        
        if context_key == "iteration_count":
            return self.context.get_step_iteration_count(current_step_id) if current_step_id else 0
        elif context_key == "previous_result":
            return self.context.get_previous_step_result(current_step_id) if current_step_id else None
        elif context_key == "iteration_history":
            return self.context.get_step_iteration_history(current_step_id) if current_step_id else []
        elif context_key.startswith("previous_result."):
            # Access specific field from previous result
            field_name = context_key[16:]  # Remove "previous_result." prefix
            prev_result = self.context.get_previous_step_result(current_step_id) if current_step_id else None
            if isinstance(prev_result, dict) and field_name in prev_result:
                return prev_result[field_name]
            return None
        else:
            raise ValueError(f"Unknown iteration context key: {context_key}")
    
    def _get_next_step(self, current_step: StepConfig, step_result: Any, 
                      workflow: WorkflowConfig) -> Optional[str]:
        """Determine the next step to execute based on conditional routing"""
        # Check if this step has conditional routing
        if current_step.routes and isinstance(step_result, dict):
            # For condition steps, use the condition_result
            if current_step.type == "condition" and "condition_result" in step_result:
                condition_result = step_result["condition_result"]
                next_step_id = current_step.routes.get(condition_result)
                if next_step_id:
                    return next_step_id
            
            # For other steps, check if they have routing based on any output field
            for field_name, field_value in step_result.items():
                if str(field_value) in current_step.routes:
                    return current_step.routes[str(field_value)]
        
        # Check for special routing keywords
        if current_step.routes:
            # Check for "default" route
            if "default" in current_step.routes:
                return current_step.routes["default"]
            
            # Check for "end" or "exit" to terminate workflow
            if isinstance(step_result, dict):
                for field_value in step_result.values():
                    if str(field_value) in ["end", "exit", "terminate"]:
                        return None  # End workflow
        
        return None  # Continue to next step in sequence
    
    def _get_final_output(self, workflow: WorkflowConfig) -> Any:
        """Get the final output from the workflow"""
        # Look for a step with type "output"
        for step in workflow.steps:
            if step.type == "output":
                return self.context.step_outputs.get(step.id)
        
        # If no output step, return the last step's output
        if workflow.steps:
            last_step = workflow.steps[-1]
            return self.context.step_outputs.get(last_step.id)
        
        return None