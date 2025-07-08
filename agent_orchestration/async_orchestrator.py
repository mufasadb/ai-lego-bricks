"""
Async agent orchestration classes for parallel workflow execution
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass

from .models import WorkflowConfig, ExecutionContext, WorkflowResult, StepConfig, StepType
from .orchestrator import AgentOrchestrator

# Import existing services
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParallelizationMode(Enum):
    """Parallelization execution modes"""
    DISABLED = "disabled"           # No parallelization
    SELECTIVE = "selective"         # Parallel where possible, sequential for streaming
    AGGRESSIVE = "aggressive"       # Maximum parallelization (may break streaming)


class StreamingMode(Enum):
    """Streaming operation modes"""
    INACTIVE = "inactive"           # No streaming operations
    ACTIVE = "active"               # Streaming operations present
    PARTIAL = "partial"             # Some streaming operations


@dataclass
class StreamingAnalysis:
    """Analysis of streaming operations in workflow"""
    has_streaming: bool
    streaming_steps: List[str]
    dependent_steps: List[str]
    parallelizable_steps: List[str]
    streaming_mode: StreamingMode


@dataclass
class ParallelGroup:
    """Group of steps that can execute in parallel"""
    steps: List[str]
    dependencies: Set[str]
    can_parallelize: bool
    resource_requirements: Dict[str, int]


class StreamingDetector:
    """Detects streaming operations and analyzes constraints"""
    
    @staticmethod
    def detect_streaming_operations(workflow: WorkflowConfig) -> StreamingAnalysis:
        """Analyze workflow for streaming constraints"""
        streaming_steps = []
        dependent_steps = []
        
        for step in workflow.steps:
            if StreamingDetector._is_streaming_step(step):
                streaming_steps.append(step.id)
                # Find all dependent steps
                dependent_steps.extend(
                    StreamingDetector._find_dependent_steps(step.id, workflow)
                )
        
        # Remove duplicates
        dependent_steps = list(set(dependent_steps))
        
        # Find parallelizable steps
        parallelizable_steps = []
        for step in workflow.steps:
            if step.id not in streaming_steps and step.id not in dependent_steps:
                if StreamingDetector._can_parallelize_step(step):
                    parallelizable_steps.append(step.id)
        
        # Determine streaming mode
        if not streaming_steps:
            streaming_mode = StreamingMode.INACTIVE
        elif len(streaming_steps) >= len(workflow.steps) / 2:
            streaming_mode = StreamingMode.ACTIVE
        else:
            streaming_mode = StreamingMode.PARTIAL
        
        return StreamingAnalysis(
            has_streaming=bool(streaming_steps),
            streaming_steps=streaming_steps,
            dependent_steps=dependent_steps,
            parallelizable_steps=parallelizable_steps,
            streaming_mode=streaming_mode
        )
    
    @staticmethod
    def _is_streaming_step(step: StepConfig) -> bool:
        """Check if step uses streaming"""
        return (
            step.type in [StepType.LLM_CHAT, StepType.TTS] and
            step.config.get("stream", False)
        )
    
    @staticmethod
    def _find_dependent_steps(step_id: str, workflow: WorkflowConfig) -> List[str]:
        """Find all steps that depend on the given step"""
        dependent_steps = []
        
        for step in workflow.steps:
            for input_config in step.inputs.values():
                if isinstance(input_config, dict) and input_config.get("from_step") == step_id:
                    dependent_steps.append(step.id)
                    # Recursively find steps that depend on this dependent step
                    dependent_steps.extend(
                        StreamingDetector._find_dependent_steps(step.id, workflow)
                    )
        
        return dependent_steps
    
    @staticmethod
    def _can_parallelize_step(step: StepConfig) -> bool:
        """Check if step can be executed in parallel"""
        # Memory retrieve operations are safe for parallel execution
        if step.type == StepType.MEMORY_RETRIEVE:
            return True
        
        # Document processing can be parallelized if not streaming
        if step.type == StepType.DOCUMENT_PROCESSING:
            return True
        
        # Chunking operations can be parallelized
        if step.type == StepType.CHUNK_TEXT:
            return True
        
        # Non-streaming LLM operations can be parallelized
        if step.type == StepType.LLM_CHAT and not step.config.get("stream", False):
            return True
        
        if step.type == StepType.LLM_STRUCTURED:
            return True
        
        if step.type == StepType.LLM_VISION:
            return True
        
        # Input and output steps are generally safe
        if step.type in [StepType.INPUT, StepType.OUTPUT]:
            return True
        
        return False


class WorkflowDependencyGraph:
    """Builds and analyzes workflow dependency graphs"""
    
    def __init__(self, workflow: WorkflowConfig):
        self.workflow = workflow
        self.dependency_graph = self._build_dependency_graph()
        self.parallel_groups = self._identify_parallel_groups()
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build dependency graph from step inputs/outputs"""
        dependencies = {}
        
        for step in self.workflow.steps:
            step_deps = set()
            
            # Analyze inputs for dependencies
            for input_config in step.inputs.values():
                if isinstance(input_config, dict) and "from_step" in input_config:
                    step_deps.add(input_config["from_step"])
            
            dependencies[step.id] = step_deps
        
        return dependencies
    
    def _identify_parallel_groups(self) -> List[ParallelGroup]:
        """Group steps that can execute in parallel"""
        parallel_groups = []
        processed_steps = set()
        
        # Process steps in dependency order
        for step in self.workflow.steps:
            if step.id in processed_steps:
                continue
            
            # Find all steps that can run in parallel with this step
            parallel_candidates = [step.id]
            step_deps = self.dependency_graph[step.id]
            
            # Look for other steps with same dependencies
            for other_step in self.workflow.steps:
                if (other_step.id != step.id and 
                    other_step.id not in processed_steps and
                    self.dependency_graph[other_step.id] == step_deps):
                    parallel_candidates.append(other_step.id)
            
            # Check if group can be parallelized
            can_parallelize = all(
                StreamingDetector._can_parallelize_step(
                    next(s for s in self.workflow.steps if s.id == step_id)
                )
                for step_id in parallel_candidates
            )
            
            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(parallel_candidates)
            
            parallel_group = ParallelGroup(
                steps=parallel_candidates,
                dependencies=step_deps,
                can_parallelize=can_parallelize,
                resource_requirements=resource_requirements
            )
            
            parallel_groups.append(parallel_group)
            processed_steps.update(parallel_candidates)
        
        return parallel_groups
    
    def _calculate_resource_requirements(self, step_ids: List[str]) -> Dict[str, int]:
        """Calculate resource requirements for parallel execution"""
        requirements = {"llm": 0, "tts": 0, "memory": 0, "document": 0}
        
        for step_id in step_ids:
            step = next(s for s in self.workflow.steps if s.id == step_id)
            
            if step.type in [StepType.LLM_CHAT, StepType.LLM_STRUCTURED, StepType.LLM_VISION]:
                requirements["llm"] += 1
            elif step.type == StepType.TTS:
                requirements["tts"] += 1
            elif step.type in [StepType.MEMORY_STORE, StepType.MEMORY_RETRIEVE]:
                requirements["memory"] += 1
            elif step.type == StepType.DOCUMENT_PROCESSING:
                requirements["document"] += 1
        
        return requirements


class ResourceManager:
    """Manages concurrent resource access for parallel execution"""
    
    def __init__(self, max_concurrent_llm: int = 3, max_concurrent_tts: int = 2,
                 max_concurrent_memory: int = 5, max_concurrent_document: int = 3):
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.tts_semaphore = asyncio.Semaphore(max_concurrent_tts)
        self.memory_semaphore = asyncio.Semaphore(max_concurrent_memory)
        self.document_semaphore = asyncio.Semaphore(max_concurrent_document)
        self.global_lock = asyncio.Lock()
    
    async def acquire_resources(self, requirements: Dict[str, int]) -> List[asyncio.Semaphore]:
        """Acquire required resources for parallel execution"""
        acquired_semaphores = []
        
        try:
            # Acquire resources in order to prevent deadlocks
            if requirements.get("llm", 0) > 0:
                for _ in range(requirements["llm"]):
                    await self.llm_semaphore.acquire()
                    acquired_semaphores.append(self.llm_semaphore)
            
            if requirements.get("tts", 0) > 0:
                for _ in range(requirements["tts"]):
                    await self.tts_semaphore.acquire()
                    acquired_semaphores.append(self.tts_semaphore)
            
            if requirements.get("memory", 0) > 0:
                for _ in range(requirements["memory"]):
                    await self.memory_semaphore.acquire()
                    acquired_semaphores.append(self.memory_semaphore)
            
            if requirements.get("document", 0) > 0:
                for _ in range(requirements["document"]):
                    await self.document_semaphore.acquire()
                    acquired_semaphores.append(self.document_semaphore)
            
            return acquired_semaphores
        
        except Exception:
            # Release any acquired resources on failure
            for semaphore in acquired_semaphores:
                semaphore.release()
            raise
    
    def release_resources(self, semaphores: List[asyncio.Semaphore]):
        """Release acquired resources"""
        for semaphore in semaphores:
            semaphore.release()


class IsolatedExecutionContext:
    """Isolated execution context for parallel operations"""
    
    def __init__(self, parent_context: ExecutionContext):
        self.parent = parent_context
        self.local_outputs = {}
        self.local_variables = {}
        self.lock = asyncio.Lock()
    
    async def get_output(self, key: str) -> Any:
        """Get output with parent fallback"""
        async with self.lock:
            if key in self.local_outputs:
                return self.local_outputs[key]
            return self.parent.step_outputs.get(key)
    
    async def set_output(self, key: str, value: Any):
        """Set output in local context"""
        async with self.lock:
            self.local_outputs[key] = value
    
    async def merge_to_parent(self):
        """Merge local context back to parent"""
        async with self.lock:
            # This would need proper synchronization with parent in real implementation
            self.parent.step_outputs.update(self.local_outputs)
            self.parent.global_variables.update(self.local_variables)


class AsyncWorkflowExecutor:
    """
    Async workflow executor with parallel step execution capabilities
    """
    
    def __init__(self, orchestrator: AgentOrchestrator, 
                 parallelization_mode: ParallelizationMode = ParallelizationMode.SELECTIVE):
        self.orchestrator = orchestrator
        self.parallelization_mode = parallelization_mode
        self.context = ExecutionContext()
        self.resource_manager = ResourceManager()
    
    async def execute(self, workflow: WorkflowConfig, 
                     initial_inputs: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute workflow with selective parallelization"""
        start_time = time.time()
        
        try:
            # Initialize context with initial inputs
            self.context.global_variables.update(initial_inputs or {})
            self.context.global_config = workflow.config
            
            # Analyze streaming operations
            streaming_analysis = StreamingDetector.detect_streaming_operations(workflow)
            
            # Choose execution strategy based on streaming analysis and mode
            if (self.parallelization_mode == ParallelizationMode.DISABLED or
                streaming_analysis.streaming_mode == StreamingMode.ACTIVE):
                # Fall back to sequential execution
                return await self._execute_sequential(workflow, initial_inputs or {})
            else:
                # Use parallel execution
                return await self._execute_parallel(workflow, initial_inputs or {}, streaming_analysis)
        
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowResult(
                success=False,
                error=str(e),
                step_outputs=dict(self.context.step_outputs),
                execution_time=execution_time
            )
    
    async def _execute_sequential(self, workflow: WorkflowConfig, 
                                 initial_inputs: Dict[str, Any]) -> WorkflowResult:
        """Execute workflow sequentially (fallback for streaming)"""
        # Import synchronous executor to handle this
        from .orchestrator import WorkflowExecutor
        
        sync_executor = WorkflowExecutor(self.orchestrator)
        return sync_executor.execute(workflow, initial_inputs)
    
    async def _execute_parallel(self, workflow: WorkflowConfig, 
                               initial_inputs: Dict[str, Any],
                               streaming_analysis: StreamingAnalysis) -> WorkflowResult:
        """Execute workflow with parallel step execution"""
        start_time = time.time()
        dependency_graph = WorkflowDependencyGraph(workflow)
        
        # Execute parallel groups in dependency order
        for parallel_group in dependency_graph.parallel_groups:
            if parallel_group.can_parallelize and len(parallel_group.steps) > 1:
                # Execute steps in parallel
                await self._execute_parallel_group(parallel_group, workflow)
            else:
                # Execute steps sequentially
                await self._execute_sequential_group(parallel_group, workflow)
        
        # Get final output
        final_output = self._get_final_output(workflow)
        
        execution_time = time.time() - start_time
        
        return WorkflowResult(
            success=True,
            final_output=final_output,
            step_outputs=dict(self.context.step_outputs),
            execution_time=execution_time
        )
    
    async def _execute_parallel_group(self, parallel_group: ParallelGroup, 
                                     workflow: WorkflowConfig):
        """Execute a group of steps in parallel"""
        # Acquire resources for the group
        semaphores = await self.resource_manager.acquire_resources(
            parallel_group.resource_requirements
        )
        
        try:
            # Create tasks for each step
            tasks = []
            for step_id in parallel_group.steps:
                step = next(s for s in workflow.steps if s.id == step_id)
                task = asyncio.create_task(self._execute_step_async(step))
                tasks.append((step_id, task))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            for (step_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    raise result
                
                # Store step outputs
                step = next(s for s in workflow.steps if s.id == step_id)
                self._store_step_output(step, result)
        
        finally:
            # Release resources
            self.resource_manager.release_resources(semaphores)
    
    async def _execute_sequential_group(self, parallel_group: ParallelGroup, 
                                       workflow: WorkflowConfig):
        """Execute a group of steps sequentially"""
        for step_id in parallel_group.steps:
            step = next(s for s in workflow.steps if s.id == step_id)
            result = await self._execute_step_async(step)
            self._store_step_output(step, result)
    
    async def _execute_step_async(self, step: StepConfig) -> Any:
        """Execute a single step asynchronously"""
        # Resolve inputs
        resolved_inputs = await self._resolve_inputs_async(step.inputs)
        
        # For input steps, include global variables
        if step.type.value == "input":
            resolved_inputs.update(self.context.global_variables)
        
        # Get handler and execute
        handler = self.orchestrator.step_registry.get_handler(step.type)
        if not handler:
            raise ValueError(f"No handler found for step type: {step.type}")
        
        # Execute step (wrap synchronous handlers in async)
        # Run the synchronous handler in a thread pool to make it truly async
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, handler, step, resolved_inputs, self.context)
        
        return result
    
    async def _resolve_inputs_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input references to actual values asynchronously"""
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
                    # Handle missing dependency
                    raise ValueError(f"Missing dependency: {step_id}.{field}")
            else:
                resolved[key] = value
        
        return resolved
    
    def _store_step_output(self, step: StepConfig, result: Any):
        """Store step output in context"""
        if step.outputs:
            for output_name in step.outputs:
                if isinstance(result, dict) and output_name in result:
                    self.context.step_outputs[f"{step.id}.{output_name}"] = result[output_name]
                else:
                    self.context.step_outputs[f"{step.id}.{output_name}"] = result
        else:
            self.context.step_outputs[step.id] = result
    
    def _get_final_output(self, workflow: WorkflowConfig) -> Any:
        """Get final workflow output"""
        # Find the last step or output step
        output_steps = [step for step in workflow.steps if step.type == StepType.OUTPUT]
        
        if output_steps:
            last_output_step = output_steps[-1]
            return self.context.step_outputs.get(last_output_step.id)
        elif workflow.steps:
            last_step = workflow.steps[-1]
            return self.context.step_outputs.get(last_step.id)
        else:
            return None


class AsyncAgentOrchestrator(AgentOrchestrator):
    """
    Async version of AgentOrchestrator with parallel execution capabilities
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None,
                 parallelization_mode: ParallelizationMode = ParallelizationMode.SELECTIVE):
        super().__init__(config)
        self.parallelization_mode = parallelization_mode
    
    async def execute_workflow_async(self, workflow: WorkflowConfig, 
                                    initial_inputs: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute a workflow asynchronously with parallel execution"""
        executor = AsyncWorkflowExecutor(self, self.parallelization_mode)
        return await executor.execute(workflow, initial_inputs)
    
    def execute_workflow(self, workflow: WorkflowConfig, 
                        initial_inputs: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute a workflow (maintains backward compatibility)"""
        # For backward compatibility, run async version in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If loop is already running, fall back to sync execution
            return super().execute_workflow(workflow, initial_inputs)
        else:
            # Run async version
            return loop.run_until_complete(
                self.execute_workflow_async(workflow, initial_inputs)
            )