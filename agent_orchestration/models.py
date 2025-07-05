"""
Pydantic models for agent orchestration configuration
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class StepType(str, Enum):
    """Available step types for workflow execution"""
    INPUT = "input"
    DOCUMENT_PROCESSING = "document_processing"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    LLM_CHAT = "llm_chat"
    LLM_STRUCTURED = "llm_structured"
    LLM_VISION = "llm_vision"
    CHUNK_TEXT = "chunk_text"
    CONDITION = "condition"
    LOOP = "loop"
    OUTPUT = "output"
    FILE_OUTPUT = "file_output"
    HUMAN_APPROVAL = "human_approval"
    # Conversation management step types
    START_CONVERSATION = "start_conversation"
    ADD_TO_CONVERSATION = "add_to_conversation"
    CONTINUE_CONVERSATION = "continue_conversation"


class InputReference(BaseModel):
    """Reference to output from another step"""
    from_step: str
    field: str


class ConversationMessage(BaseModel):
    """A single message in a conversation thread"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ConversationThread(BaseModel):
    """A conversation thread containing multiple messages"""
    id: str
    name: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """Add a message to the conversation thread"""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message
    
    def get_messages_for_llm(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption"""
        formatted_messages = []
        for msg in self.messages:
            if not include_system and msg.role == 'system':
                continue
            formatted_messages.append({
                'role': msg.role,
                'content': msg.content
            })
        return formatted_messages
    
    def get_last_messages(self, count: int, include_system: bool = True) -> List[ConversationMessage]:
        """Get the last N messages from the conversation"""
        messages = self.messages
        if not include_system:
            messages = [msg for msg in messages if msg.role != 'system']
        return messages[-count:] if count > 0 else messages


class ConditionalConfig(BaseModel):
    """Configuration for conditional step execution"""
    if_condition: str = Field(alias="if")
    then_step: Optional["StepConfig"] = None
    else_step: Optional["StepConfig"] = None


class LoopConfig(BaseModel):
    """Configuration for loop execution"""
    over: str  # Variable or step output to iterate over
    body: "StepConfig"


class StepConfig(BaseModel):
    """Configuration for a single workflow step"""
    id: str
    type: StepType
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)  # Allow any type for inputs
    outputs: List[str] = Field(default_factory=list)
    condition: Optional[ConditionalConfig] = None
    loop: Optional[LoopConfig] = None
    routes: Optional[Dict[str, str]] = Field(default_factory=dict)  # Maps condition results to next step IDs
    max_iterations: Optional[int] = None  # Maximum number of times this step can be executed
    preserve_previous_results: bool = False  # Whether to preserve results from previous iterations


class WorkflowGlobalConfig(BaseModel):
    """Global configuration for the entire workflow"""
    memory_backend: str = "auto"
    default_llm_provider: str = "gemini"
    default_model: Optional[str] = None
    max_iterations: int = 10  # Default maximum iterations per step for loop-back protection


class WorkflowConfig(BaseModel):
    """Complete workflow configuration"""
    name: str
    description: str
    config: WorkflowGlobalConfig = Field(default_factory=WorkflowGlobalConfig)
    steps: List[StepConfig]


class ExecutionContext(BaseModel):
    """Runtime context for workflow execution"""
    step_outputs: Dict[str, Any] = Field(default_factory=dict)
    global_variables: Dict[str, Any] = Field(default_factory=dict)
    current_step_index: int = 0
    # Conversation management
    conversations: Dict[str, ConversationThread] = Field(default_factory=dict)
    active_conversation_id: Optional[str] = None
    # Iteration tracking for loop-back support
    step_iteration_counts: Dict[str, int] = Field(default_factory=dict)  # step_id -> iteration count
    step_iteration_history: Dict[str, List[Any]] = Field(default_factory=dict)  # step_id -> list of previous results
    
    def get_active_conversation(self) -> Optional[ConversationThread]:
        """Get the currently active conversation thread"""
        if self.active_conversation_id and self.active_conversation_id in self.conversations:
            return self.conversations[self.active_conversation_id]
        return None
    
    def create_conversation(self, conversation_id: str, name: Optional[str] = None) -> ConversationThread:
        """Create a new conversation thread"""
        conversation = ConversationThread(id=conversation_id, name=name)
        self.conversations[conversation_id] = conversation
        return conversation
    
    def set_active_conversation(self, conversation_id: str) -> bool:
        """Set the active conversation thread"""
        if conversation_id in self.conversations:
            self.active_conversation_id = conversation_id
            return True
        return False
    
    def increment_step_iteration(self, step_id: str) -> int:
        """Increment and return the iteration count for a step"""
        current_count = self.step_iteration_counts.get(step_id, 0)
        new_count = current_count + 1
        self.step_iteration_counts[step_id] = new_count
        return new_count
    
    def get_step_iteration_count(self, step_id: str) -> int:
        """Get the current iteration count for a step"""
        return self.step_iteration_counts.get(step_id, 0)
    
    def add_step_iteration_result(self, step_id: str, result: Any) -> None:
        """Add a result to the iteration history for a step"""
        if step_id not in self.step_iteration_history:
            self.step_iteration_history[step_id] = []
        self.step_iteration_history[step_id].append(result)
    
    def get_step_iteration_history(self, step_id: str) -> List[Any]:
        """Get the iteration history for a step"""
        return self.step_iteration_history.get(step_id, [])
    
    def get_previous_step_result(self, step_id: str) -> Optional[Any]:
        """Get the most recent previous result for a step"""
        history = self.get_step_iteration_history(step_id)
        return history[-1] if history else None


class WorkflowResult(BaseModel):
    """Result of workflow execution"""
    success: bool
    final_output: Any = None
    step_outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None


# Update forward references
StepConfig.model_rebuild()
ConditionalConfig.model_rebuild()
LoopConfig.model_rebuild()