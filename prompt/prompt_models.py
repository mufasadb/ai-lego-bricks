"""
Pydantic models for prompt management
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import semantic_version


class PromptRole(str, Enum):
    """Available prompt roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class PromptStatus(str, Enum):
    """Prompt lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PromptMetadata(BaseModel):
    """Metadata for prompt management"""
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    use_case: Optional[str] = None
    model_recommendations: List[str] = Field(default_factory=list)
    language: str = "en"
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class PromptTemplate(BaseModel):
    """Template configuration for prompt variable substitution"""
    template: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    required_variables: List[str] = Field(default_factory=list)
    template_engine: str = "jinja2"  # Support for different engines
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with provided context"""
        import jinja2
        
        # Check required variables
        missing_vars = [var for var in self.required_variables if var not in context]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Merge default variables with context
        render_context = {**self.variables, **context}
        
        if self.template_engine == "jinja2":
            template = jinja2.Template(self.template)
            return template.render(**render_context)
        else:
            raise ValueError(f"Unsupported template engine: {self.template_engine}")


class PromptContent(BaseModel):
    """Content of a prompt with role and template support"""
    role: PromptRole
    content: Union[str, PromptTemplate]
    
    def render(self, context: Dict[str, Any] = None) -> str:
        """Render content with optional context"""
        if isinstance(self.content, PromptTemplate):
            return self.content.render(context or {})
        return self.content


class PromptVersion(BaseModel):
    """Semantic version information for prompts"""
    version: str
    changelog: Optional[str] = None
    compatibility_notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate semantic version format"""
        try:
            semantic_version.Version(self.version)
        except ValueError as e:
            raise ValueError(f"Invalid semantic version: {self.version}") from e
    
    @property
    def parsed_version(self) -> semantic_version.Version:
        """Get parsed semantic version object"""
        return semantic_version.Version(self.version)


class Prompt(BaseModel):
    """Complete prompt definition with versioning and metadata"""
    id: str
    name: str
    content: List[PromptContent]  # Support multi-role prompts
    version: PromptVersion
    status: PromptStatus = PromptStatus.DRAFT
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    parent_id: Optional[str] = None  # For version chains
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    def render(self, context: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Render all content parts with context"""
        return [
            {
                "role": content.role.value,
                "content": content.render(context)
            }
            for content in self.content
        ]
    
    def get_content_by_role(self, role: PromptRole) -> Optional[PromptContent]:
        """Get content by role"""
        for content in self.content:
            if content.role == role:
                return content
        return None
    
    def get_system_message(self, context: Dict[str, Any] = None) -> Optional[str]:
        """Get rendered system message"""
        system_content = self.get_content_by_role(PromptRole.SYSTEM)
        return system_content.render(context) if system_content else None
    
    def get_user_message(self, context: Dict[str, Any] = None) -> Optional[str]:
        """Get rendered user message"""
        user_content = self.get_content_by_role(PromptRole.USER)
        return user_content.render(context) if user_content else None


class PromptExecution(BaseModel):
    """Record of prompt execution for evaluation and training"""
    id: str
    prompt_id: str
    prompt_version: str
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    rendered_messages: List[Dict[str, str]]
    llm_provider: str
    llm_model: Optional[str] = None
    llm_response: Optional[str] = None
    execution_time_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class PromptEvaluation(BaseModel):
    """Evaluation metrics for prompt performance"""
    prompt_id: str
    prompt_version: str
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    total_executions: int
    success_rate: float
    average_response_time_ms: float
    average_tokens_used: Optional[Dict[str, float]] = None
    error_rate: float
    common_errors: List[str] = Field(default_factory=list)
    performance_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class PromptComparisonResult(BaseModel):
    """Result of A/B testing between prompt versions"""
    test_id: str
    prompt_a_id: str
    prompt_a_version: str
    prompt_b_id: str
    prompt_b_version: str
    test_description: Optional[str] = None
    sample_size: int
    winner: Optional[str] = None  # "prompt_a", "prompt_b", or "no_significant_difference"
    confidence_level: Optional[float] = None
    metrics_comparison: Dict[str, Any] = Field(default_factory=dict)
    test_duration_hours: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }