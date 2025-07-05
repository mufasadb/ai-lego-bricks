"""
Main prompt management service
"""

import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from .prompt_models import (
    Prompt, PromptContent, PromptTemplate, PromptVersion, PromptMetadata,
    PromptRole, PromptStatus, PromptExecution
)
from .prompt_storage import PromptStorageBackend, create_storage_backend
from .prompt_registry import PromptRegistry


class PromptService:
    """Main service for prompt management"""
    
    def __init__(self, storage_backend: Optional[PromptStorageBackend] = None,
                 cache_ttl: int = 3600):
        """
        Initialize prompt service
        
        Args:
            storage_backend: Storage backend to use. If None, auto-detects.
            cache_ttl: Cache TTL in seconds
        """
        if storage_backend is None:
            storage_backend = create_storage_backend("auto")
        
        self.storage = storage_backend
        self.registry = PromptRegistry(storage_backend, cache_ttl)
    
    def create_prompt(self, prompt_id: str, name: str, content: List[Dict[str, Any]],
                     version: str = "1.0.0", metadata: Optional[Dict[str, Any]] = None,
                     status: PromptStatus = PromptStatus.DRAFT) -> Prompt:
        """
        Create a new prompt
        
        Args:
            prompt_id: Unique identifier for the prompt
            name: Human-readable name
            content: List of content parts with role and content/template
            version: Semantic version string
            metadata: Optional metadata dict
            status: Prompt status
            
        Returns:
            Created Prompt object
        """
        # Parse content
        prompt_content = []
        for content_item in content:
            role = PromptRole(content_item["role"])
            
            # Handle template vs plain content
            if isinstance(content_item["content"], dict) and "template" in content_item["content"]:
                template_data = content_item["content"]
                content_obj = PromptTemplate(
                    template=template_data["template"],
                    variables=template_data.get("variables", {}),
                    required_variables=template_data.get("required_variables", []),
                    template_engine=template_data.get("template_engine", "jinja2")
                )
            else:
                content_obj = content_item["content"]
            
            prompt_content.append(PromptContent(role=role, content=content_obj))
        
        # Create metadata
        if metadata:
            prompt_metadata = PromptMetadata(**metadata)
        else:
            prompt_metadata = PromptMetadata()
        
        # Create version
        prompt_version = PromptVersion(version=version)
        
        # Create prompt
        prompt = Prompt(
            id=prompt_id,
            name=name,
            content=prompt_content,
            version=prompt_version,
            status=status,
            metadata=prompt_metadata
        )
        
        # Register the prompt
        if not self.registry.register_prompt(prompt):
            raise RuntimeError(f"Failed to register prompt {prompt_id}")
        
        return prompt
    
    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """Get a prompt by ID and optional version"""
        return self.registry.get_prompt(prompt_id, version)
    
    def get_active_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get the active version of a prompt"""
        return self.registry.get_active_prompt(prompt_id)
    
    def render_prompt(self, prompt_id: str, context: Dict[str, Any] = None,
                     version: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """
        Render a prompt with context variables
        
        Args:
            prompt_id: ID of the prompt to render
            context: Variables to use in template rendering
            version: Optional specific version
            
        Returns:
            List of rendered messages or None if prompt not found
        """
        prompt = self.get_prompt(prompt_id, version)
        if not prompt:
            return None
        
        return prompt.render(context or {})
    
    def create_prompt_version(self, prompt_id: str, new_version: str,
                            content: Optional[List[Dict[str, Any]]] = None,
                            metadata: Optional[Dict[str, Any]] = None,
                            changelog: Optional[str] = None) -> Optional[Prompt]:
        """
        Create a new version of an existing prompt
        
        Args:
            prompt_id: ID of the existing prompt
            new_version: New semantic version
            content: New content (if None, copies from current version)
            metadata: Updated metadata (if None, copies from current version)
            changelog: Description of changes
            
        Returns:
            New prompt version or None if base prompt not found
        """
        base_prompt = self.get_prompt(prompt_id)
        if not base_prompt:
            return None
        
        # Use provided content or copy from base
        if content is None:
            new_content = base_prompt.content
        else:
            # Parse new content
            new_content = []
            for content_item in content:
                role = PromptRole(content_item["role"])
                
                if isinstance(content_item["content"], dict) and "template" in content_item["content"]:
                    template_data = content_item["content"]
                    content_obj = PromptTemplate(
                        template=template_data["template"],
                        variables=template_data.get("variables", {}),
                        required_variables=template_data.get("required_variables", []),
                        template_engine=template_data.get("template_engine", "jinja2")
                    )
                else:
                    content_obj = content_item["content"]
                
                new_content.append(PromptContent(role=role, content=content_obj))
        
        # Use provided metadata or copy from base
        if metadata is None:
            new_metadata = base_prompt.metadata
        else:
            new_metadata = PromptMetadata(**metadata)
        
        # Create new version
        new_version_obj = PromptVersion(version=new_version, changelog=changelog)
        
        # Create new prompt
        new_prompt = Prompt(
            id=prompt_id,
            name=base_prompt.name,
            content=new_content,
            version=new_version_obj,
            status=PromptStatus.DRAFT,  # New versions start as draft
            metadata=new_metadata,
            parent_id=base_prompt.id
        )
        
        # Register the new version
        if not self.registry.register_prompt(new_prompt):
            raise RuntimeError(f"Failed to register new version {new_version} for prompt {prompt_id}")
        
        return new_prompt
    
    def activate_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Activate a prompt version"""
        # Deactivate current active version first
        current_active = self.get_active_prompt(prompt_id)
        if current_active:
            self.registry.update_prompt_status(prompt_id, PromptStatus.DEPRECATED, 
                                             current_active.version.version)
        
        # Activate the specified version
        return self.registry.update_prompt_status(prompt_id, PromptStatus.ACTIVE, version)
    
    def deactivate_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Deactivate a prompt version"""
        return self.registry.update_prompt_status(prompt_id, PromptStatus.DEPRECATED, version)
    
    def list_prompts(self, status: Optional[str] = None, 
                    category: Optional[str] = None) -> List[Prompt]:
        """List prompts with optional filters"""
        return self.registry.list_prompts(status, category)
    
    def list_prompt_versions(self, prompt_id: str) -> List[str]:
        """List all versions of a prompt"""
        return self.registry.get_prompt_versions(prompt_id)
    
    def delete_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Delete a prompt or specific version"""
        return self.registry.delete_prompt(prompt_id, version)
    
    def log_execution(self, prompt_id: str, prompt_version: str,
                     execution_context: Dict[str, Any],
                     rendered_messages: List[Dict[str, str]],
                     llm_provider: str, llm_model: Optional[str] = None,
                     llm_response: Optional[str] = None,
                     execution_time_ms: Optional[float] = None,
                     token_usage: Optional[Dict[str, int]] = None,
                     success: bool = True, error_message: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log prompt execution for evaluation and training
        
        Returns:
            Execution ID
        """
        execution_id = str(uuid.uuid4())
        
        execution = PromptExecution(
            id=execution_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            execution_context=execution_context,
            rendered_messages=rendered_messages,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_response=llm_response,
            execution_time_ms=execution_time_ms,
            token_usage=token_usage,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        if not self.storage.save_execution(execution):
            print(f"Warning: Failed to save execution log {execution_id}")
        
        return execution_id
    
    def get_execution_history(self, prompt_id: str, version: Optional[str] = None,
                             limit: int = 100) -> List[PromptExecution]:
        """Get execution history for a prompt"""
        return self.storage.get_executions(prompt_id, version, limit)
    
    def validate_template(self, template: str, required_variables: List[str],
                         test_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a prompt template
        
        Returns:
            Validation result with success status and any errors
        """
        try:
            prompt_template = PromptTemplate(
                template=template,
                required_variables=required_variables
            )
            
            # Try to render with test context
            rendered = prompt_template.render(test_context)
            
            return {
                "valid": True,
                "rendered": rendered,
                "errors": []
            }
        except Exception as e:
            return {
                "valid": False,
                "rendered": None,
                "errors": [str(e)]
            }
    
    def export_training_data(self, prompt_id: str, version: Optional[str] = None,
                           format: str = "jsonl", limit: int = 1000) -> str:
        """
        Export execution history as training data
        
        Args:
            prompt_id: Prompt to export data for
            version: Optional specific version
            format: Export format ('jsonl', 'csv')
            limit: Maximum number of records
            
        Returns:
            Formatted training data as string
        """
        executions = self.get_execution_history(prompt_id, version, limit)
        
        if format == "jsonl":
            import json
            lines = []
            for execution in executions:
                if execution.success and execution.llm_response:
                    training_record = {
                        "messages": execution.rendered_messages,
                        "response": execution.llm_response,
                        "metadata": {
                            "prompt_id": execution.prompt_id,
                            "prompt_version": execution.prompt_version,
                            "execution_time_ms": execution.execution_time_ms,
                            "token_usage": execution.token_usage
                        }
                    }
                    lines.append(json.dumps(training_record))
            return "\n".join(lines)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(["prompt_id", "version", "input", "output", "execution_time_ms"])
            
            # Data
            for execution in executions:
                if execution.success and execution.llm_response:
                    # Combine messages into single input
                    input_text = " ".join([msg["content"] for msg in execution.rendered_messages])
                    writer.writerow([
                        execution.prompt_id,
                        execution.prompt_version,
                        input_text,
                        execution.llm_response,
                        execution.execution_time_ms
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        cache_stats = self.registry.get_cache_stats()
        
        return {
            "cache_size": cache_stats["cache_size"],
            "active_prompts_count": cache_stats["active_prompts_count"],
            "storage_backend": type(self.storage).__name__
        }
    
    def refresh(self) -> None:
        """Refresh service state from storage"""
        self.registry.refresh()


def create_prompt_service(backend_type: str = "auto", **kwargs) -> PromptService:
    """
    Factory function to create a prompt service
    
    Args:
        backend_type: Storage backend type ('auto', 'file', 'supabase')
        **kwargs: Additional arguments for storage backend
        
    Returns:
        Configured PromptService instance
    """
    # Get cache TTL from environment or use default
    cache_ttl = int(os.getenv("PROMPT_CACHE_TTL", "3600"))
    
    # Create storage backend
    storage_backend = create_storage_backend(backend_type, **kwargs)
    
    return PromptService(storage_backend, cache_ttl)