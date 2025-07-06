"""
Storage backends for prompt management
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .prompt_models import Prompt, PromptExecution, PromptEvaluation


class PromptStorageBackend(ABC):
    """Abstract base class for prompt storage backends"""
    
    @abstractmethod
    def save_prompt(self, prompt: Prompt) -> bool:
        """Save a prompt to storage"""
        pass
    
    @abstractmethod
    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """Retrieve a prompt by ID and optional version"""
        pass
    
    @abstractmethod
    def list_prompts(self, status: Optional[str] = None, 
                    category: Optional[str] = None) -> List[Prompt]:
        """List prompts with optional filters"""
        pass
    
    @abstractmethod
    def delete_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Delete a prompt"""
        pass
    
    @abstractmethod
    def get_prompt_versions(self, prompt_id: str) -> List[str]:
        """Get all versions of a prompt"""
        pass
    
    @abstractmethod
    def save_execution(self, execution: PromptExecution) -> bool:
        """Save prompt execution record"""
        pass
    
    @abstractmethod
    def get_executions(self, prompt_id: str, version: Optional[str] = None,
                      limit: int = 100) -> List[PromptExecution]:
        """Get execution records for a prompt"""
        pass


class FilePromptStorage(PromptStorageBackend):
    """File-based prompt storage backend"""
    
    def __init__(self, storage_path: str = "./prompts"):
        self.storage_path = Path(storage_path)
        self.prompts_dir = self.storage_path / "prompts"
        self.executions_dir = self.storage_path / "executions"
        
        # Create directories if they don't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.executions_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_prompt_file_path(self, prompt_id: str, version: str) -> Path:
        """Get file path for a prompt"""
        return self.prompts_dir / f"{prompt_id}_v{version}.json"
    
    def _get_execution_file_path(self, execution_id: str) -> Path:
        """Get file path for an execution record"""
        return self.executions_dir / f"{execution_id}.json"
    
    def save_prompt(self, prompt: Prompt) -> bool:
        """Save a prompt to file"""
        try:
            file_path = self._get_prompt_file_path(prompt.id, prompt.version.version)
            
            # Update timestamp
            prompt.updated_at = datetime.now()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt.model_dump_json(indent=2))
            
            return True
        except Exception as e:
            print(f"Error saving prompt {prompt.id}: {e}")
            return False
    
    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """Retrieve a prompt from file"""
        try:
            if version:
                file_path = self._get_prompt_file_path(prompt_id, version)
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return Prompt.model_validate(data)
            else:
                # Get the latest version
                versions = self.get_prompt_versions(prompt_id)
                if versions:
                    latest_version = max(versions)  # Simple string comparison for now
                    return self.get_prompt(prompt_id, latest_version)
            
            return None
        except Exception as e:
            print(f"Error retrieving prompt {prompt_id}: {e}")
            return None
    
    def list_prompts(self, status: Optional[str] = None, 
                    category: Optional[str] = None) -> List[Prompt]:
        """List all prompts with optional filters"""
        prompts = []
        
        try:
            for file_path in self.prompts_dir.glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                prompt = Prompt.model_validate(data)
                
                # Apply filters
                if status and prompt.status.value != status:
                    continue
                if category and prompt.metadata.category != category:
                    continue
                
                prompts.append(prompt)
        
        except Exception as e:
            print(f"Error listing prompts: {e}")
        
        return prompts
    
    def delete_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Delete a prompt file"""
        try:
            if version:
                file_path = self._get_prompt_file_path(prompt_id, version)
                if file_path.exists():
                    file_path.unlink()
                    return True
            else:
                # Delete all versions
                versions = self.get_prompt_versions(prompt_id)
                for v in versions:
                    self.delete_prompt(prompt_id, v)
                return len(versions) > 0
            
            return False
        except Exception as e:
            print(f"Error deleting prompt {prompt_id}: {e}")
            return False
    
    def get_prompt_versions(self, prompt_id: str) -> List[str]:
        """Get all versions of a prompt"""
        versions = []
        
        try:
            for file_path in self.prompts_dir.glob(f"{prompt_id}_v*.json"):
                # Extract version from filename
                filename = file_path.stem
                version_part = filename.split('_v')[1]
                versions.append(version_part)
        
        except Exception as e:
            print(f"Error getting versions for prompt {prompt_id}: {e}")
        
        return sorted(versions)
    
    def save_execution(self, execution: PromptExecution) -> bool:
        """Save execution record to file"""
        try:
            file_path = self._get_execution_file_path(execution.id)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(execution.model_dump_json(indent=2))
            
            return True
        except Exception as e:
            print(f"Error saving execution {execution.id}: {e}")
            return False
    
    def get_executions(self, prompt_id: str, version: Optional[str] = None,
                      limit: int = 100) -> List[PromptExecution]:
        """Get execution records for a prompt"""
        executions = []
        
        try:
            count = 0
            for file_path in self.executions_dir.glob("*.json"):
                if count >= limit:
                    break
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                execution = PromptExecution.model_validate(data)
                
                # Filter by prompt_id and version
                if execution.prompt_id == prompt_id:
                    if version is None or execution.prompt_version == version:
                        executions.append(execution)
                        count += 1
        
        except Exception as e:
            print(f"Error getting executions for prompt {prompt_id}: {e}")
        
        return executions


class SupabasePromptStorage(PromptStorageBackend):
    """Supabase-based prompt storage backend"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        try:
            from supabase import create_client
            import os
            
            url = supabase_url or os.getenv("SUPABASE_URL")
            key = supabase_key or os.getenv("SUPABASE_ANON_KEY")
            
            if not url or not key:
                raise ValueError("Supabase URL and key required")
            
            self.client = create_client(url, key)
            self._ensure_tables()
            
        except ImportError:
            raise ImportError("supabase package required for SupabasePromptStorage")
    
    def _ensure_tables(self):
        """Ensure required tables exist"""
        try:
            # Create prompts table if it doesn't exist
            self.client.table("prompts").select("id").limit(1).execute()
        except Exception:
            # Table doesn't exist, create it
            try:
                self.client.rpc("create_prompts_table").execute()
            except Exception:
                # If RPC doesn't exist, we need to create tables manually
                print("Warning: prompts table doesn't exist. Please create it manually or use Supabase migrations.")
                print("Required schema:")
                print("- prompts table: id, name, content (jsonb), metadata (jsonb), version (jsonb), status, created_at, updated_at, parent_id")
                print("- prompt_executions table: id, prompt_id, prompt_version, execution_context (jsonb), rendered_messages (jsonb), llm_provider, llm_model, llm_response, execution_time_ms, token_usage (jsonb), success, error_message, metadata (jsonb), created_at")
        
        try:
            # Create prompt_executions table if it doesn't exist
            self.client.table("prompt_executions").select("id").limit(1).execute()
        except Exception:
            # Table doesn't exist, warn user
            print("Warning: prompt_executions table doesn't exist. Please create it manually or use Supabase migrations.")
    
    def save_prompt(self, prompt: Prompt) -> bool:
        """Save a prompt to Supabase"""
        try:
            # Update timestamp
            prompt.updated_at = datetime.now()
            
            # Convert to dict for storage
            prompt_data = prompt.model_dump(mode='json')
            # Keep JSON fields as objects for JSONB columns, don't double-encode
            # prompt_data['content'] = json.dumps(prompt_data['content'])
            # prompt_data['metadata'] = json.dumps(prompt_data['metadata'])
            # prompt_data['version'] = json.dumps(prompt_data['version'])
            
            # Upsert the prompt
            result = self.client.table("prompts").upsert(prompt_data).execute()
            
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving prompt {prompt.id} to Supabase: {e}")
            return False
    
    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """Retrieve a prompt from Supabase"""
        try:
            query = self.client.table("prompts").select("*").eq("id", prompt_id)
            
            if version:
                query = query.eq("version->>version", version)
            else:
                # Get latest version - order by version and take first
                query = query.order("version->>version", desc=True).limit(1)
            
            result = query.execute()
            
            if result.data:
                data = result.data[0]
                # JSON fields are already parsed by Supabase for JSONB columns
                # data['content'] = json.loads(data['content'])
                # data['metadata'] = json.loads(data['metadata'])
                # data['version'] = json.loads(data['version'])
                
                return Prompt.model_validate(data)
            
            return None
        except Exception as e:
            print(f"Error retrieving prompt {prompt_id} from Supabase: {e}")
            return None
    
    def list_prompts(self, status: Optional[str] = None, 
                    category: Optional[str] = None) -> List[Prompt]:
        """List prompts from Supabase with optional filters"""
        try:
            query = self.client.table("prompts").select("*")
            
            if status:
                query = query.eq("status", status)
            if category:
                query = query.eq("metadata->>category", category)
            
            result = query.execute()
            
            prompts = []
            for data in result.data:
                # Parse JSON fields back
                data['content'] = json.loads(data['content'])
                data['metadata'] = json.loads(data['metadata'])
                data['version'] = json.loads(data['version'])
                
                prompts.append(Prompt.model_validate(data))
            
            return prompts
        except Exception as e:
            print(f"Error listing prompts from Supabase: {e}")
            return []
    
    def delete_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Delete a prompt from Supabase"""
        try:
            query = self.client.table("prompts").delete().eq("id", prompt_id)
            
            if version:
                query = query.eq("version->>version", version)
            
            result = query.execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error deleting prompt {prompt_id} from Supabase: {e}")
            return False
    
    def get_prompt_versions(self, prompt_id: str) -> List[str]:
        """Get all versions of a prompt from Supabase"""
        try:
            result = self.client.table("prompts").select("version").eq("id", prompt_id).execute()
            
            versions = []
            for data in result.data:
                version_data = json.loads(data['version'])
                versions.append(version_data['version'])
            
            return sorted(versions)
        except Exception as e:
            print(f"Error getting versions for prompt {prompt_id} from Supabase: {e}")
            return []
    
    def save_execution(self, execution: PromptExecution) -> bool:
        """Save execution record to Supabase"""
        try:
            execution_data = execution.model_dump()
            execution_data['execution_context'] = json.dumps(execution_data['execution_context'])
            execution_data['rendered_messages'] = json.dumps(execution_data['rendered_messages'])
            execution_data['token_usage'] = json.dumps(execution_data['token_usage']) if execution_data['token_usage'] else None
            execution_data['metadata'] = json.dumps(execution_data['metadata'])
            
            result = self.client.table("prompt_executions").insert(execution_data).execute()
            
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving execution {execution.id} to Supabase: {e}")
            return False
    
    def get_executions(self, prompt_id: str, version: Optional[str] = None,
                      limit: int = 100) -> List[PromptExecution]:
        """Get execution records for a prompt from Supabase"""
        try:
            query = self.client.table("prompt_executions").select("*").eq("prompt_id", prompt_id).limit(limit)
            
            if version:
                query = query.eq("prompt_version", version)
            
            result = query.execute()
            
            executions = []
            for data in result.data:
                # Parse JSON fields back
                data['execution_context'] = json.loads(data['execution_context'])
                data['rendered_messages'] = json.loads(data['rendered_messages'])
                data['token_usage'] = json.loads(data['token_usage']) if data['token_usage'] else None
                data['metadata'] = json.loads(data['metadata'])
                
                executions.append(PromptExecution.model_validate(data))
            
            return executions
        except Exception as e:
            print(f"Error getting executions for prompt {prompt_id} from Supabase: {e}")
            return []


def create_storage_backend(backend_type: str = "auto", **kwargs) -> PromptStorageBackend:
    """Factory function to create storage backends"""
    if backend_type == "auto":
        # Auto-detect available backend
        try:
            import os
            if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"):
                backend_type = "supabase"
            else:
                backend_type = "file"
        except:
            backend_type = "file"
    
    if backend_type == "file":
        storage_path = kwargs.get("storage_path", "./prompts")
        return FilePromptStorage(storage_path)
    
    elif backend_type == "supabase":
        supabase_url = kwargs.get("supabase_url")
        supabase_key = kwargs.get("supabase_key")
        return SupabasePromptStorage(supabase_url, supabase_key)
    
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")