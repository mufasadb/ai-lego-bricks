"""
Prompt registry with caching and fast access
"""

import time
from typing import Dict, Optional, List, Set
from threading import RLock

from .prompt_models import Prompt, PromptStatus
from .prompt_storage import PromptStorageBackend


class PromptCache:
    """Thread-safe cache for prompts with TTL support"""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict] = {}  # key -> {prompt, timestamp}
        self._lock = RLock()

    def _make_key(self, prompt_id: str, version: Optional[str] = None) -> str:
        """Create cache key"""
        if version:
            return f"{prompt_id}::{version}"
        return f"{prompt_id}::latest"

    def get(self, prompt_id: str, version: Optional[str] = None) -> Optional[Prompt]:
        """Get prompt from cache"""
        key = self._make_key(prompt_id, version)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if time.time() - entry["timestamp"] > self.ttl_seconds:
                    del self._cache[key]
                    return None

                return entry["prompt"]

        return None

    def put(self, prompt: Prompt, version: Optional[str] = None) -> None:
        """Put prompt in cache"""
        key = self._make_key(prompt.id, version or prompt.version.version)

        with self._lock:
            self._cache[key] = {"prompt": prompt, "timestamp": time.time()}

    def invalidate(self, prompt_id: str, version: Optional[str] = None) -> None:
        """Invalidate cached prompt"""
        if version:
            key = self._make_key(prompt_id, version)
            with self._lock:
                self._cache.pop(key, None)
        else:
            # Invalidate all versions of this prompt
            with self._lock:
                keys_to_remove = [
                    k for k in self._cache.keys() if k.startswith(f"{prompt_id}::")
                ]
                for key in keys_to_remove:
                    del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get cache size"""
        with self._lock:
            return len(self._cache)


class PromptRegistry:
    """Registry for managing prompts with caching and fast access"""

    def __init__(self, storage_backend: PromptStorageBackend, cache_ttl: int = 3600):
        self.storage = storage_backend
        self.cache = PromptCache(cache_ttl)
        self._active_prompts: Set[str] = set()
        self._lock = RLock()

        # Load active prompts on startup
        self._refresh_active_prompts()

    def _refresh_active_prompts(self) -> None:
        """Refresh list of active prompt IDs"""
        try:
            active_prompts = self.storage.list_prompts(status=PromptStatus.ACTIVE.value)
            with self._lock:
                self._active_prompts = {p.id for p in active_prompts}
        except Exception as e:
            print(f"Warning: Could not refresh active prompts: {e}")

    def register_prompt(self, prompt: Prompt) -> bool:
        """Register a new prompt"""
        try:
            # Save to storage
            success = self.storage.save_prompt(prompt)

            if success:
                # Update cache
                self.cache.put(prompt)
                self.cache.put(prompt, prompt.version.version)

                # Update active prompts if needed
                if prompt.status == PromptStatus.ACTIVE:
                    with self._lock:
                        self._active_prompts.add(prompt.id)

                return True

            return False
        except Exception as e:
            print(f"Error registering prompt {prompt.id}: {e}")
            return False

    def get_prompt(
        self, prompt_id: str, version: Optional[str] = None
    ) -> Optional[Prompt]:
        """Get a prompt by ID and optional version"""
        # Try cache first
        cached_prompt = self.cache.get(prompt_id, version)
        if cached_prompt:
            return cached_prompt

        # Load from storage
        try:
            prompt = self.storage.get_prompt(prompt_id, version)

            if prompt:
                # Cache the result
                self.cache.put(prompt, version)
                if not version:  # Also cache as latest
                    self.cache.put(prompt)

                return prompt

            return None
        except Exception as e:
            print(f"Error getting prompt {prompt_id}: {e}")
            return None

    def get_active_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get active version of a prompt"""
        # Check if prompt is in active set
        with self._lock:
            if prompt_id not in self._active_prompts:
                return None

        prompt = self.get_prompt(prompt_id)

        # Double-check status
        if prompt and prompt.status == PromptStatus.ACTIVE:
            return prompt

        return None

    def list_prompts(
        self, status: Optional[str] = None, category: Optional[str] = None
    ) -> List[Prompt]:
        """List prompts with optional filters"""
        try:
            return self.storage.list_prompts(status, category)
        except Exception as e:
            print(f"Error listing prompts: {e}")
            return []

    def get_prompt_versions(self, prompt_id: str) -> List[str]:
        """Get all versions of a prompt"""
        try:
            return self.storage.get_prompt_versions(prompt_id)
        except Exception as e:
            print(f"Error getting versions for prompt {prompt_id}: {e}")
            return []

    def update_prompt_status(
        self, prompt_id: str, status: PromptStatus, version: Optional[str] = None
    ) -> bool:
        """Update prompt status"""
        try:
            prompt = self.get_prompt(prompt_id, version)
            if not prompt:
                return False

            prompt.status = status
            success = self.storage.save_prompt(prompt)

            if success:
                # Invalidate cache
                self.cache.invalidate(prompt_id, version)

                # Update active prompts set
                with self._lock:
                    if status == PromptStatus.ACTIVE:
                        self._active_prompts.add(prompt_id)
                    else:
                        self._active_prompts.discard(prompt_id)

                return True

            return False
        except Exception as e:
            print(f"Error updating prompt {prompt_id} status: {e}")
            return False

    def delete_prompt(self, prompt_id: str, version: Optional[str] = None) -> bool:
        """Delete a prompt"""
        try:
            success = self.storage.delete_prompt(prompt_id, version)

            if success:
                # Invalidate cache
                self.cache.invalidate(prompt_id, version)

                # Remove from active prompts if no versions left
                if not version:
                    with self._lock:
                        self._active_prompts.discard(prompt_id)

                return True

            return False
        except Exception as e:
            print(f"Error deleting prompt {prompt_id}: {e}")
            return False

    def validate_prompt_reference(
        self, prompt_id: str, version: Optional[str] = None
    ) -> bool:
        """Validate that a prompt reference exists and is accessible"""
        try:
            prompt = self.get_prompt(prompt_id, version)
            return prompt is not None
        except Exception:
            return False

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": self.cache.size(),
            "active_prompts_count": len(self._active_prompts),
        }

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()

    def refresh(self) -> None:
        """Refresh registry state from storage"""
        self.clear_cache()
        self._refresh_active_prompts()

    def preload_prompts(self, prompt_ids: List[str]) -> None:
        """Preload specific prompts into cache"""
        for prompt_id in prompt_ids:
            try:
                self.get_prompt(prompt_id)  # This will cache it
            except Exception as e:
                print(f"Warning: Could not preload prompt {prompt_id}: {e}")

    def get_prompt_by_name(
        self, name: str, version: Optional[str] = None
    ) -> Optional[Prompt]:
        """Get prompt by name (convenience method)"""
        # In a production system, you might want to maintain a name->id mapping
        # For now, we'll search through all prompts
        try:
            all_prompts = self.list_prompts()
            for prompt in all_prompts:
                if prompt.name == name:
                    if version is None or prompt.version.version == version:
                        return prompt
            return None
        except Exception as e:
            print(f"Error getting prompt by name {name}: {e}")
            return None
