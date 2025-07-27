"""
Factory for creating and managing ChunkingService instances with different configurations.
"""

from typing import Dict, Optional, Any
from .chunking_service import ChunkingService, ChunkingConfig


class ChunkingServiceFactory:
    """
    Factory class for creating and managing ChunkingService instances.

    This factory provides a clean interface for creating chunking services
    with different configurations while maintaining service lifecycle management.
    """

    def __init__(self):
        """Initialize the factory with default configuration."""
        self._cached_services: Dict[str, ChunkingService] = {}
        self._default_config = ChunkingConfig(
            target_size=1000,
            tolerance=200,
            preserve_paragraphs=True,
            preserve_sentences=True,
            preserve_words=True,
            paragraph_separator="\n\n",
            sentence_pattern=r"[.!?]+\s+",
        )

    def create_service(
        self, config: Optional[ChunkingConfig] = None
    ) -> ChunkingService:
        """
        Create a new ChunkingService instance with the given configuration.

        Args:
            config: Optional configuration. If None, uses default configuration.

        Returns:
            ChunkingService instance configured with the provided settings.
        """
        if config is None:
            config = self._default_config

        return ChunkingService(config)

    def get_or_create_service(self, config_dict: Dict[str, Any]) -> ChunkingService:
        """
        Get or create a ChunkingService instance based on configuration dictionary.

        This method caches services based on configuration to avoid recreating
        identical services repeatedly.

        Args:
            config_dict: Dictionary containing chunking configuration parameters.

        Returns:
            ChunkingService instance with the specified configuration.
        """
        # Create a cache key based on configuration
        cache_key = self._create_cache_key(config_dict)

        # Return cached service if available
        if cache_key in self._cached_services:
            return self._cached_services[cache_key]

        # Create new service with merged configuration
        config = self._create_config_from_dict(config_dict)
        service = self.create_service(config)

        # Cache the service
        self._cached_services[cache_key] = service

        return service

    def get_default_service(self) -> ChunkingService:
        """
        Get a ChunkingService instance with default configuration.

        Returns:
            ChunkingService instance with default settings.
        """
        return self.get_or_create_service({})

    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> ChunkingConfig:
        """
        Create ChunkingConfig from dictionary, merging with defaults.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            ChunkingConfig instance with merged settings.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        # Start with default values
        config_params = {
            "target_size": self._default_config.target_size,
            "tolerance": self._default_config.tolerance,
            "preserve_paragraphs": self._default_config.preserve_paragraphs,
            "preserve_sentences": self._default_config.preserve_sentences,
            "preserve_words": self._default_config.preserve_words,
            "paragraph_separator": self._default_config.paragraph_separator,
            "sentence_pattern": self._default_config.sentence_pattern,
        }

        # Override with provided values and validate
        for key, value in config_dict.items():
            if key in config_params:
                validated_value = self._validate_config_parameter(key, value)
                config_params[key] = validated_value
            # Note: We ignore unknown parameters to allow for forward compatibility

        try:
            return ChunkingConfig(**config_params)
        except Exception as e:
            raise ValueError(f"Failed to create chunking configuration: {e}")

    def _validate_config_parameter(self, key: str, value: Any) -> Any:
        """
        Validate a configuration parameter.

        Args:
            key: Parameter name.
            value: Parameter value.

        Returns:
            Validated parameter value.

        Raises:
            ValueError: If parameter value is invalid.
        """
        if key == "target_size":
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"target_size must be a positive integer, got: {value}"
                )
        elif key == "tolerance":
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"tolerance must be a non-negative integer, got: {value}"
                )
        elif key in ["preserve_paragraphs", "preserve_sentences", "preserve_words"]:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean, got: {value}")
        elif key == "paragraph_separator":
            if not isinstance(value, str):
                raise ValueError(f"paragraph_separator must be a string, got: {value}")
        elif key == "sentence_pattern":
            if not isinstance(value, str):
                raise ValueError(f"sentence_pattern must be a string, got: {value}")
            # Test if it's a valid regex pattern
            try:
                import re

                re.compile(value)
            except re.error as e:
                raise ValueError(f"sentence_pattern must be a valid regex pattern: {e}")

        return value

    def _create_cache_key(self, config_dict: Dict[str, Any]) -> str:
        """
        Create a cache key from configuration dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            String cache key representing the configuration.
        """
        # Sort keys to ensure consistent cache keys
        sorted_items = sorted(config_dict.items())
        return str(hash(tuple(sorted_items)))

    def clear_cache(self):
        """Clear all cached services."""
        self._cached_services.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached services.

        Returns:
            Dictionary containing cache statistics.
        """
        return {
            "cached_services": len(self._cached_services),
            "cache_keys": list(self._cached_services.keys()),
        }


# Factory function for creating ChunkingService instances
def create_chunking_service(config: Optional[Dict[str, Any]] = None) -> ChunkingService:
    """
    Factory function to create a ChunkingService instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        ChunkingService instance
    """
    factory = ChunkingServiceFactory()
    if config is None:
        return factory.get_default_service()
    else:
        return factory.get_or_create_service(config)
