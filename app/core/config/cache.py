"""
Configuration caching: in-memory config caching with memory tracking.

Extracted from ConfigManager cache logic (_config_cache, clear_cache, get_memory_usage).

Requirements: 1.1
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from datetime import datetime

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema
    from app.core.credential_encryptor import CredentialEncryptor

logger = logging.getLogger(__name__)


class ConfigCache:
    """In-memory configuration cache with memory usage tracking."""

    def __init__(self) -> None:
        self._config_cache: dict[str, Any] | None = None
        self._cache_timestamp: datetime | None = None
        self._memory_usage_mb: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self._config_cache is not None

    def get(self) -> dict[str, Any] | None:
        if self._config_cache is not None:
            logger.debug("Returning cached configuration")
            return self._config_cache.copy()
        return None

    def put(self, config: dict[str, Any]) -> None:
        self._config_cache = config.copy()
        self._cache_timestamp = datetime.now()

    def clear(self) -> None:
        self._config_cache = None
        self._cache_timestamp = None
        self._memory_usage_mb = 0.0
        logger.debug("Configuration cache cleared")

    def get_memory_usage(
        self,
        schema: ConfigSchema | None = None,
        encryptor: CredentialEncryptor | None = None,
        config_path: str | Path | None = None,
    ) -> float:
        """
        Get current memory usage of the configuration system in megabytes.

        Args:
            schema: Optional schema object to include in measurement
            encryptor: Optional encryptor object to include in measurement
            config_path: Optional config_path to include in measurement

        Returns:
            Memory usage in megabytes
        """
        total_bytes = 0

        if self._config_cache is not None:
            total_bytes += sys.getsizeof(self._config_cache)
            total_bytes += self._get_deep_size(self._config_cache)

        if schema is not None:
            total_bytes += sys.getsizeof(schema)
            if hasattr(schema, "options"):
                total_bytes += sys.getsizeof(schema.options)
                total_bytes += self._get_deep_size(schema.options)

        if encryptor is not None:
            total_bytes += sys.getsizeof(encryptor)

        if config_path is not None:
            total_bytes += sys.getsizeof(config_path)

        total_bytes += sys.getsizeof(self._cache_timestamp)

        memory_mb = total_bytes / (1024 * 1024)
        self._memory_usage_mb = memory_mb
        return memory_mb

    def _get_deep_size(self, obj: Any, seen: set[int] | None = None) -> int:
        """Recursively calculate the size of an object and its contents."""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += self._get_deep_size(key, seen)
                size += self._get_deep_size(value, seen)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                size += self._get_deep_size(item, seen)

        return size
