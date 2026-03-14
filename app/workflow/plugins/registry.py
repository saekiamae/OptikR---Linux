"""
Unified Plugin System - Plugin Registry

Stores and queries plugin metadata.
"""
import logging
import threading

from app.workflow.base.plugin_interface import PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Thread-safe registry for plugin metadata."""

    def __init__(self) -> None:
        self._metadata: dict[str, PluginMetadata] = {}
        self._lock = threading.RLock()

    def register(self, metadata: PluginMetadata) -> None:
        """Register plugin metadata. Overwrites if name already exists."""
        with self._lock:
            self._metadata[metadata.name] = metadata
            logger.info("Registered plugin metadata: %s", metadata.name)

    def get_metadata(self, name: str) -> PluginMetadata | None:
        """Return metadata for *name*, or ``None``."""
        with self._lock:
            return self._metadata.get(name)

    def get_all_metadata(self) -> dict[str, PluginMetadata]:
        """Return a snapshot of all registered metadata."""
        with self._lock:
            return dict(self._metadata)

    def get_by_type(self, plugin_type: PluginType) -> list[PluginMetadata]:
        """Return metadata for all plugins matching *plugin_type*."""
        with self._lock:
            return [
                m for m in self._metadata.values()
                if m.type == plugin_type
            ]
