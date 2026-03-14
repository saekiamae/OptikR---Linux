"""
Unified Plugin Manager

Thin wrapper that exposes a PluginRegistry for plugin metadata
discovery and querying across all plugin types.
"""
import logging

from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class UnifiedPluginManager:
    """Single manager for all plugin types."""

    def __init__(
        self,
        plugin_directories: list[str] | None = None,
    ) -> None:
        self._plugin_directories: list[str] = plugin_directories or []
        self._registry = PluginRegistry()

    @property
    def plugin_directories(self) -> list[str]:
        return list(self._plugin_directories)

    @property
    def registry(self) -> PluginRegistry:
        return self._registry
