"""
Unified Plugin System

Provides a single entry point for plugin metadata registration
and querying across all plugin types.
"""

from .registry import PluginRegistry
from .manager import UnifiedPluginManager

__all__ = [
    "PluginRegistry",
    "UnifiedPluginManager",
]
