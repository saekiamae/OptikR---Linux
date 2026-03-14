"""
Context Manager Plugin
Entry point for the context manager optimizer plugin.

Provides domain-aware context profiles for consistent translations
through locked terms, translation memory, regex rules, and formatting control.
"""

from .context_manager import ContextManagerPlugin
from .context_profile import ContextProfile


def initialize(config):
    """Initialize the context manager plugin."""
    return ContextManagerPlugin(config)
