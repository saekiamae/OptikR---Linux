"""
Context Manager Optimizer Plugin
Thin wrapper so OptimizerPluginLoader can discover and load the Context Manager
through the standard optimizer.py + initialize() convention.

The real implementation lives in context_manager.py (ContextManagerPlugin).
Since context_manager.py uses relative imports, we must ensure the package
is properly importable before loading it.
"""

import importlib
import sys
from pathlib import Path
from typing import Any

# Ensure the plugins directory is on sys.path so that
# 'plugins.enhancers.optimizers.context_manager' is importable as a package.
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import via the package path so relative imports inside context_manager.py work
_mod = importlib.import_module("plugins.enhancers.optimizers.context_manager.context_manager")
ContextManagerPlugin = _mod.ContextManagerPlugin


def initialize(config: dict[str, Any]) -> "ContextManagerPlugin":
    """Initialize the context manager optimizer plugin."""
    return ContextManagerPlugin(config)
