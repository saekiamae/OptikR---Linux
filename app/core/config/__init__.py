"""
Configuration sub-package for OptikR core.

Re-exports ConfigFacade as ConfigManager and SimpleConfigManager
for backward compatibility.

Requirements: 1.3, 10.2
"""

from app.core.config.facade import ConfigFacade
from app.core.config.utils import get_nested_value, set_nested_value

# Backward-compatible aliases
ConfigManager = ConfigFacade
SimpleConfigManager = ConfigFacade

__all__ = [
    "ConfigFacade", "ConfigManager", "SimpleConfigManager",
    "get_nested_value", "set_nested_value",
]
