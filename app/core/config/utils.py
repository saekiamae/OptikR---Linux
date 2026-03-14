"""
Shared configuration utilities: nested dict access via dot-notation keys.

Used by ConfigEncryptor, ConfigValidator, ConfigMigrator, and ConfigFacade.
"""
from typing import Any


def get_nested_value(config: dict[str, Any], key: str) -> Any:
    """Get a value from a nested dict using a dot-notation key.

    Args:
        config: Nested configuration dictionary.
        key: Dot-separated path (e.g. ``"capture.fps"``).

    Returns:
        The value at *key*, or ``None`` if any segment is missing.
    """
    keys = key.split(".")
    value: Any = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def set_nested_value(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-notation key.

    Intermediate dicts are created automatically when they don't exist.

    Args:
        config: Nested configuration dictionary (mutated in-place).
        key: Dot-separated path (e.g. ``"capture.fps"``).
        value: The value to store.
    """
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value
