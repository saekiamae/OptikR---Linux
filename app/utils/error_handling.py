"""
Import fallback utility for OptikR.

Provides try_import() which attempts imports in order with descriptive
failure messages.

Requirements: 7.2
"""
import importlib
from types import ModuleType


def try_import(*module_names: str) -> ModuleType:
    """Attempt to import modules in order, returning the first successful import.

    Args:
        *module_names: One or more module names to try importing, in priority order.

    Returns:
        The first successfully imported module.

    Raises:
        ImportError: If all import attempts fail, with a message listing all attempted names.
        TypeError: If no module names are provided.
    """
    if not module_names:
        raise TypeError("try_import() requires at least one module name")

    errors: list[str] = []
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name} ({exc})")

    attempted = ", ".join(module_names)
    raise ImportError(
        f"Could not import any of [{attempted}]. "
        f"Errors: {'; '.join(errors)}"
    )
