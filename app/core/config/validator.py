"""
Configuration validation: schema validation and fixing invalid values.

Extracted from ConfigManager._validate_and_fix.

Requirements: 1.1
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.config.utils import get_nested_value, set_nested_value

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration against a schema and fixes invalid values."""

    def __init__(self, schema: ConfigSchema) -> None:
        self.schema = schema

    def validate_and_fix(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Validate configuration and replace invalid values with defaults.

        Args:
            config: Configuration to validate

        Returns:
            Fixed configuration
        """
        fixed_config = config.copy()

        for key, option in self.schema.get_all_options().items():
            current_value = get_nested_value(fixed_config, key)

            if current_value is None:
                set_nested_value(fixed_config, key, option.default)
                logger.debug(f"Using default for missing key {key}: {option.default}")
            else:
                is_valid, error_msg = option.validate(current_value)
                if not is_valid:
                    set_nested_value(fixed_config, key, option.default)
                    logger.warning(
                        f"Invalid value for {key}: {error_msg}. Using default: {option.default}"
                    )

        return fixed_config
