"""
Configuration encryption: encrypt/decrypt sensitive fields.

Extracted from ConfigManager._encrypt_sensitive_values / _decrypt_sensitive_values.

Requirements: 1.1
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.config.utils import get_nested_value, set_nested_value

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema
    from app.core.credential_encryptor import CredentialEncryptor

logger = logging.getLogger(__name__)


class ConfigEncryptor:
    """Encrypts and decrypts sensitive configuration values."""

    def __init__(self, encryptor: CredentialEncryptor, schema: ConfigSchema) -> None:
        """
        Args:
            encryptor: CredentialEncryptor instance
            schema: ConfigSchema instance (provides get_sensitive_keys())
        """
        self.encryptor = encryptor
        self.schema = schema

    def encrypt_sensitive_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive values in configuration."""
        encrypted_config = config.copy()

        for key in self.schema.get_sensitive_keys():
            value = get_nested_value(encrypted_config, key)

            if value and isinstance(value, str) and value != "":
                if not self.encryptor.is_encrypted(value):
                    try:
                        encrypted_value = self.encryptor.encrypt(value)
                        set_nested_value(encrypted_config, key, encrypted_value)
                        logger.debug(f"Encrypted sensitive value for {key}")
                    except Exception as e:
                        logger.error(f"Failed to encrypt {key}: {e}")

        return encrypted_config

    def decrypt_sensitive_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Decrypt sensitive values in configuration."""
        decrypted_config = config.copy()

        for key in self.schema.get_sensitive_keys():
            value = get_nested_value(decrypted_config, key)

            if value and isinstance(value, str) and value != "":
                if self.encryptor.is_encrypted(value):
                    try:
                        decrypted_value = self.encryptor.decrypt(value)
                        set_nested_value(decrypted_config, key, decrypted_value)
                        logger.debug(f"Decrypted sensitive value for {key}")
                    except Exception as e:
                        logger.error(f"Failed to decrypt {key}: {e}")
                        set_nested_value(decrypted_config, key, "")

        return decrypted_config
