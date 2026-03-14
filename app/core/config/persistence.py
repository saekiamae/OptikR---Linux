"""
Configuration persistence: load/save JSON, backup, corruption recovery.

Extracted from ConfigManager to handle all file I/O concerns.

Requirements: 1.1
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from datetime import datetime

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema
    from app.core.credential_encryptor import CredentialEncryptor

logger = logging.getLogger(__name__)


class ConfigPersistence:
    """Handles loading and saving configuration to/from disk."""

    def __init__(
        self,
        config_path: str | Path,
        schema: ConfigSchema,
        encryptor: CredentialEncryptor | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.schema = schema
        self.encryptor = encryptor

    def load(
        self,
        decrypt_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        migrate_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        validate_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        default_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Load configuration from disk.

        Pipeline order: deserialize -> decrypt -> migrate -> validate.

        Args:
            decrypt_fn: callable(config) -> config with decrypted sensitive values
            migrate_fn: callable(config) -> config with legacy keys migrated.
                        Called only when it returns a different dict (i.e. when
                        migration was needed).  A pre-migration backup is
                        created automatically.
            validate_fn: callable(config) -> validated config
            default_fn: callable() -> default config dict

        Returns:
            Configuration dictionary
        """
        if default_fn is None:
            default_fn = dict

        if not self.config_path.exists():
            logger.info("Config file not found, using defaults: %s", self.config_path)
            return default_fn()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config: dict[str, Any] = json.load(f)

            logger.info("Loaded configuration from %s", self.config_path)

            if decrypt_fn is not None:
                config = decrypt_fn(config)
            if migrate_fn is not None:
                migrated = migrate_fn(config)
                if migrated is not config:
                    self._create_backup("pre_migration")
                    config = migrated
            if validate_fn is not None:
                config = validate_fn(config)

            return config

        except json.JSONDecodeError as e:
            logger.error("Corrupted JSON in %s: %s", self.config_path, e)
            return self._handle_corrupted_config(default_fn)

        except PermissionError as e:
            logger.error("Permission denied reading %s: %s", self.config_path, e)
            return default_fn()

        except Exception as e:
            logger.error("Failed to load config from %s: %s", self.config_path, e)
            return default_fn()

    def save(
        self,
        config: dict[str, Any],
        encrypt_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        validate_all_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Save configuration to disk with atomic write.

        Args:
            config: Configuration dictionary to save
            encrypt_fn: callable(config) -> config with encrypted sensitive values
            validate_all_fn: callable(config) -> list of error strings

        Returns:
            Tuple of (success, error_message)
        """
        if validate_all_fn is not None:
            errors = validate_all_fn(config)
            if errors:
                error_msg = f"Configuration validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                return False, error_msg

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix(".json.bak")
                shutil.copy2(self.config_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")

            config_to_save = config
            if encrypt_fn is not None:
                config_to_save = encrypt_fn(config)

            # Atomic write via temp file
            temp_path = self.config_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)

            temp_path.replace(self.config_path)
            logger.info(f"Configuration saved to {self.config_path}")
            return True, ""

        except PermissionError as e:
            error_msg = f"Permission denied writing to {self.config_path}"
            logger.error(f"{error_msg}: {e}")
            return False, error_msg

        except OSError as e:
            if e.errno == 28:
                error_msg = "Insufficient disk space to save configuration"
            else:
                error_msg = f"I/O error: {e}"
            logger.error(f"Failed to save config: {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error saving configuration: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _create_backup(self, label: str) -> Path | None:
        """Create a timestamped backup of the current config file.

        Returns the backup path, or ``None`` if no backup was created.
        """
        if not self.config_path.exists():
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_path.with_suffix(f".{label}_{timestamp}.json")
            shutil.copy2(self.config_path, backup_path)
            logger.info("Created %s backup: %s", label, backup_path)
            return backup_path
        except Exception as e:
            logger.error("Failed to create %s backup: %s", label, e)
            return None

    def _handle_corrupted_config(self, default_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Handle corrupted configuration file by backing up and returning defaults."""
        self._create_backup("corrupted")
        logger.warning("Loading default configuration due to corruption")
        return default_fn()
