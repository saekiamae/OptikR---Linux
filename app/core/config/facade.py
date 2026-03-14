"""
ConfigFacade: unified configuration API.

Single entry point for all configuration operations.  Every load and save
goes through the decomposed sub-modules (persistence, validator, encryptor,
migrator, cache) — there is no bypass path.

Requirements: 1.2, 1.3
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config.persistence import ConfigPersistence
from app.core.config.validator import ConfigValidator
from app.core.config.encryptor import ConfigEncryptor
from app.core.config.migrator import ConfigMigrator
from app.core.config.cache import ConfigCache
from app.core.config.utils import get_nested_value, set_nested_value

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema
    from app.core.credential_encryptor import CredentialEncryptor

logger = logging.getLogger(__name__)


class ConfigFacade:
    """
    Unified configuration manager.

    All load/save operations delegate to the sub-modules:
    - ConfigPersistence for disk I/O (backup, atomic write, corruption recovery)
    - ConfigValidator for schema validation
    - ConfigEncryptor for encrypting/decrypting sensitive fields
    - ConfigMigrator for legacy format migration
    - ConfigCache for in-memory caching
    """

    def __init__(self, config_path: str | Path | None = None, schema: ConfigSchema | None = None) -> None:
        if config_path is None:
            from app.utils.path_utils import get_config_file
            config_path = get_config_file()
        if schema is None:
            from app.core.config_schema import ConfigSchema
            schema = ConfigSchema()  # type: ignore[no-untyped-call]

        self._lock = threading.RLock()

        self.config_path = Path(config_path)
        self.schema = schema

        self.encryptor: CredentialEncryptor | None
        try:
            from app.core.credential_encryptor import CredentialEncryptor
            self.encryptor = CredentialEncryptor()
        except ImportError:
            self.encryptor = None

        # Compose sub-modules
        self._persistence = ConfigPersistence(self.config_path, self.schema, self.encryptor)
        self._validator = ConfigValidator(self.schema)
        self._encryptor_adapter = (
            ConfigEncryptor(self.encryptor, self.schema) if self.encryptor else None
        )
        self._migrator = ConfigMigrator(self.schema)
        self._cache = ConfigCache()

        # Load config on init — single path through sub-modules
        self.config = self.load()

    # ------------------------------------------------------------------
    # Core API — all paths go through sub-modules
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load configuration from disk (with decryption, migration, validation, caching)."""
        with self._lock:
            cached = self._cache.get()
            if cached is not None:
                self.config = cached
                return cached

            decrypt_fn = (
                self._encryptor_adapter.decrypt_sensitive_values
                if self._encryptor_adapter
                else None
            )
            validate_fn = self._validator.validate_and_fix
            default_fn = self._get_schema_default_config

            config = self._persistence.load(
                decrypt_fn=decrypt_fn,
                migrate_fn=self._migrate_if_needed,
                validate_fn=validate_fn,
                default_fn=default_fn,
            )

            self._cache.put(config)
            self.config = config
            return config

    def _migrate_if_needed(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a migrated copy if *config* contains legacy flat keys, otherwise return *config* unchanged.

        Returning the same object signals to the persistence layer that no
        migration occurred (so no backup is created).
        """
        if not self._migrator.has_legacy_keys(config):
            return config
        return self._migrator.migrate(config, default_fn=self._get_schema_default_config)

    def save(self, config: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Save configuration to disk (with encryption, validation, backup)."""
        with self._lock:
            if config is None:
                config = self.config

            encrypt_fn = (
                self._encryptor_adapter.encrypt_sensitive_values
                if self._encryptor_adapter
                else None
            )
            validate_all_fn = self.schema.validate_all

            success, error = self._persistence.save(
                config, encrypt_fn=encrypt_fn, validate_all_fn=validate_all_fn
            )

            if success:
                self._cache.put(config)
                self.config = config

            return success, error

    def save_config(self) -> tuple[bool, str]:
        """Save current in-memory config to disk. Alias for save()."""
        with self._lock:
            return self.save(self.config)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value using dot notation.

        Unlike :func:`get_nested_value`, returns *default* only when a key
        segment is missing (a value of ``None`` stored at the key is returned
        as-is).
        """
        with self._lock:
            keys = key.split(".")
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value using dot notation."""
        with self._lock:
            set_nested_value(self.config, key, value)

    # ------------------------------------------------------------------
    # Domain-specific convenience methods
    # ------------------------------------------------------------------

    def get_consent_info(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = self.config.get("consent", {})
            return result

    def set_consent_info(self, consent_given: bool, version: str = "pre-realese-1.0.0") -> None:
        with self._lock:
            self.config["consent"] = {
                "consent_given": consent_given,
                "consent_date": datetime.now().isoformat() if consent_given else None,
                "version": version,
            }

    def get_installation_info(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = self.config.get("installation", {})
            return result

    def set_installation_info(self, install_info: dict[str, Any]) -> None:
        with self._lock:
            self.config["installation"] = install_info

    def get_region_presets(self) -> dict[str, Any]:
        with self._lock:
            presets: dict[str, Any] = self.config.get("presets", {})
            result: dict[str, Any] = presets.get("regions", {})
            return result

    def set_region_preset(self, preset_name: str, preset_data: dict[str, Any]) -> None:
        with self._lock:
            if "presets" not in self.config:
                self.config["presets"] = {"regions": {}}
            if "regions" not in self.config["presets"]:
                self.config["presets"]["regions"] = {}
            self.config["presets"]["regions"][preset_name] = preset_data

    def delete_region_preset(self, preset_name: str) -> bool:
        with self._lock:
            presets = self.config.get("presets", {}).get("regions", {})
            if preset_name in presets:
                del presets[preset_name]
                return True
            return False

    def get_amd_hardware_config(self) -> dict[str, Any]:
        with self._lock:
            hardware: dict[str, Any] = self.config.get("hardware", {})
            result: dict[str, Any] = hardware.get("amd", {})
            return result

    def set_amd_hardware_config(self, amd_config: dict[str, Any]) -> None:
        with self._lock:
            if "hardware" not in self.config:
                self.config["hardware"] = {}
            if "amd" not in self.config["hardware"]:
                self.config["hardware"]["amd"] = {}
            self.config["hardware"]["amd"].update(amd_config)

    def update_amd_cpu_config(self, cpu_info: dict[str, Any]) -> None:
        with self._lock:
            if "hardware" not in self.config:
                self.config["hardware"] = {}
            if "amd" not in self.config["hardware"]:
                self.config["hardware"]["amd"] = {}
            amd_config = self.config["hardware"]["amd"]
            amd_config["cpu_detected"] = cpu_info.get("is_amd", False)
            if amd_config["cpu_detected"]:
                amd_config["cpu_vendor"] = cpu_info.get("vendor", None)
                amd_config["cpu_model"] = cpu_info.get("model", None)
                amd_config["cpu_cores"] = cpu_info.get("cores", None)
                amd_config["cpu_zen_generation"] = cpu_info.get("zen_generation", None)
                amd_config["cpu_simd_support"] = cpu_info.get("simd_support", [])
            amd_config["last_detection_timestamp"] = datetime.now().isoformat()

    def update_amd_gpu_config(self, gpu_info: dict[str, Any] | None, backend: str) -> None:
        with self._lock:
            if "hardware" not in self.config:
                self.config["hardware"] = {}
            if "amd" not in self.config["hardware"]:
                self.config["hardware"]["amd"] = {}
            amd_config = self.config["hardware"]["amd"]
            if gpu_info:
                amd_config["gpu_detected"] = True
                amd_config["gpu_model"] = gpu_info.get("model", None)
                amd_config["gpu_memory_mb"] = gpu_info.get("memory_mb", None)
                amd_config["gpu_device_count"] = gpu_info.get("device_count", None)
                amd_config["rocm_available"] = gpu_info.get("rocm_available", False)
                amd_config["opencl_available"] = "opencl_version" in gpu_info
                amd_config["opencl_version"] = gpu_info.get("opencl_version", None)
            else:
                amd_config["gpu_detected"] = False
                amd_config["gpu_model"] = None
                amd_config["gpu_memory_mb"] = None
                amd_config["gpu_device_count"] = None
                amd_config["rocm_available"] = False
                amd_config["opencl_available"] = False
                amd_config["opencl_version"] = None
            amd_config["gpu_backend"] = backend.lower() if backend else "none"
            amd_config["last_detection_timestamp"] = datetime.now().isoformat()

    def detect_runtime_mode(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "gpu"
        except ImportError:
            pass
        return "cpu"

    def get_runtime_mode(self) -> str:
        mode: str = self.get_setting("performance.runtime_mode", "auto")
        if mode == "auto":
            return self.detect_runtime_mode()
        return mode

    def is_gpu_available(self) -> bool:
        return self.get_runtime_mode() == "gpu"

    # ------------------------------------------------------------------
    # Validated get/set with schema enforcement
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        with self._lock:
            return self.get_setting(key, default)

    def set(self, key: str, value: Any) -> tuple[bool, str]:
        """Set configuration value using dot notation with validation."""
        with self._lock:
            is_valid, error_msg = self.schema.validate(key, value)
            if not is_valid:
                msg = error_msg or "Validation failed"
                logger.warning("Validation failed for %s: %s", key, msg)
                return False, msg

            self.set_setting(key, value)

            # Keep cache in sync
            cached = self._cache.get()
            if cached is not None:
                set_nested_value(cached, key, value)
                self._cache.put(cached)

            option = self.schema.get_option(key)
            if option and option.sensitive:
                logger.debug("Set %s = ***REDACTED***", key)
            else:
                logger.debug("Set %s = %s", key, value)
            return True, ""

    # ------------------------------------------------------------------
    # Migration & defaults
    # ------------------------------------------------------------------

    def migrate(self, old_config: dict[str, Any]) -> dict[str, Any]:
        """Migrate old configuration format to new format."""
        with self._lock:
            return self._migrator.migrate(old_config, default_fn=self._get_schema_default_config)

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration from schema. Used by reset-to-defaults callers."""
        return self._get_schema_default_config()

    def _get_schema_default_config(self) -> dict[str, Any]:
        """Build default configuration dict from schema options."""
        config: dict[str, Any] = {}
        for key, option in self.schema.get_all_options().items():
            set_nested_value(config, key, option.default)
        return config

    # ------------------------------------------------------------------
    # Internal helpers (delegate to shared utils, kept for backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def _set_nested_value_static(config: dict[str, Any], key: str, value: Any) -> None:
        set_nested_value(config, key, value)

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any) -> None:
        set_nested_value(config, key, value)

    def _get_nested_value(self, config: dict[str, Any], key: str) -> Any:
        return get_nested_value(config, key)

    def _merge_configs(self, base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge *updates* into *base*; update values take precedence."""
        merged = dict(base)
        for key, val in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key] = self._merge_configs(merged[key], val)
            else:
                merged[key] = val
        return merged

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_memory_usage(self) -> float:
        with self._lock:
            return self._cache.get_memory_usage(self.schema, self.encryptor, self.config_path)
