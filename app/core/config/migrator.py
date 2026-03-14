"""
Configuration migration: version migration logic.

Extracted from ConfigManager.migrate.

Requirements: 1.1
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from app.core.config.utils import set_nested_value

if TYPE_CHECKING:
    from app.core.config_schema import ConfigSchema

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Migrates old configuration formats to the current format."""

    FLAT_KEY_MIGRATIONS = {
        "capture_fps": "capture.fps",
        "capture_quality": "capture.quality",
        "capture_timeout": "capture.timeout_ms",
        "capture_method": "capture.method",
        "ocr_engine": "ocr.engine",
        "ocr_confidence": "ocr.confidence_threshold",
        "ocr_retries": "ocr.max_retries",
        "ocr_languages": "ocr.languages",
        "translation_engine": "translation.engine",
        "source_language": "translation.source_language",
        "target_language": "translation.target_language",
        "translation_batch": "translation.batch_size",
        "translation_cache": "cache.translation_cache_size",
        "enable_gpu": "performance.enable_gpu",
        "enable_frame_skip": "performance.enable_frame_skip",
        "enable_cache": "performance.enable_translation_cache",
        "enable_dictionary": "performance.enable_smart_dictionary",
        "google_api_key": "translation.google_api_key",
        "deepl_api_key": "translation.deepl_api_key",
        "azure_api_key": "translation.azure_api_key",
        "azure_region": "translation.azure_region",
    }

    # Renames within nested sections: {section: {old_sub_key: new_sub_key}}
    # Applied after section carry-over so the old sub-key is replaced in-place.
    NESTED_KEY_RENAMES = {
        "translation": {
            "primary_engine": "engine",
            "cache_size_mb": None,
        },
    }

    # Keys that previously lived under their own api_keys section and now
    # belong under translation.  Used by migrate() to relocate them.
    _API_KEYS_TO_TRANSLATION = {
        "google_api_key": "translation.google_api_key",
        "deepl_api_key": "translation.deepl_api_key",
        "azure_api_key": "translation.azure_api_key",
        "azure_region": "translation.azure_region",
    }

    SECTION_KEYS = ["capture", "ocr", "translation", "performance", "cache"]

    def __init__(self, schema: ConfigSchema) -> None:
        self.schema = schema

    def has_legacy_keys(self, config: dict[str, Any]) -> bool:
        """Return True if *config* contains any flat legacy keys, the
        obsolete ``api_keys`` section, or renamed nested keys that need
        migration."""
        if set(self.FLAT_KEY_MIGRATIONS) & set(config):
            return True
        if "api_keys" in config and isinstance(config["api_keys"], dict):
            return True
        for section, renames in self.NESTED_KEY_RENAMES.items():
            if section in config and isinstance(config[section], dict):
                if set(renames) & set(config[section]):
                    return True
        return False

    def migrate(
        self,
        old_config: dict[str, Any],
        default_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Migrate old configuration format to new format.

        Flat legacy keys (e.g. ``capture_fps``) are mapped to their nested
        equivalents (``capture.fps``).  Existing section dicts in *old_config*
        are carried over, but values already set by the flat-key migration
        take priority so that explicitly migrated values are never overwritten.

        Args:
            old_config: Configuration in old format
            default_fn: callable() -> default config dict

        Returns:
            Configuration in new format
        """
        logger.info("Migrating configuration to new format")

        new_config = default_fn() if default_fn else {}

        migrated_dotkeys: set[str] = set()

        for old_key, new_key in self.FLAT_KEY_MIGRATIONS.items():
            if old_key in old_config:
                value = old_config[old_key]
                is_valid, _ = self.schema.validate(new_key, value)
                if is_valid:
                    set_nested_value(new_config, new_key, value)
                    migrated_dotkeys.add(new_key)
                    logger.debug("Migrated %s -> %s", old_key, new_key)
                else:
                    logger.warning("Skipped invalid value for %s: %s", old_key, value)

        for section in self.SECTION_KEYS:
            if section in old_config and isinstance(old_config[section], dict):
                if section not in new_config:
                    new_config[section] = {}
                for sub_key, sub_value in old_config[section].items():
                    dotkey = f"{section}.{sub_key}"
                    if dotkey not in migrated_dotkeys:
                        new_config[section][sub_key] = sub_value

        # Migrate legacy api_keys section into translation.*
        if "api_keys" in old_config and isinstance(old_config["api_keys"], dict):
            for sub_key, new_dotkey in self._API_KEYS_TO_TRANSLATION.items():
                if sub_key in old_config["api_keys"] and new_dotkey not in migrated_dotkeys:
                    value = old_config["api_keys"][sub_key]
                    set_nested_value(new_config, new_dotkey, value)
                    migrated_dotkeys.add(new_dotkey)
                    logger.debug("Migrated api_keys.%s -> %s", sub_key, new_dotkey)

        # Rename sub-keys within sections (e.g. translation.primary_engine → engine)
        for section, renames in self.NESTED_KEY_RENAMES.items():
            if section not in new_config or not isinstance(new_config[section], dict):
                continue
            for old_sub, new_sub in renames.items():
                if old_sub not in new_config[section]:
                    continue
                if new_sub is None:
                    del new_config[section][old_sub]
                    logger.debug("Removed obsolete key %s.%s", section, old_sub)
                elif new_sub not in new_config[section]:
                    new_config[section][new_sub] = new_config[section].pop(old_sub)
                    logger.debug("Renamed %s.%s -> %s.%s", section, old_sub, section, new_sub)
                else:
                    del new_config[section][old_sub]
                    logger.debug(
                        "Dropped %s.%s (canonical %s.%s already set)",
                        section, old_sub, section, new_sub,
                    )

        logger.info("Configuration migration completed")
        return new_config
