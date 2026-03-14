"""
Plugin loaders for optimizer and text-processor enhancer plugins.

Extracted from ``runtime_pipeline_optimized.py`` so that
``PipelineFactory`` (and future consumers) can import them without
pulling in the full pipeline module.
"""

import json
import logging
import traceback
import importlib.util
from pathlib import Path
from typing import Any


class TextProcessorPluginLoader:
    """Loads and manages text processor plugins."""

    def __init__(self, plugins_dir: str) -> None:
        self.plugins_dir = Path(plugins_dir)
        self.plugins: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def load_plugins(self) -> dict[str, Any]:
        """Load all text processor plugins from directory."""
        if not self.plugins_dir.exists():
            self.logger.warning("Text processor plugins directory not found: %s", self.plugins_dir)
            return {}

        loaded_names: list[str] = []
        skipped_names: list[str] = []
        failed: list[tuple[str, str]] = []

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_json = plugin_dir / "plugin.json"
            processor_py = plugin_dir / "processor.py"

            if not plugin_json.exists() or not processor_py.exists():
                continue

            try:
                with open(plugin_json, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                name = metadata.get("name", plugin_dir.name)

                if not metadata.get("enabled", True):
                    self.logger.info("Text processor plugin %s is disabled", name)
                    skipped_names.append(name)
                    continue

                self.logger.debug(
                    "Loading text processor plugin: name=%s version=%s category=%s priority=%s",
                    name,
                    metadata.get("version", "?"),
                    metadata.get("category", "?"),
                    metadata.get("priority", "?"),
                )

                spec = importlib.util.spec_from_file_location(
                    f"processor_{name}", processor_py,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                settings = metadata.get("settings", {})
                config = {k: v.get("default") for k, v in settings.items()}
                processor = module.initialize(config)

                self.plugins[name] = {
                    "metadata": metadata,
                    "processor": processor,
                    "config": config,
                }

                loaded_names.append(name)
                self.logger.info("Loaded text processor plugin: %s", metadata.get("display_name", name))

            except Exception as e:
                fail_name = plugin_dir.name
                reason = str(e)
                failed.append((fail_name, reason))
                self.logger.error(
                    "Failed to load text processor plugin %s (%s): %s\n%s",
                    fail_name, processor_py, e, traceback.format_exc(),
                )

        self.logger.info(
            "Text processor plugin summary: %d loaded [%s] | %d skipped [%s] | %d failed [%s]",
            len(loaded_names), ", ".join(loaded_names) or "none",
            len(skipped_names), ", ".join(skipped_names) or "none",
            len(failed), ", ".join(f"{n} ({r})" for n, r in failed) or "none",
        )
        return self.plugins

    def get_plugin(self, name: str) -> Any | None:
        """Get processor by name."""
        plugin = self.plugins.get(name)
        return plugin["processor"] if plugin else None

    def cleanup_all(self) -> None:
        """Clean up all loaded text processor plugins."""
        for name, plugin_info in self.plugins.items():
            processor = plugin_info.get("processor")
            if processor is None:
                continue
            try:
                if hasattr(processor, "cleanup"):
                    processor.cleanup()
                elif hasattr(processor, "reset"):
                    processor.reset()
            except Exception as e:
                self.logger.error("Error cleaning up text processor %s: %s", name, e)


class OptimizerPluginLoader:
    """Loads and manages optimizer plugins."""

    def __init__(self, plugins_dir: str, config_manager: Any = None) -> None:
        self.plugins_dir = Path(plugins_dir)
        self.plugins: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager

    def load_plugins(self, enable_all: bool = True) -> dict[str, Any]:
        """Load optimizer plugins from directory.

        Args:
            enable_all: If True, load all enabled plugins.
                        If False, only load essential plugins.
        """
        if not self.plugins_dir.exists():
            self.logger.warning("Plugins directory not found: %s", self.plugins_dir)
            return {}

        loaded_names: list[str] = []
        skipped_names: list[str] = []
        failed: list[tuple[str, str]] = []

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_json = plugin_dir / "plugin.json"
            optimizer_py = plugin_dir / "optimizer.py"

            if not plugin_json.exists() or not optimizer_py.exists():
                continue

            try:
                with open(plugin_json, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                name = metadata.get("name", plugin_dir.name)

                if not metadata.get("enabled", True):
                    self.logger.info("Plugin %s is disabled", name)
                    skipped_names.append(name)
                    continue

                is_essential = metadata.get("essential", False)
                if not enable_all and not is_essential:
                    self.logger.info("Plugin %s skipped (not essential)", name)
                    skipped_names.append(name)
                    continue

                self.logger.debug(
                    "Loading optimizer plugin: name=%s version=%s target_stage=%s stage=%s essential=%s",
                    name,
                    metadata.get("version", "?"),
                    metadata.get("target_stage", "?"),
                    metadata.get("stage", "?"),
                    is_essential,
                )

                spec = importlib.util.spec_from_file_location(
                    f"optimizer_{name}", optimizer_py,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                settings = metadata.get("settings", {})
                config = {k: v.get("default") for k, v in settings.items()}

                if self.config_manager:
                    runtime_mode = self.config_manager.get_setting(
                        "performance.runtime_mode", "auto",
                    )
                    config["runtime_mode"] = runtime_mode
                    if name == "frame_skip":
                        config["manga_mode"] = self.config_manager.get_setting(
                            "general.manga_mode", False,
                        )

                optimizer = module.initialize(config)

                self.plugins[name] = {
                    "metadata": metadata,
                    "optimizer": optimizer,
                    "config": config,
                }

                loaded_names.append(name)
                essential_tag = " (essential)" if is_essential else ""
                self.logger.info(
                    "Loaded optimizer plugin: %s%s", metadata.get("display_name", name), essential_tag,
                )

            except Exception as e:
                fail_name = plugin_dir.name
                reason = str(e)
                failed.append((fail_name, reason))
                self.logger.error(
                    "Failed to load optimizer plugin %s (%s): %s\n%s",
                    fail_name, optimizer_py, e, traceback.format_exc(),
                )

        self.logger.info(
            "Optimizer plugin summary: %d loaded [%s] | %d skipped [%s] | %d failed [%s]",
            len(loaded_names), ", ".join(loaded_names) or "none",
            len(skipped_names), ", ".join(skipped_names) or "none",
            len(failed), ", ".join(f"{n} ({r})" for n, r in failed) or "none",
        )
        return self.plugins

    def get_plugin(self, name: str) -> Any | None:
        """Get optimizer by name."""
        plugin = self.plugins.get(name)
        return plugin["optimizer"] if plugin else None

    def cleanup_all(self) -> None:
        """Clean up all loaded optimizer plugins."""
        for name, plugin_info in self.plugins.items():
            optimizer = plugin_info.get("optimizer")
            if optimizer is None:
                continue
            try:
                if hasattr(optimizer, "cleanup"):
                    optimizer.cleanup()
                elif hasattr(optimizer, "reset"):
                    optimizer.reset()
            except Exception as e:
                self.logger.error("Error cleaning up optimizer plugin %s: %s", name, e)
