"""
LLM Plugin Management System

This module provides plugin discovery, loading, registration, and lifecycle
management for LLM engines.  It follows the same patterns established by
:mod:`app.ocr.ocr_plugin_manager`.
"""

import os
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Any
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum

from .llm_engine_interface import ILLMEngine, LLMEnginePlugin, LLMEngineType


class PluginLoadStatus(Enum):
    """Plugin loading status enumeration."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class LLMPluginInfo:
    """Plugin information and metadata for an LLM plugin."""
    name: str
    version: str
    description: str
    author: str
    engine_type: LLMEngineType
    plugin_path: str
    entry_point: str = "__init__.py"
    config_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    supported_platforms: list[str] = field(default_factory=lambda: ["windows", "linux", "macos"])
    min_python_version: str = "3.8"
    load_status: PluginLoadStatus = PluginLoadStatus.NOT_LOADED
    load_error: str | None = None


class LLMPluginRegistry:
    """Registry for managing LLM engine plugins."""

    def __init__(self):
        self._plugins: dict[str, LLMPluginInfo] = {}
        self._loaded_engines: dict[str, ILLMEngine] = {}
        self._plugin_instances: dict[str, LLMEnginePlugin] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("llm.plugin_registry")

    def register_plugin_info(self, plugin_info: LLMPluginInfo) -> bool:
        with self._lock:
            if plugin_info.name in self._plugins:
                self._logger.warning(f"Plugin {plugin_info.name} already registered, updating info")
            self._plugins[plugin_info.name] = plugin_info
            self._logger.info(f"Registered plugin info: {plugin_info.name} v{plugin_info.version}")
            return True

    def get_plugin_info(self, plugin_name: str) -> LLMPluginInfo | None:
        with self._lock:
            return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> dict[str, LLMPluginInfo]:
        with self._lock:
            return self._plugins.copy()

    def get_loaded_plugins(self) -> dict[str, LLMPluginInfo]:
        with self._lock:
            return {
                name: info
                for name, info in self._plugins.items()
                if info.load_status == PluginLoadStatus.LOADED
            }

    def get_available_engines(self) -> dict[str, ILLMEngine]:
        with self._lock:
            return {
                name: engine
                for name, engine in self._loaded_engines.items()
                if engine.is_ready()
            }

    def get_engine(self, engine_name: str) -> ILLMEngine | None:
        with self._lock:
            return self._loaded_engines.get(engine_name)

    def is_engine_loaded(self, engine_name: str) -> bool:
        with self._lock:
            plugin_info = self._plugins.get(engine_name)
            if not plugin_info:
                return False
            return plugin_info.load_status == PluginLoadStatus.LOADED

    def register_engine(self, engine: ILLMEngine) -> bool:
        with self._lock:
            if engine.engine_name in self._loaded_engines:
                self._logger.warning(f"Engine {engine.engine_name} already registered, replacing")
            self._loaded_engines[engine.engine_name] = engine
            self._logger.info(f"Registered LLM engine: {engine.engine_name}")
            return True

    def unregister_engine(self, engine_name: str) -> bool:
        with self._lock:
            if engine_name in self._loaded_engines:
                self._loaded_engines.pop(engine_name)
                self._logger.info(f"Unregistered LLM engine: {engine_name}")
                return True
            return False


class LLMPluginManager:
    """Comprehensive LLM plugin management system."""

    PACKAGE_IMPORT_MAP: dict[str, str] = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
    }

    def __init__(
        self,
        plugin_directories: list[str] | None = None,
        config_manager=None,
    ):
        """
        Args:
            plugin_directories: Directories to search for plugins
            config_manager: Configuration manager for runtime settings
        """
        self.registry = LLMPluginRegistry()
        self._plugin_directories = plugin_directories or []
        self.config_manager = config_manager
        self._lock = threading.RLock()
        self._logger = logging.getLogger("llm.plugin_manager")

        self._add_default_plugin_directories()

    def _add_default_plugin_directories(self) -> None:
        current_dir = Path(__file__).parent
        self._plugin_directories.extend([
            str(current_dir.parent.parent / "plugins" / "stages" / "llm"),
            str(current_dir / "plugins"),
            str(current_dir / "engines"),
        ])
        user_plugins = Path.home() / ".translation_system" / "plugins" / "stages" / "llm"
        self._plugin_directories.append(str(user_plugins))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self) -> list[LLMPluginInfo]:
        """Discover available LLM plugins whose dependencies are installed."""
        discovered: list[LLMPluginInfo] = []

        for directory in self._plugin_directories:
            if not os.path.exists(directory):
                continue

            self._logger.info(f"Scanning for LLM plugins in: {directory}")

            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                manifest_path = os.path.join(item_path, "plugin.json")

                if os.path.isdir(item_path) and os.path.exists(manifest_path):
                    try:
                        plugin_info = self._load_plugin_manifest(manifest_path, item_path)
                        if plugin_info:
                            if self._check_plugin_dependencies(plugin_info):
                                discovered.append(plugin_info)
                                self.registry.register_plugin_info(plugin_info)
                            else:
                                self._logger.info(
                                    f"Skipping LLM plugin {plugin_info.name} "
                                    "- dependencies not installed"
                                )
                    except Exception as e:
                        self._logger.error(
                            f"Failed to load plugin manifest {manifest_path}: {e}"
                        )

        self._logger.info(
            f"Discovered {len(discovered)} LLM plugins with installed dependencies"
        )
        return discovered

    def _check_plugin_dependencies(self, plugin_info: LLMPluginInfo) -> bool:
        if not plugin_info.dependencies:
            return True

        for dependency in plugin_info.dependencies:
            import_name = self.PACKAGE_IMPORT_MAP.get(dependency, dependency)
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                self._logger.debug(
                    f"Dependency {dependency} (import: {import_name}) "
                    f"not found for plugin {plugin_info.name}"
                )
                return False
        return True

    def _load_plugin_manifest(
        self, manifest_path: str, plugin_path: str
    ) -> LLMPluginInfo | None:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_fields = [
                "name", "version", "description", "author",
                "engine_type", "entry_point",
            ]
            for fld in required_fields:
                if fld not in data:
                    raise ValueError(f"Missing required field: {fld}")

            return LLMPluginInfo(
                name=data["name"],
                version=data["version"],
                description=data["description"],
                author=data["author"],
                engine_type=LLMEngineType(data["engine_type"]),
                plugin_path=plugin_path,
                entry_point=data.get("entry_point", "__init__.py"),
                config_schema=data.get("config_schema", {}),
                dependencies=data.get("dependencies", []),
                supported_platforms=data.get(
                    "supported_platforms", ["windows", "linux", "macos"]
                ),
                min_python_version=data.get("min_python_version", "3.8"),
            )

        except Exception as e:
            self._logger.error(
                f"Failed to parse LLM plugin manifest {manifest_path}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load_plugin(
        self, plugin_name: str, config: dict[str, Any] | None = None
    ) -> bool:
        with self._lock:
            plugin_info = self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                self._logger.error(f"Plugin {plugin_name} not found in registry")
                return False

            if plugin_info.load_status == PluginLoadStatus.LOADED:
                self._logger.info(f"Plugin {plugin_name} already loaded")
                return True

            try:
                plugin_info.load_status = PluginLoadStatus.LOADING

                plugin_module = self._load_plugin_module(plugin_info)
                if not plugin_module:
                    raise ImportError(
                        f"Failed to load plugin module for {plugin_name}"
                    )

                engine_class = getattr(plugin_module, "LLMEngine", None)
                if not engine_class:
                    raise AttributeError(
                        f"Plugin {plugin_name} does not define LLMEngine class"
                    )

                plugin_instance = LLMEnginePlugin(engine_class, plugin_info.__dict__)

                plugin_config = config or {}

                use_gpu = self._should_use_gpu()
                plugin_config["gpu"] = use_gpu
                plugin_config["use_gpu"] = use_gpu
                self._logger.info(f"Configuring {plugin_name} with GPU={use_gpu}")

                init_success = plugin_instance.initialize(plugin_config)

                if not init_success and use_gpu:
                    self._logger.warning(
                        f"GPU init failed for {plugin_name}, falling back to CPU"
                    )
                    plugin_config["gpu"] = False
                    plugin_config["use_gpu"] = False
                    init_success = plugin_instance.initialize(plugin_config)

                if not init_success:
                    raise RuntimeError(
                        f"Failed to initialize plugin {plugin_name} "
                        "(tried both GPU and CPU modes)"
                    )

                self.registry._plugin_instances[plugin_name] = plugin_instance
                engine = plugin_instance.get_engine_instance()
                if engine:
                    self.registry.register_engine(engine)

                plugin_info.load_status = PluginLoadStatus.LOADED
                plugin_info.load_error = None

                self._logger.info(f"Successfully loaded LLM plugin: {plugin_name}")
                return True

            except Exception as e:
                plugin_info.load_status = PluginLoadStatus.FAILED
                plugin_info.load_error = str(e)
                self._logger.error(f"Failed to load LLM plugin {plugin_name}: {e}")
                return False

    def _load_plugin_module(self, plugin_info: LLMPluginInfo):
        try:
            main_file = os.path.join(plugin_info.plugin_path, plugin_info.entry_point)
            if not os.path.exists(main_file):
                main_file = os.path.join(plugin_info.plugin_path, "__init__.py")
            if not os.path.exists(main_file):
                main_file = os.path.join(
                    plugin_info.plugin_path, f"{plugin_info.name}.py"
                )

            if not os.path.exists(main_file):
                raise FileNotFoundError(
                    f"Plugin main file not found for {plugin_info.name}"
                )

            spec = importlib.util.spec_from_file_location(plugin_info.name, main_file)
            if not spec or not spec.loader:
                raise ImportError(
                    f"Failed to create module spec for {plugin_info.name}"
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"llm_plugin_{plugin_info.name}"] = module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            self._logger.error(
                f"Failed to load module for LLM plugin {plugin_info.name}: {e}"
            )
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        with self._lock:
            plugin_info = self.registry.get_plugin_info(plugin_name)
            if not plugin_info or plugin_info.load_status != PluginLoadStatus.LOADED:
                return False

            try:
                plugin_instance = self.registry._plugin_instances.get(plugin_name)
                if plugin_instance:
                    engine = plugin_instance.get_engine_instance()
                    if engine:
                        self.registry.unregister_engine(engine.engine_name)
                    plugin_instance.cleanup()
                    del self.registry._plugin_instances[plugin_name]

                plugin_info.load_status = PluginLoadStatus.NOT_LOADED

                module_name = f"llm_plugin_{plugin_name}"
                if module_name in sys.modules:
                    del sys.modules[module_name]

                self._logger.info(f"Successfully unloaded LLM plugin: {plugin_name}")
                return True

            except Exception as e:
                self._logger.error(f"Failed to unload LLM plugin {plugin_name}: {e}")
                return False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_loaded_plugins(self) -> list[str]:
        return [
            name
            for name, info in self.registry.get_all_plugins().items()
            if info.load_status == PluginLoadStatus.LOADED
        ]

    def get_plugin_info(self, plugin_name: str) -> dict[str, Any]:
        plugin_info = self.registry.get_plugin_info(plugin_name)
        return plugin_info.__dict__ if plugin_info else {}

    def get_available_engines(self) -> list[str]:
        return list(self.registry.get_available_engines().keys())

    def get_engine(self, engine_name: str) -> ILLMEngine | None:
        return self.registry.get_engine(engine_name)

    def _should_use_gpu(self) -> bool:
        if not self.config_manager:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                self._logger.info(f"No config manager, auto-detecting GPU: {gpu_available}")
                return gpu_available
            except ImportError:
                self._logger.info("No config manager and PyTorch not available, using CPU")
                return False

        runtime_mode = self.config_manager.get_setting("performance.runtime_mode", "auto")
        self._logger.info(f"Runtime mode from config: {runtime_mode}")

        if runtime_mode == "cpu":
            return False
        elif runtime_mode == "gpu":
            return True
        else:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                self._logger.info(f"Runtime mode Auto, GPU available: {gpu_available}")
                return gpu_available
            except ImportError:
                self._logger.info("Runtime mode Auto, PyTorch not available, using CPU")
                return False

    def load_all_plugins(
        self, config: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        results = {}
        for plugin_name in self.registry.get_all_plugins():
            results[plugin_name] = self.load_plugin(plugin_name, config)
        return results
