"""
Translation Plugin Manager

Manages translation engine plugins with discovery, loading, and lifecycle management.
Based on OCR plugin manager architecture.
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Any
import threading
import logging

from app.workflow.base.plugin_interface import PluginMetadata


class TranslationPluginRegistry:
    """Registry for translation engine plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: dict[str, PluginMetadata] = {}
        self._plugin_raw_data: dict[str, dict] = {}  # Store raw plugin.json data
        self._loaded_engines: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("translation.plugin_registry")
    
    def register_plugin_info(self, plugin_info: PluginMetadata, raw_data: dict | None = None) -> bool:
        """
        Register plugin metadata in the registry.
        
        Args:
            plugin_info: Plugin metadata to register
            raw_data: Optional raw plugin.json data (for runtime_requirements, etc.)
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if plugin_info.name in self._plugins:
                self._logger.warning(f"Plugin {plugin_info.name} already registered, updating")
            
            self._plugins[plugin_info.name] = plugin_info
            if raw_data:
                self._plugin_raw_data[plugin_info.name] = raw_data
            self._logger.info(f"Registered plugin: {plugin_info.name} v{plugin_info.version}")
            return True
    
    def get_plugin_info(self, plugin_name: str) -> PluginMetadata | None:
        """Get plugin metadata by name."""
        with self._lock:
            return self._plugins.get(plugin_name)
    
    def get_plugin_raw_data(self, plugin_name: str) -> dict | None:
        """Get raw plugin.json data by name (includes runtime_requirements, etc.)."""
        with self._lock:
            return self._plugin_raw_data.get(plugin_name)
    
    def get_all_plugins(self) -> dict[str, PluginMetadata]:
        """Get all registered plugin metadata."""
        with self._lock:
            return self._plugins.copy()
    
    def register_engine(self, engine: Any) -> bool:
        """
        Register a loaded translation engine.
        
        Args:
            engine: Translation engine instance
            
        Returns:
            True if registration successful
        """
        with self._lock:
            engine_name = getattr(engine, 'engine_name', 'unknown')
            if engine_name in self._loaded_engines:
                self._logger.warning(f"Engine {engine_name} already registered, replacing")
            
            self._loaded_engines[engine_name] = engine
            self._logger.info(f"Registered translation engine: {engine_name}")
            return True
    
    def unregister_engine(self, engine_name: str) -> bool:
        """
        Unregister a translation engine.
        
        Args:
            engine_name: Name of engine to unregister
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if engine_name in self._loaded_engines:
                del self._loaded_engines[engine_name]
                self._logger.info(f"Unregistered translation engine: {engine_name}")
                return True
            return False
    
    def get_engine(self, engine_name: str) -> Any | None:
        """Get specific translation engine by name."""
        with self._lock:
            return self._loaded_engines.get(engine_name)
    
    def get_available_engines(self) -> dict[str, Any]:
        """Get all loaded and ready translation engines."""
        with self._lock:
            return {name: engine for name, engine in self._loaded_engines.items() 
                   if hasattr(engine, 'is_available') and engine.is_available()}
    
    def is_engine_loaded(self, engine_name: str) -> bool:
        """
        Check if a translation engine is loaded and ready.
        
        Args:
            engine_name: Name of engine to check
            
        Returns:
            True if engine is loaded and ready
        """
        with self._lock:
            if engine_name not in self._loaded_engines:
                return False
            engine = self._loaded_engines[engine_name]
            return hasattr(engine, 'is_available') and engine.is_available()


class TranslationPluginManager:
    """Comprehensive translation plugin management system."""
    
    def __init__(self, plugin_directories: list[str] | None = None, 
                 config_manager=None):
        """
        Initialize translation plugin manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
            config_manager: Configuration manager for runtime settings
        """
        self.registry = TranslationPluginRegistry()
        self._plugin_directories = plugin_directories or []
        self.config_manager = config_manager
        self._lock = threading.RLock()
        self._logger = logging.getLogger("translation.plugin_manager")
        
        # Add default plugin directories
        self._add_default_plugin_directories()
    
    def _add_default_plugin_directories(self) -> None:
        """Add default plugin search directories."""
        # Current directory plugins
        current_dir = Path(__file__).parent
        plugin_dirs = [
            str(current_dir.parent.parent / "plugins" / "stages" / "translation"),
            str(Path.home() / ".optikr" / "plugins" / "stages" / "translation")
        ]
        self._plugin_directories.extend(plugin_dirs)
        self._logger.info(f"Plugin directories: {plugin_dirs}")
    
    def discover_plugins(self) -> list[PluginMetadata]:
        """
        Discover available plugins in configured directories.
        
        Returns:
            List of discovered plugin metadata
        """
        discovered_plugins = []
        
        for directory in self._plugin_directories:
            if not os.path.exists(directory):
                continue
            
            self._logger.info(f"Scanning for translation plugins in: {directory}")
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # Check for plugin manifest file
                manifest_path = os.path.join(item_path, "plugin.json")
                if os.path.isdir(item_path) and os.path.exists(manifest_path):
                    try:
                        plugin_info, raw_data = self._load_plugin_manifest(manifest_path, item_path)
                        if plugin_info:
                            discovered_plugins.append(plugin_info)
                            self.registry.register_plugin_info(plugin_info, raw_data)
                    except Exception as e:
                        self._logger.error(f"Failed to load plugin manifest {manifest_path}: {e}")
        
        self._logger.info(f"Discovered {len(discovered_plugins)} translation plugins")
        return discovered_plugins
    
    def _load_plugin_manifest(self, manifest_path: str, plugin_path: str) -> tuple[PluginMetadata | None, dict | None]:
        """
        Load plugin manifest from JSON file.
        
        Args:
            manifest_path: Path to plugin.json manifest file
            plugin_path: Path to plugin directory
            
        Returns:
            Tuple of (Plugin metadata, raw JSON data) or (None, None) if invalid
        """
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Validate required fields
            required_fields = ["name", "version", "description", "author", "type"]
            for field in required_fields:
                if field not in manifest_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create plugin metadata
            plugin_info = PluginMetadata.from_dict(manifest_data)
            
            return plugin_info, manifest_data
            
        except Exception as e:
            self._logger.error(f"Failed to parse plugin manifest {manifest_path}: {e}")
            return None, None

    
    def load_plugin(self, plugin_name: str, config: dict[str, Any] | None = None) -> bool:
        """
        Load and initialize a translation plugin.
        
        Args:
            plugin_name: Name of plugin to load
            config: Optional configuration for plugin initialization
            
        Returns:
            True if plugin loaded successfully
        """
        with self._lock:
            plugin_info = self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                self._logger.error(f"Plugin {plugin_name} not found in registry")
                return False
            
            if self.registry.is_engine_loaded(plugin_name):
                self._logger.info(f"Plugin {plugin_name} already loaded")
                return True
            
            try:
                self._logger.info(f"Loading translation plugin: {plugin_name}")
                
                # Get plugin path
                plugin_path = self._get_plugin_path(plugin_name)
                if not plugin_path:
                    raise FileNotFoundError(f"Plugin directory not found for {plugin_name}")
                
                # Load plugin module
                plugin_module = self._load_plugin_module(plugin_info, plugin_path)
                if not plugin_module:
                    raise ImportError(f"Failed to load plugin module for {plugin_name}")
                
                # Get engine class from module
                engine_class = getattr(plugin_module, "TranslationEngine", None)
                if not engine_class:
                    raise AttributeError(f"Plugin {plugin_name} does not define TranslationEngine class")
                
                # Create engine instance
                engine = engine_class()
                
                # Initialize engine with config
                plugin_config = config or {}
                
                # Apply config from config_manager if available
                if self.config_manager:
                    src_lang = self.config_manager.get_setting('translation.source_language', 'en')
                    tgt_lang = self.config_manager.get_setting('translation.target_language', 'de')
                    runtime_mode = self.config_manager.get_setting('performance.runtime_mode', 'auto')
                    
                    plugin_config.setdefault('source_language', src_lang)
                    plugin_config.setdefault('target_language', tgt_lang)
                    plugin_config.setdefault('runtime_mode', runtime_mode)

                    # Pass through Qwen3-specific prompt template when available
                    if plugin_name == "qwen3":
                        qwen_tpl = self.config_manager.get_setting(
                            'translation.qwen_prompt_template',
                            '',
                        )
                        if qwen_tpl:
                            plugin_config.setdefault('prompt_template', qwen_tpl)
                    
                    # Determine GPU usage based on runtime mode
                    try:
                        import torch
                        if runtime_mode == 'cpu':
                            use_gpu = False
                        elif runtime_mode == 'gpu':
                            use_gpu = torch.cuda.is_available()
                        else:  # 'auto'
                            use_gpu = torch.cuda.is_available()
                        
                        plugin_config.setdefault('gpu', use_gpu)
                        self._logger.info(f"Translation plugin config: runtime_mode={runtime_mode}, gpu={use_gpu}")
                    except ImportError:
                        plugin_config.setdefault('gpu', False)
                        self._logger.warning("PyTorch not available, defaulting to CPU")
                
                # Initialize engine
                if hasattr(engine, 'initialize'):
                    init_success = engine.initialize(plugin_config)
                    if not init_success:
                        raise RuntimeError(f"Failed to initialize plugin {plugin_name}")
                
                # Register engine
                self.registry.register_engine(engine)
                
                self._logger.info(f"Successfully loaded translation plugin: {plugin_name}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to load plugin {plugin_name}: {e}", exc_info=True)
                return False
    
    def _get_plugin_path(self, plugin_name: str) -> Path | None:
        """
        Get path to plugin directory.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Path to plugin directory or None if not found
        """
        self._logger.debug(f"Searching for plugin '{plugin_name}' in {len(self._plugin_directories)} directories")
        for directory in self._plugin_directories:
            plugin_path = Path(directory) / plugin_name
            self._logger.debug(f"  Checking: {plugin_path} - exists: {plugin_path.exists()}")
            if plugin_path.exists():
                self._logger.debug(f"  Found plugin at: {plugin_path}")
                return plugin_path
        self._logger.warning(f"Plugin '{plugin_name}' not found in any directory")
        return None
    
    def _load_plugin_module(self, plugin_info: PluginMetadata, plugin_path: Path):
        """
        Load plugin module from file system.

        Args:
            plugin_info: Plugin metadata
            plugin_path: Path to plugin directory

        Returns:
            Loaded module or None if failed
        """
        try:
            # Find main plugin file
            self._logger.debug(f"Looking for plugin files in: {plugin_path}")

            main_file = plugin_path / "__init__.py"

            if not main_file.exists():
                # Try base name without _gpu/_cpu suffix (e.g., marianmt_engine.py for marianmt_gpu)
                base_name = plugin_info.name.replace('_gpu', '').replace('_cpu', '')
                main_file = plugin_path / f"{base_name}_engine.py"

            if not main_file.exists():
                # Try full plugin name
                main_file = plugin_path / f"{plugin_info.name}_engine.py"

            if not main_file.exists():
                # Try generic engine.py
                main_file = plugin_path / "engine.py"

            if not main_file.exists():
                # Try the worker_script declared in plugin.json (e.g. worker.py)
                worker = getattr(plugin_info, 'worker_script', None)
                if worker:
                    candidate = plugin_path / worker
                    if candidate.exists():
                        main_file = candidate

            if not main_file.exists():
                # List what files actually exist
                if plugin_path.exists():
                    files = list(plugin_path.glob("*.py"))
                    self._logger.error(f"Available Python files in {plugin_path}: {[f.name for f in files]}")
                raise FileNotFoundError(f"Plugin main file not found for {plugin_info.name}")

            # Load module
            spec = importlib.util.spec_from_file_location(plugin_info.name, main_file)
            if not spec or not spec.loader:
                raise ImportError(f"Failed to create module spec for {plugin_info.name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"translation_plugin_{plugin_info.name}"] = module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            self._logger.error(f"Failed to load module for plugin {plugin_info.name}: {e}")
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a translation plugin and clean up its resources.

        Calls ``cleanup()`` on the engine (which releases GPU memory for
        PyTorch-based engines) and removes it from the plugin registry.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if plugin unloaded successfully
        """
        with self._lock:
            engine = self.registry.get_engine(plugin_name)
            if not engine:
                self._logger.warning(
                    "Translation plugin %s not loaded, nothing to unload",
                    plugin_name,
                )
                return False

            try:
                engine.cleanup()
                self.registry.unregister_engine(plugin_name)

                module_name = f"translation_plugin_{plugin_name}"
                if module_name in sys.modules:
                    del sys.modules[module_name]

                self._logger.info("Successfully unloaded translation plugin: %s", plugin_name)
                return True
            except Exception as e:
                self._logger.error("Failed to unload plugin %s: %s", plugin_name, e)
                return False

    def get_available_engines(self) -> list[str]:
        """Get list of available translation engine names."""
        return list(self.registry.get_available_engines().keys())
    
    def get_engine(self, engine_name: str) -> Any | None:
        """Get translation engine by name."""
        return self.registry.get_engine(engine_name)
