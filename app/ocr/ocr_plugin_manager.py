"""
OCR Plugin Management System

This module provides comprehensive plugin management for OCR engines,
including plugin discovery, loading, registration, and lifecycle management.
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

from .ocr_engine_interface import IOCREngine, OCREnginePlugin, OCREngineType

_PACKAGE_IMPORT_MAP = {
    'easyocr': 'easyocr',
    'pytesseract': 'pytesseract',
    'tesseract-ocr': 'pytesseract',
    'paddleocr': 'paddleocr',
    'paddlepaddle': 'paddle',
    'mokuro': 'mokuro',
    'manga_ocr': 'manga_ocr',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'transformers': 'transformers',
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'surya-ocr': 'surya',
    'python-doctr[torch]': 'doctr',
    'python-doctr': 'doctr',
    'rapidocr-onnxruntime': 'rapidocr_onnxruntime',
    'winocr': 'winocr',
}


class PluginLoadStatus(Enum):
    """Plugin loading status enumeration."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Plugin information and metadata."""
    name: str
    version: str
    description: str
    author: str
    engine_type: OCREngineType
    plugin_path: str
    config_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    optional_dependencies: list[str] = field(default_factory=list)
    supported_platforms: list[str] = field(default_factory=lambda: ["windows", "linux", "macos"])
    min_python_version: str = "3.8"
    load_status: PluginLoadStatus = PluginLoadStatus.NOT_LOADED
    load_error: str | None = None


class OCRPluginRegistry:
    """Registry for managing OCR engine plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: dict[str, PluginInfo] = {}
        self._loaded_engines: dict[str, IOCREngine] = {}
        self._plugin_instances: dict[str, OCREnginePlugin] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ocr.plugin_registry")
    
    def register_plugin_info(self, plugin_info: PluginInfo) -> bool:
        """
        Register plugin information in the registry.
        
        Args:
            plugin_info: Plugin information to register
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if plugin_info.name in self._plugins:
                self._logger.warning(f"Plugin {plugin_info.name} already registered, updating info")
            
            self._plugins[plugin_info.name] = plugin_info
            self._logger.info(f"Registered plugin info: {plugin_info.name} v{plugin_info.version}")
            return True
    
    def get_plugin_info(self, plugin_name: str) -> PluginInfo | None:
        """Get plugin information by name."""
        with self._lock:
            return self._plugins.get(plugin_name)
    
    def get_all_plugins(self) -> dict[str, PluginInfo]:
        """Get all registered plugin information."""
        with self._lock:
            return self._plugins.copy()
    
    def get_loaded_plugins(self) -> dict[str, PluginInfo]:
        """Get information for loaded plugins only."""
        with self._lock:
            return {name: info for name, info in self._plugins.items() 
                   if info.load_status == PluginLoadStatus.LOADED}
    
    def get_available_engines(self) -> dict[str, IOCREngine]:
        """Get all loaded and ready OCR engines."""
        with self._lock:
            return {name: engine for name, engine in self._loaded_engines.items() 
                   if engine.is_ready()}
    
    def get_engine(self, engine_name: str) -> IOCREngine | None:
        """Get specific OCR engine by name."""
        with self._lock:
            return self._loaded_engines.get(engine_name)
    
    def is_engine_loaded(self, engine_name: str) -> bool:
        """
        Check if an OCR engine is loaded and ready.
        
        Args:
            engine_name: Name of engine to check
            
        Returns:
            True if engine is loaded and ready
        """
        with self._lock:
            plugin_info = self._plugins.get(engine_name)
            if not plugin_info:
                return False
            return plugin_info.load_status == PluginLoadStatus.LOADED
    
    def register_engine(self, engine: IOCREngine) -> bool:
        """
        Register a loaded OCR engine.
        
        Args:
            engine: OCR engine instance to register
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if engine.engine_name in self._loaded_engines:
                self._logger.warning(f"Engine {engine.engine_name} already registered, replacing")
            
            self._loaded_engines[engine.engine_name] = engine
            self._logger.info(f"Registered OCR engine: {engine.engine_name}")
            return True
    
    def unregister_engine(self, engine_name: str) -> bool:
        """
        Unregister an OCR engine.
        
        Args:
            engine_name: Name of engine to unregister
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if engine_name in self._loaded_engines:
                engine = self._loaded_engines.pop(engine_name)
                self._logger.info(f"Unregistered OCR engine: {engine_name}")
                return True
            return False
    

class OCRPluginManager:
    """Comprehensive OCR plugin management system."""

    DEFAULT_VISIBLE_PLUGINS = {
        "easyocr",
        "mokuro",
        "judge_ocr",
        "hybrid_ocr",
    }
    
    def __init__(self, plugin_directories: list[str] | None = None, config_manager=None):
        """
        Initialize OCR plugin manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
            config_manager: Configuration manager for runtime settings
        """
        self.registry = OCRPluginRegistry()
        self._plugin_directories = plugin_directories or []
        self.config_manager = config_manager
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ocr.plugin_manager")
        
        # Add default plugin directories
        self._add_default_plugin_directories()
    
    def _add_default_plugin_directories(self) -> None:
        """Add default plugin search directories."""
        default_dirs = _get_default_ocr_plugin_directories()
        self._plugin_directories.extend(default_dirs)
        self._plugin_directories = _dedupe_paths(self._plugin_directories)
        self._logger.info("OCR plugin search directories: %s", self._plugin_directories)
    
    def discover_plugins(self) -> list[PluginInfo]:
        """
        Discover available plugins in configured directories.
        Only includes plugins whose dependencies are actually installed.
        Auto-generates missing plugin folders for installed packages.
        
        Returns:
            List of discovered plugin information
        """
        discovered_plugins = []
        
        auto_generate = False
        if self.config_manager:
            auto_generate = bool(self.config_manager.get_setting("ocr.auto_generate_plugins", False))
        if auto_generate:
            self._auto_generate_missing_plugins()
        
        for directory in self._plugin_directories:
            if not os.path.exists(directory):
                continue
            
            self._logger.info(f"Scanning for plugins in: {directory}")
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # Check for plugin manifest file
                manifest_path = os.path.join(item_path, "plugin.json")
                if os.path.isdir(item_path) and os.path.exists(manifest_path):
                    try:
                        plugin_info = self._load_plugin_manifest(manifest_path, item_path)
                        if plugin_info:
                            # Check if plugin dependencies are actually installed
                            if self._check_plugin_dependencies(plugin_info):
                                discovered_plugins.append(plugin_info)
                                self.registry.register_plugin_info(plugin_info)
                            else:
                                missing = self._get_missing_required_dependencies(plugin_info)
                                if missing:
                                    self._logger.info(
                                        "Skipping plugin %s - missing dependencies: %s",
                                        plugin_info.name,
                                        ", ".join(missing),
                                    )
                                else:
                                    self._logger.info(
                                        "Skipping plugin %s - dependencies not installed",
                                        plugin_info.name,
                                    )
                    except Exception as e:
                        self._logger.error(f"Failed to load plugin manifest {manifest_path}: {e}")
        
        self._logger.info(f"Discovered {len(discovered_plugins)} OCR plugins with installed dependencies")
        return discovered_plugins

    def _get_missing_required_dependencies(self, plugin_info: PluginInfo) -> list[str]:
        """Return required dependencies that are currently missing."""
        if not plugin_info.dependencies:
            return []

        optional_deps = set(plugin_info.optional_dependencies or [])
        missing: list[str] = []
        for dependency in plugin_info.dependencies:
            if dependency in optional_deps:
                continue
            import_name = _PACKAGE_IMPORT_MAP.get(dependency, dependency)
            if importlib.util.find_spec(import_name) is None:
                missing.append(dependency)
        return missing

    def _get_visible_plugin_names(self) -> set[str]:
        """Return OCR plugin names that should be visible to users."""
        names = set(self.DEFAULT_VISIBLE_PLUGINS)
        if not self.config_manager:
            return names

        configured = self.config_manager.get_setting("ocr.visible_plugins", None)
        if isinstance(configured, list):
            normalized = {str(item).strip().lower() for item in configured if str(item).strip()}
            if normalized:
                names |= normalized

        selected_engine = self.config_manager.get_setting("ocr.engine", "")
        if selected_engine:
            names.add(str(selected_engine).strip().lower())

        return names

    def _is_plugin_visible(self, plugin_name: str) -> bool:
        """Check whether a plugin should be listed in UI/discovery."""
        return plugin_name.lower() in self._get_visible_plugin_names()
    
    def _auto_generate_missing_plugins(self):
        """
        Auto-generate plugin folders for installed OCR packages that don't have plugins yet.
        Uses the universal plugin generator to create the necessary files.
        Skips packages that already have a plugin (even with different names).
        """
        try:
            # Use the universal plugin generator from workflow
            from app.workflow.universal_plugin_generator import PluginGenerator
            
            generator = PluginGenerator(output_dir="plugins")
            
            # Check which OCR packages are installed
            installed_engines = []
            
            # Check for each known engine
            engine_checks = {
                'mokuro': 'mokuro',
                'paddleocr': 'paddleocr',
                'tesseract': 'pytesseract',
                'easyocr': 'easyocr'
            }
            
            for engine_name, import_name in engine_checks.items():
                spec = importlib.util.find_spec(import_name)
                if spec is not None:
                    installed_engines.append((engine_name, import_name))
            
            # Get the main plugins directory (dev/plugins/stages/ocr)
            main_plugin_dir = None
            for directory in self._plugin_directories:
                if 'plugins' in directory and 'ocr' in directory and 'src' not in directory:
                    main_plugin_dir = Path(directory)
                    break
            
            if not main_plugin_dir:
                return
            
            # First, scan existing plugins to see which packages they use
            existing_packages = set()
            if main_plugin_dir.exists():
                for item in main_plugin_dir.iterdir():
                    if item.is_dir():
                        plugin_json = item / "plugin.json"
                        if plugin_json.exists():
                            try:
                                with open(plugin_json, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    # Track which packages are used by existing plugins
                                    deps = data.get('dependencies', [])
                                    for dep in deps:
                                        # Normalize package names
                                        if dep in ['easyocr', 'torch', 'torchvision']:
                                            existing_packages.add('easyocr')
                                        elif dep in ['mokuro']:
                                            existing_packages.add('mokuro')
                                        elif dep in ['paddleocr', 'paddlepaddle']:
                                            existing_packages.add('paddleocr')
                                        elif dep in ['pytesseract', 'tesseract-ocr']:
                                            existing_packages.add('tesseract')
                            except (json.JSONDecodeError, OSError):
                                pass
            
            # For each installed engine, check if we should create a plugin
            for engine_name, import_name in installed_engines:
                # Skip if a plugin already exists for this package
                if engine_name in existing_packages:
                    self._logger.debug(f"Skipping {engine_name} - plugin already exists")
                    continue
                
                plugin_folder = main_plugin_dir / engine_name
                plugin_json_path = plugin_folder / "plugin.json"
                
                # If plugin folder doesn't exist or is incomplete, create it
                if not plugin_json_path.exists():
                    self._logger.info(f"Auto-generating plugin for installed package: {engine_name}")
                    
                    # Use universal generator to create plugin
                    success = generator.create_plugin_programmatically(
                        plugin_type='ocr',
                        name=engine_name,
                        display_name=engine_name.replace('_', ' ').title(),
                        description=f"OCR engine using {engine_name} library",
                        dependencies=[import_name],
                        settings={}
                    )
                    
                    if success:
                        self._logger.info(f"✓ Auto-generated plugin for {engine_name}")
                    else:
                        self._logger.warning(f"Failed to auto-generate plugin for {engine_name}")
        
        except Exception as e:
            self._logger.warning(f"Failed to auto-generate plugins: {e}")
    
    def _check_plugin_dependencies(self, plugin_info: PluginInfo) -> bool:
        """
        Check if all plugin dependencies are installed.
        
        Args:
            plugin_info: Plugin information with dependencies list
            
        Returns:
            True if all dependencies are installed, False otherwise
        """
        if not plugin_info.dependencies:
            return True
        
        optional_deps = set(plugin_info.optional_dependencies or [])

        for dependency in plugin_info.dependencies:
            if dependency in optional_deps:
                continue
            # Get the import name for this dependency
            import_name = _PACKAGE_IMPORT_MAP.get(dependency, dependency)
            
            # Check if the module can be imported
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                self._logger.debug(f"Dependency {dependency} (import: {import_name}) not found for plugin {plugin_info.name}")
                return False
        
        return True
    
    def _load_plugin_manifest(self, manifest_path: str, plugin_path: str) -> PluginInfo | None:
        """
        Load plugin manifest from JSON file.
        
        Args:
            manifest_path: Path to plugin.json manifest file
            plugin_path: Path to plugin directory
            
        Returns:
            Plugin information or None if invalid
        """
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Validate required fields.
            # NOTE: entry_point is optional in many legacy OCR plugin manifests.
            required_fields = ["name", "version", "description", "author", "engine_type"]
            for field in required_fields:
                if field not in manifest_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create plugin info
            plugin_info = PluginInfo(
                name=manifest_data["name"],
                version=manifest_data["version"],
                description=manifest_data["description"],
                author=manifest_data["author"],
                engine_type=OCREngineType(manifest_data["engine_type"]),
                plugin_path=plugin_path,
                config_schema=manifest_data.get("config_schema", {}),
                dependencies=manifest_data.get("dependencies", []),
                optional_dependencies=manifest_data.get("optional_dependencies", []),
                supported_platforms=manifest_data.get("supported_platforms", ["windows", "linux", "macos"]),
                min_python_version=manifest_data.get("min_python_version", "3.8")
            )
            
            return plugin_info
            
        except Exception as e:
            self._logger.error(f"Failed to parse plugin manifest {manifest_path}: {e}")
            return None
    
    def load_plugin(self, plugin_name: str, config: dict[str, Any] | None = None) -> bool:
        """
        Load and initialize a plugin.
        
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
            
            if plugin_info.load_status == PluginLoadStatus.LOADED:
                self._logger.info(f"Plugin {plugin_name} already loaded")
                return True
            
            try:
                plugin_info.load_status = PluginLoadStatus.LOADING
                
                # Load plugin module
                plugin_module = self._load_plugin_module(plugin_info)
                if not plugin_module:
                    raise ImportError(f"Failed to load plugin module for {plugin_name}")
                
                # Get engine class from module
                engine_class = getattr(plugin_module, "OCREngine", None)
                if not engine_class:
                    raise AttributeError(f"Plugin {plugin_name} does not define OCREngine class")
                
                # Create plugin instance
                plugin_instance = OCREnginePlugin(engine_class, plugin_info.__dict__)
                
                # Initialize plugin with engine-specific config
                plugin_config = config or {}
                
                # Apply GPU configuration based on runtime mode
                use_gpu = self._should_use_gpu()
                plugin_config['gpu'] = use_gpu
                plugin_config['use_gpu'] = use_gpu  # Some engines use 'use_gpu' instead of 'gpu'
                self._logger.info(f"Configuring {plugin_name} with GPU={use_gpu}")
                
                # Apply engine-specific language code if available
                if hasattr(self, '_engine_languages') and plugin_name in self._engine_languages:
                    plugin_config['language'] = self._engine_languages[plugin_name]
                    self._logger.info(f"Using engine-specific language for {plugin_name}: {plugin_config['language']}")
                
                # Try to initialize with GPU first, fallback to CPU if it fails
                init_success = plugin_instance.initialize(plugin_config)
                
                if not init_success and use_gpu:
                    # GPU initialization failed, try CPU mode
                    self._logger.warning(f"GPU initialization failed for {plugin_name}, falling back to CPU mode")
                    plugin_config['gpu'] = False
                    plugin_config['use_gpu'] = False
                    init_success = plugin_instance.initialize(plugin_config)
                
                if not init_success:
                    raise RuntimeError(f"Failed to initialize plugin {plugin_name} (tried both GPU and CPU modes)")
                
                # Register plugin and engine
                self.registry._plugin_instances[plugin_name] = plugin_instance
                engine = plugin_instance.get_engine_instance()
                if engine:
                    self.registry.register_engine(engine)
                
                plugin_info.load_status = PluginLoadStatus.LOADED
                plugin_info.load_error = None
                
                self._logger.info(f"Successfully loaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                plugin_info.load_status = PluginLoadStatus.FAILED
                plugin_info.load_error = str(e)
                self._logger.error(f"Failed to load plugin {plugin_name}: {e}")
                return False
    
    def _load_plugin_module(self, plugin_info: PluginInfo):
        """
        Load plugin module from file system.
        
        Args:
            plugin_info: Plugin information
            
        Returns:
            Loaded module or None if failed
        """
        try:
            # Find main plugin file
            main_file = os.path.join(plugin_info.plugin_path, "__init__.py")
            if not os.path.exists(main_file):
                main_file = os.path.join(plugin_info.plugin_path, f"{plugin_info.name}.py")
            if not os.path.exists(main_file):
                main_file = os.path.join(plugin_info.plugin_path, "worker.py")
            
            if not os.path.exists(main_file):
                raise FileNotFoundError(f"Plugin main file not found for {plugin_info.name}")
            
            # Load module. For package-style plugins (__init__.py with relative
            # imports), set submodule_search_locations so ".foo" imports work.
            module_name = f"ocr_plugin_{plugin_info.name}"
            if os.path.basename(main_file) == "__init__.py":
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    main_file,
                    submodule_search_locations=[plugin_info.plugin_path],
                )
            else:
                spec = importlib.util.spec_from_file_location(module_name, main_file)
            if not spec or not spec.loader:
                raise ImportError(f"Failed to create module spec for {plugin_info.name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            self._logger.error(f"Failed to load module for plugin {plugin_info.name}: {e}")
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin and clean up resources.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if plugin unloaded successfully
        """
        with self._lock:
            plugin_info = self.registry.get_plugin_info(plugin_name)
            if not plugin_info or plugin_info.load_status != PluginLoadStatus.LOADED:
                return False
            
            try:
                # Get plugin instance
                plugin_instance = self.registry._plugin_instances.get(plugin_name)
                if plugin_instance:
                    # Unregister engine
                    engine = plugin_instance.get_engine_instance()
                    if engine:
                        self.registry.unregister_engine(engine.engine_name)
                    
                    # Cleanup plugin
                    plugin_instance.cleanup()
                    del self.registry._plugin_instances[plugin_name]
                
                # Update status
                plugin_info.load_status = PluginLoadStatus.NOT_LOADED
                
                # Remove from sys.modules
                module_name = f"ocr_plugin_{plugin_name}"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                self._logger.info(f"Successfully unloaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def get_loaded_plugins(self) -> list[str]:
        """Get list of loaded plugin names."""
        return [name for name, info in self.registry.get_all_plugins().items() 
                if info.load_status == PluginLoadStatus.LOADED]
    
    def get_plugin_info(self, plugin_name: str) -> dict[str, Any]:
        """Get plugin information as dictionary."""
        plugin_info = self.registry.get_plugin_info(plugin_name)
        return plugin_info.__dict__ if plugin_info else {}
    
    def get_available_engines(self) -> list[str]:
        """Get list of available OCR engine names."""
        return list(self.registry.get_available_engines().keys())
    
    def get_engine(self, engine_name: str) -> IOCREngine | None:
        """Get OCR engine by name."""
        return self.registry.get_engine(engine_name)
    
    def _should_use_gpu(self) -> bool:
        """
        Determine if GPU should be used based on runtime mode configuration.
        
        Returns:
            True if GPU should be used, False otherwise
        """
        if not self.config_manager:
            # No config manager, default to auto-detect
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                self._logger.info(f"No config manager, auto-detecting GPU: {gpu_available}")
                return gpu_available
            except ImportError:
                self._logger.info("No config manager and PyTorch not available, using CPU")
                return False
        
        runtime_mode = self.config_manager.get_setting('performance.runtime_mode', 'auto')
        self._logger.info(f"Runtime mode from config: {runtime_mode}")
        
        if runtime_mode == 'cpu':
            self._logger.info("Runtime mode set to CPU only")
            return False
        elif runtime_mode == 'gpu':
            self._logger.info("Runtime mode set to GPU acceleration")
            return True
        else:  # 'auto'
            # Auto-detect GPU availability
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                self._logger.info(f"Runtime mode set to Auto, GPU available: {gpu_available}")
                return gpu_available
            except ImportError:
                self._logger.info("Runtime mode set to Auto, PyTorch not available, using CPU")
                return False
    
    def load_all_plugins(self, config: dict[str, Any] | None = None) -> dict[str, bool]:
        """
        Load all discovered plugins.
        
        Args:
            config: Optional configuration for all plugins
            
        Returns:
            Dictionary mapping plugin names to load success status
        """
        results = {}
        for plugin_name in self.registry.get_all_plugins().keys():
            results[plugin_name] = self.load_plugin(plugin_name, config)
        return results


def discover_ocr_plugin_names_from_disk(
    visible_plugins: set[str] | None = None,
    include_all: bool = False,
) -> list[str]:
    """
    Scan OCR plugin directories for plugin.json and return plugin names.
    Does not register plugins or check dependencies. Used by the settings UI
    to show available engines before the pipeline has loaded.
    """
    directories = _get_default_ocr_plugin_directories()
    allowed = {p.lower() for p in visible_plugins} if visible_plugins else None

    names: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            manifest_path = os.path.join(item_path, "plugin.json")
            if os.path.isdir(item_path) and os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("name")
                    if not name:
                        continue
                    normalized = name.lower()
                    if not include_all and allowed is not None and normalized not in allowed:
                        continue
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
                except (json.JSONDecodeError, OSError):
                    pass
    return names


def inspect_ocr_plugin_dependencies(plugin_name: str) -> tuple[bool, list[str]]:
    """
    Inspect a plugin manifest and return missing required dependencies.

    Returns:
        tuple[exists_on_disk, missing_dependencies]
    """
    target = (plugin_name or "").strip().lower()
    if not target:
        return False, []

    for directory in _get_default_ocr_plugin_directories():
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue

        manifest = dir_path / target / "plugin.json"
        if not manifest.is_file():
            continue

        try:
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return True, []

        dependencies = data.get("dependencies", []) or []
        optional = set(data.get("optional_dependencies", []) or [])

        missing: list[str] = []
        for dep in dependencies:
            if dep in optional:
                continue
            import_name = _PACKAGE_IMPORT_MAP.get(dep, dep)
            if importlib.util.find_spec(import_name) is None:
                missing.append(dep)
        return True, missing

    return False, []


def _dedupe_paths(paths: list[str]) -> list[str]:
    """Deduplicate and normalize path list while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for path_str in paths:
        if not path_str:
            continue
        normalized = str(Path(path_str).expanduser().resolve(strict=False))
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _get_default_ocr_plugin_directories() -> list[str]:
    """Return OCR plugin directories that work in dev and deployed setups."""
    candidates: list[str] = []

    # Single source of truth for app-root-relative plugin path.
    try:
        from app.utils.path_utils import get_plugin_stages_dir
        candidates.append(str(get_plugin_stages_dir("ocr")))
    except Exception:
        pass

    # Fallbacks for older layouts and direct source execution.
    current_dir = Path(__file__).resolve().parent
    candidates.extend([
        str(current_dir.parent.parent / "plugins" / "stages" / "ocr"),
        str(current_dir / "plugins"),
        str(current_dir / "engines"),
    ])

    # Support running from arbitrary current working directory.
    candidates.append(str(Path.cwd() / "plugins" / "stages" / "ocr"))

    # Support frozen/packaged app layout where plugins are beside executable.
    if getattr(sys, "frozen", False):
        candidates.append(str(Path(sys.executable).resolve().parent / "plugins" / "stages" / "ocr"))

    # User plugin directories (legacy + current naming).
    candidates.append(str(Path.home() / ".translation_system" / "plugins" / "stages" / "ocr"))
    candidates.append(str(Path.home() / ".optikr" / "plugins" / "stages" / "ocr"))

    return _dedupe_paths(candidates)