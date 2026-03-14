"""
Plugin Manager - Discovers and loads plugins from disk.

Scans the plugins/ directory for plugin.json files and manages plugin lifecycle.
"""

import json
from pathlib import Path
import logging

from .base.plugin_interface import PluginMetadata, PluginSettings, PluginType
from .plugins.manager import UnifiedPluginManager


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    def __init__(self, plugin_directories: list[str] = None, config_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        if plugin_directories is None:
            plugin_directories = ['plugins/']
        self.plugin_directories = [Path(d) for d in plugin_directories]
        
        # Loaded plugins: {plugin_name: (metadata, plugin_path)}
        self.plugins: dict[str, tuple[PluginMetadata, Path]] = {}
        
        # Plugin settings: {plugin_name: PluginSettings}
        self.plugin_settings: dict[str, PluginSettings] = {}
        
        self._unified = UnifiedPluginManager(plugin_directories)
        
        self.logger.info("Plugin manager initialized")

    def _infer_plugin_type(self, plugin_json_path: Path) -> str | None:
        """Infer plugin type from path when legacy metadata omits 'type'."""
        parts = {p.lower() for p in plugin_json_path.parts}
        if "optimizers" in parts:
            return "optimizer"
        if "text_processors" in parts:
            return "text_processor"
        if "translation" in parts:
            return "translation"
        if "ocr" in parts:
            return "ocr"
        if "capture" in parts:
            return "capture"
        if "llm" in parts:
            return "llm"
        if "vision" in parts:
            return "vision"
        return None

    def _resolve_worker_script(self, plugin_dir: Path, metadata: PluginMetadata) -> str | None:
        """Resolve worker script with type-aware fallbacks."""
        candidates = [metadata.worker_script]
        if metadata.type == PluginType.OPTIMIZER:
            candidates.append("optimizer.py")
        elif metadata.type == PluginType.TEXT_PROCESSOR:
            candidates.append("processor.py")
        else:
            candidates.append("worker.py")
            candidates.append("engine.py")
            candidates.append(f"{metadata.name}_engine.py")
            candidates.append(f"{metadata.name}.py")

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (plugin_dir / candidate).exists():
                return candidate
        return None
    
    def scan_plugins(self) -> int:
        """Scan plugin directories for plugins.
        
        Returns:
            Number of plugins found
        """
        self.plugins.clear()
        found_count = 0
        
        for plugin_dir in self.plugin_directories:
            if not plugin_dir.exists():
                self.logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue
            
            self.logger.info(f"Scanning plugin directory: {plugin_dir}")
            
            for plugin_json in plugin_dir.rglob('plugin.json'):
                try:
                    plugin_path = plugin_json.parent
                    metadata = self._load_plugin_metadata(plugin_json)
                    
                    if metadata:
                        self.plugins[metadata.name] = (metadata, plugin_path)
                        found_count += 1
                        self.logger.info(f"Loaded plugin: {metadata.name} ({metadata.type.value})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load plugin from {plugin_json}: {e}")
        
        self.logger.info(f"Found {found_count} plugins")
        for name, (meta, _) in self.plugins.items():
            self._unified.registry.register(meta)
        return found_count
    
    def _load_plugin_metadata(self, plugin_json_path: Path) -> PluginMetadata | None:
        """Load plugin metadata from plugin.json file."""
        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            enabled = bool(data.get('enabled_by_default', data.get('enabled', True)))
            if 'type' not in data:
                inferred_type = self._infer_plugin_type(plugin_json_path)
                if inferred_type:
                    data['type'] = inferred_type
                    self.logger.debug("Inferred plugin type '%s' for %s", inferred_type, plugin_json_path)
                elif not enabled:
                    self.logger.info("Skipping disabled plugin missing type: %s", plugin_json_path)
                    return None

            metadata = PluginMetadata.from_dict(data)
            
            errors = metadata.validate()
            if errors:
                # Disabled plugins may be placeholders/incomplete and should not
                # flood startup logs.
                if not metadata.enabled_by_default:
                    self.logger.info("Skipping disabled plugin with validation issues: %s", plugin_json_path)
                    return None
                self.logger.error(f"Plugin validation failed: {plugin_json_path}")
                for error in errors:
                    self.logger.error(f"  - {error}")
                return None
            
            plugin_dir = plugin_json_path.parent
            resolved_worker = self._resolve_worker_script(plugin_dir, metadata)
            if resolved_worker:
                metadata.worker_script = resolved_worker
            else:
                # Placeholder/catalog-only plugins may contain no Python module yet.
                if not list(plugin_dir.glob("*.py")):
                    self.logger.debug("Skipping plugin with no Python module files: %s", plugin_json_path)
                    return None

                if metadata.enabled_by_default:
                    self.logger.debug("Skipping plugin with unresolved worker script: %s", plugin_json_path)
                else:
                    self.logger.debug("Skipping disabled plugin without worker script: %s", plugin_json_path)
                return None
            
            return metadata
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {plugin_json_path}: {e}")
            return None
        except KeyError as e:
            self.logger.error(f"Missing required field in {plugin_json_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading plugin metadata: {e}")
            return None
    
    def get_all_plugins(self) -> list[PluginMetadata]:
        """Get all loaded plugins."""
        return [metadata for metadata, _ in self.plugins.values()]

    def get_plugin(self, plugin_name: str) -> PluginMetadata | None:
        """Get metadata for a named plugin, or None if not found."""
        entry = self.plugins.get(plugin_name)
        return entry[0] if entry else None

    def get_plugins_by_type(self, plugin_type) -> list[PluginMetadata]:
        """Return all plugins matching *plugin_type*."""
        return [meta for meta, _ in self.plugins.values() if meta.type == plugin_type]

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Return whether *plugin_name* is enabled."""
        settings = self.plugin_settings.get(plugin_name)
        if settings is not None:
            return settings.enabled
        entry = self.plugins.get(plugin_name)
        if entry:
            return entry[0].enabled_by_default
        return False

    def set_plugin_enabled(self, plugin_name: str, enabled: bool) -> None:
        """Enable or disable a plugin."""
        if plugin_name not in self.plugin_settings:
            self.plugin_settings[plugin_name] = PluginSettings(plugin_name=plugin_name)
        self.plugin_settings[plugin_name].enabled = enabled

    def get_plugin_path(self, plugin_name: str) -> Path | None:
        """Get plugin directory path."""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name][1]
        return None
    
    def get_plugin_settings(self, plugin_name: str) -> PluginSettings | None:
        """Get runtime settings for a plugin."""
        return self.plugin_settings.get(plugin_name)
    
    def set_plugin_setting(self, plugin_name: str, key: str, value):
        """Set a plugin setting value."""
        if plugin_name not in self.plugin_settings:
            self.plugin_settings[plugin_name] = PluginSettings(plugin_name=plugin_name)
        
        self.plugin_settings[plugin_name].set(key, value)
        from app.utils.credential_filter import get_credential_filter
        _filter = get_credential_filter()
        if _filter._is_credential_key(key):
            self.logger.info("Plugin %s setting '%s' = ***REDACTED***", plugin_name, key)
        else:
            self.logger.info("Plugin %s setting '%s' = %s", plugin_name, key, value)
