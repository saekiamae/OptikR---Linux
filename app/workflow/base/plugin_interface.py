"""
Plugin Interface - Definitions for plugin metadata and settings.

Provides data structures for plugin.json files.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
from enum import Enum


class PluginType(Enum):
    """Types of plugins."""
    CAPTURE = "capture"
    OCR = "ocr"
    TRANSLATION = "translation"
    LLM = "llm"
    VISION = "vision"
    OPTIMIZER = "optimizer"
    TEXT_PROCESSOR = "text_processor"


class IPlugin(Protocol):
    """Protocol that all plugins must satisfy."""

    def initialize(self, config: dict[str, Any]) -> bool:
        """Initialize the plugin with the given configuration."""
        ...

    def cleanup(self) -> None:
        """Release resources held by the plugin."""
        ...

    def get_info(self) -> dict[str, Any]:
        """Return plugin metadata as a dictionary."""
        ...


class SettingType(Enum):
    """Types of plugin settings."""
    STRING = "string"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    ARRAY = "array"


@dataclass
class PluginSetting:
    """Definition of a plugin setting."""
    name: str
    type: SettingType
    default: Any
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    options: list[str] | None = None  # For dropdown/select
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'PluginSetting':
        """Create PluginSetting from dictionary."""
        raw_type = str(data.get('type', 'string')).strip().lower()
        type_aliases = {
            'str': 'string',
            'string': 'string',
            'int': 'int',
            'integer': 'int',
            'float': 'float',
            'double': 'float',
            'bool': 'bool',
            'boolean': 'bool',
            'array': 'array',
            'list': 'array',
            # Store object-like settings as strings in the generic dialog fallback.
            'dict': 'string',
            'object': 'string',
        }
        normalized_type = type_aliases.get(raw_type, 'string')
        return cls(
            name=name,
            type=SettingType(normalized_type),
            default=data.get('default'),
            description=data.get('description', ''),
            min_value=data.get('min'),
            max_value=data.get('max'),
            options=data.get('options')
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            'type': self.type.value,
            'default': self.default,
            'description': self.description
        }
        if self.min_value is not None:
            result['min'] = self.min_value
        if self.max_value is not None:
            result['max'] = self.max_value
        if self.options:
            result['options'] = self.options
        return result


@dataclass
class PluginMetadata:
    """Metadata for a plugin (from plugin.json)."""
    name: str
    display_name: str
    version: str
    author: str
    description: str
    type: PluginType
    worker_script: str
    enabled_by_default: bool = True
    essential: bool = False  # Essential plugins bypass master switch and cannot be disabled
    settings: dict[str, PluginSetting] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    
    # Optimizer-specific fields
    target_stage: str | None = None  # For optimizers
    stage: str | None = None  # "pre" or "post" for optimizers
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginMetadata':
        """Create PluginMetadata from dictionary (plugin.json)."""
        raw_type = str(data['type']).strip().lower()
        plugin_type_aliases = {
            'text-processor': 'text_processor',
            'textprocessor': 'text_processor',
        }
        normalized_type = plugin_type_aliases.get(raw_type, raw_type)

        raw_stage = data.get('stage')
        if isinstance(raw_stage, str):
            stage_aliases = {
                'before': 'pre',
                'after': 'post',
                'core': 'post',
                'global': 'post',
            }
            normalized_stage = stage_aliases.get(raw_stage.strip().lower(), raw_stage.strip().lower())
        else:
            normalized_stage = raw_stage

        # Parse settings
        settings = {}
        for name, setting_data in data.get('settings', {}).items():
            settings[name] = PluginSetting.from_dict(name, setting_data)
        
        return cls(
            name=data['name'],
            display_name=data.get('display_name', data['name']),
            version=data.get('version', 'pre-realese-1.0.0'),
            author=data.get('author', 'Unknown'),
            description=data.get('description', ''),
            type=PluginType(normalized_type),
            worker_script=data.get('worker_script', 'worker.py'),
            enabled_by_default=data.get('enabled_by_default', data.get('enabled', True)),
            essential=data.get('essential', False),
            settings=settings,
            dependencies=data.get('dependencies', []),
            target_stage=data.get('target_stage'),
            stage=normalized_stage
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for plugin.json)."""
        result = {
            'name': self.name,
            'display_name': self.display_name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'type': self.type.value,
            'worker_script': self.worker_script,
            'enabled_by_default': self.enabled_by_default,
            'essential': self.essential,
            'settings': {name: setting.to_dict() for name, setting in self.settings.items()},
            'dependencies': self.dependencies
        }
        
        if self.target_stage:
            result['target_stage'] = self.target_stage
        if self.stage:
            result['stage'] = self.stage
        
        return result
    
    def validate(self) -> list[str]:
        """
        Validate plugin metadata.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Plugin name is required")
        if not self.display_name:
            errors.append("Display name is required")
        if not self.version:
            errors.append("Version is required")
        if not self.worker_script:
            errors.append("Worker script is required")
        
        # Optimizer-specific validation
        if self.type == PluginType.OPTIMIZER:
            if not self.target_stage:
                errors.append("Optimizer must specify target_stage")
            if self.stage not in ['pre', 'post', None]:
                errors.append("Optimizer stage must be 'pre' or 'post'")
        
        return errors


@dataclass
class PluginSettings:
    """Runtime settings for a plugin instance."""
    plugin_name: str
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set setting value."""
        self.settings[key] = value
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'plugin_name': self.plugin_name,
            'enabled': self.enabled,
            'settings': self.settings
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginSettings':
        """Create from dictionary."""
        return cls(
            plugin_name=data['plugin_name'],
            enabled=data.get('enabled', True),
            settings=data.get('settings', {})
        )
