"""
LLM Module

This module provides LLM (Large Language Model) processing functionality
for the Translation System with a plugin system for supporting multiple
LLM engines.
"""

from .llm_engine_interface import (
    ILLMEngine,
    LLMEngineType,
    LLMEngineStatus,
    LLMEngineCapabilities,
    LLMEngineMetrics,
    LLMProcessingOptions,
    LLMProcessingMode,
    LLMEnginePlugin,
)

from .llm_plugin_manager import (
    LLMPluginManager,
    LLMPluginRegistry,
    LLMPluginInfo,
    PluginLoadStatus,
)

from .llm_layer import (
    LLMLayer,
    LLMLayerConfig,
    LLMLayerStatus,
    LLMResult,
    SharedModelRegistry,
)

__all__ = [
    # Engine Interface
    "ILLMEngine",
    "LLMEngineType",
    "LLMEngineStatus",
    "LLMEngineCapabilities",
    "LLMEngineMetrics",
    "LLMProcessingOptions",
    "LLMProcessingMode",
    "LLMEnginePlugin",

    # Plugin Management
    "LLMPluginManager",
    "LLMPluginRegistry",
    "LLMPluginInfo",
    "PluginLoadStatus",

    # Main LLM Layer
    "LLMLayer",
    "LLMLayerConfig",
    "LLMLayerStatus",
    "LLMResult",

    # Shared Model Registry
    "SharedModelRegistry",
]
