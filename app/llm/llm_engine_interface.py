"""
LLM Engine Interface and Plugin System

This module provides the abstract base classes and interfaces for LLM engines,
along with a plugin wrapper for registering and managing different LLM
implementations (e.g. Qwen3, custom models).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import time
import threading
import logging

from app.workflow.base.plugin_interface import IPlugin


class LLMEngineType(Enum):
    """LLM engine type enumeration."""
    QWEN3 = "qwen3"
    CUSTOM = "custom"


class LLMEngineStatus(Enum):
    """LLM engine status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class LLMProcessingMode(Enum):
    """How the LLM stage should operate in the pipeline."""
    REFINE = "refine"
    TRANSLATE = "translate"
    CUSTOM = "custom"


@dataclass
class LLMProcessingOptions:
    """Configuration options for a single LLM processing call."""
    prompt_template: str = "Refine the following translated text:\n{text}"
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    mode: LLMProcessingMode = LLMProcessingMode.REFINE
    source_language: str = ""
    target_language: str = ""
    timeout_ms: int = 30000
    custom_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMEngineCapabilities:
    """LLM engine capabilities and features."""
    supported_modes: list[LLMProcessingMode] = field(
        default_factory=lambda: [LLMProcessingMode.REFINE, LLMProcessingMode.CUSTOM]
    )
    supports_gpu: bool = False
    supports_batch_processing: bool = False
    supports_streaming: bool = False
    max_context_length: int = 4096
    memory_requirements_mb: int = 2048
    initialization_time_ms: int = 5000


@dataclass
class LLMEngineMetrics:
    """LLM engine performance metrics."""
    total_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    total_tokens_generated: int = 0
    success_count: int = 0
    error_count: int = 0
    last_processing_time_ms: float = 0.0

    def update_processing_time(self, processing_time_ms: float, tokens: int = 0) -> None:
        """Update processing time metrics."""
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = self.total_processing_time_ms / self.total_processed
        self.last_processing_time_ms = processing_time_ms
        self.total_tokens_generated += tokens

    def record_success(self) -> None:
        self.success_count += 1

    def record_error(self) -> None:
        self.error_count += 1

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class ILLMEngine(ABC):
    """Abstract base class for LLM engines."""

    def __init__(self, engine_name: str, engine_type: LLMEngineType):
        self.engine_name = engine_name
        self.engine_type = engine_type
        self.status = LLMEngineStatus.UNINITIALIZED
        self.capabilities = LLMEngineCapabilities()
        self.metrics = LLMEngineMetrics()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"llm.{engine_name}")

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> bool:
        """
        Initialize the LLM engine with configuration.

        Args:
            config: Engine-specific configuration parameters

        Returns:
            True if initialization successful, False otherwise
        """
        ...

    @abstractmethod
    def process_text(self, text: str, options: LLMProcessingOptions) -> str:
        """
        Process text through the LLM engine.

        The semantics depend on ``options.mode``:
        - *refine*: post-translation polishing
        - *translate*: full LLM-based translation (replaces translation stage)
        - *custom*: user-defined prompt

        Args:
            text: Input text to process
            options: Processing options and prompt configuration

        Returns:
            Processed / generated text
        """
        ...

    @abstractmethod
    def process_batch(self, texts: list[str], options: LLMProcessingOptions) -> list[str]:
        """
        Process multiple texts in batch.

        Args:
            texts: List of input texts
            options: Processing options

        Returns:
            List of processed texts, one per input
        """
        ...

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt used across calls.

        Args:
            prompt: System prompt string
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release engine resources and free memory."""
        ...

    # -- convenience helpers (non-abstract) ------------------------------------

    def get_capabilities(self) -> LLMEngineCapabilities:
        return self.capabilities

    def get_metrics(self) -> LLMEngineMetrics:
        with self._lock:
            return self.metrics

    def get_status(self) -> LLMEngineStatus:
        return self.status

    def is_ready(self) -> bool:
        return self.status == LLMEngineStatus.READY

    def _record_processing_start(self) -> float:
        with self._lock:
            self.status = LLMEngineStatus.PROCESSING
        return time.perf_counter() * 1000

    def _record_processing_end(
        self, start_time_ms: float, success: bool, tokens: int = 0
    ) -> None:
        end_time_ms = time.perf_counter() * 1000
        processing_time = end_time_ms - start_time_ms

        with self._lock:
            self.metrics.update_processing_time(processing_time, tokens)
            if success:
                self.metrics.record_success()
                self.status = LLMEngineStatus.READY
            else:
                self.metrics.record_error()
                self.status = LLMEngineStatus.ERROR


class LLMEnginePlugin(IPlugin):
    """Base class for LLM engine plugins."""

    def __init__(self, engine_class: type, plugin_info: dict[str, Any]):
        """
        Args:
            engine_class: Class that implements ILLMEngine
            plugin_info: Plugin metadata dictionary
        """
        self.engine_class = engine_class
        self.plugin_info = plugin_info
        self._engine_instance: ILLMEngine | None = None

    def initialize(self, config: dict[str, Any]) -> bool:
        try:
            required_methods = ["initialize", "process_text", "get_status", "is_ready"]
            for method in required_methods:
                if not hasattr(self.engine_class, method):
                    raise ValueError(f"Engine class must implement {method} method")

            engine_name = self.plugin_info.get("name", "unknown")
            engine_type = self.plugin_info.get("type", "custom")

            try:
                self._engine_instance = self.engine_class(engine_name, engine_type)
            except TypeError:
                try:
                    self._engine_instance = self.engine_class(engine_name)
                except TypeError:
                    self._engine_instance = self.engine_class()

            return self._engine_instance.initialize(config)

        except Exception as e:
            logging.error(f"Failed to initialize LLM engine plugin: {e}", exc_info=True)
            return False

    def cleanup(self) -> None:
        if self._engine_instance:
            self._engine_instance.cleanup()
            self._engine_instance = None

    def get_info(self) -> dict[str, Any]:
        return self.plugin_info.copy()

    def get_engine_instance(self) -> ILLMEngine | None:
        return self._engine_instance
