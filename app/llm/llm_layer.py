"""
LLM Layer Implementation

This module provides the main LLM layer that integrates with the plugin system
to offer unified LLM-based text processing (refinement, translation, custom
prompts) across multiple engines.

It also hosts :class:`SharedModelRegistry`, a process-wide singleton that
prevents the same HuggingFace model from being loaded twice (e.g. when Qwen3
is selected as both the translation engine and the LLM engine).
"""

import time
import threading
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .llm_engine_interface import ILLMEngine, LLMProcessingOptions, LLMProcessingMode
from .llm_plugin_manager import LLMPluginManager


class SharedModelRegistry:
    """Process-wide registry that prevents duplicate model loads.

    Both the Qwen3 translation plugin and the Qwen3 LLM plugin consult this
    registry before calling ``from_pretrained``.  If the model is already
    loaded by another component, the caller receives the cached reference
    instead of allocating GPU/RAM a second time.

    The registry is keyed by ``(model_name, device_str)`` and maintains a
    reference count so that the model is only freed when *all* users have
    released it.

    Usage::

        registry = SharedModelRegistry.instance()

        # Before loading
        shared = registry.get("Qwen/Qwen3-1.7B", "cuda")
        if shared:
            model, tokenizer = shared["model"], shared["tokenizer"]
        else:
            model = AutoModelForCausalLM.from_pretrained(...)
            tokenizer = AutoTokenizer.from_pretrained(...)
            registry.register("Qwen/Qwen3-1.7B", "cuda", model, tokenizer)

        # On cleanup
        if registry.release("Qwen/Qwen3-1.7B", "cuda"):
            del model, tokenizer  # last user — safe to free
    """

    _instance: "SharedModelRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("llm.shared_model_registry")

    @classmethod
    def instance(cls) -> "SharedModelRegistry":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(self, model_name: str, device: str) -> dict[str, Any] | None:
        """Look up a previously registered model/tokenizer pair.

        Increments the reference count on hit so that :meth:`release` knows
        how many consumers are still active.

        Returns:
            Dict with ``"model"`` and ``"tokenizer"`` keys, or *None*.
        """
        with self._lock:
            entry = self._models.get((model_name, device))
            if entry is not None:
                entry["ref_count"] += 1
                self._logger.info(
                    "Shared model hit: %s on %s (ref_count=%d)",
                    model_name, device, entry["ref_count"],
                )
                return entry
            return None

    def register(
        self, model_name: str, device: str, model: Any, tokenizer: Any,
    ) -> None:
        """Register a freshly loaded model/tokenizer for sharing."""
        with self._lock:
            key = (model_name, device)
            if key in self._models:
                self._logger.warning(
                    "Model %s on %s already registered — replacing entry",
                    model_name, device,
                )
            self._models[key] = {
                "model": model,
                "tokenizer": tokenizer,
                "ref_count": 1,
            }
            self._logger.info(
                "Registered shared model: %s on %s", model_name, device,
            )

    def release(self, model_name: str, device: str) -> bool:
        """Decrement the reference count for a shared model.

        Returns:
            ``True`` if the caller was the last user and should free the
            model's memory.  ``False`` if other users still hold a reference.
        """
        with self._lock:
            key = (model_name, device)
            entry = self._models.get(key)
            if entry is None:
                return True
            entry["ref_count"] -= 1
            self._logger.info(
                "Released shared model: %s on %s (ref_count=%d)",
                model_name, device, entry["ref_count"],
            )
            if entry["ref_count"] <= 0:
                del self._models[key]
                return True
            return False

    def is_loaded(self, model_name: str, device: str) -> bool:
        """Check whether a model is currently held in the registry."""
        with self._lock:
            return (model_name, device) in self._models


class LLMLayerStatus(Enum):
    """LLM layer status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class LLMLayerConfig:
    """LLM layer configuration."""
    default_engine: str = "qwen3"
    fallback_engines: list[str] = field(default_factory=list)
    auto_fallback_enabled: bool = True
    processing_timeout_ms: int = 30000
    default_mode: LLMProcessingMode = LLMProcessingMode.REFINE
    default_temperature: float = 0.7
    default_max_tokens: int = 512


@dataclass
class LLMResult:
    """LLM processing result with metadata."""
    output_text: str
    engine_used: str
    processing_time_ms: float
    success: bool
    error_message: str | None = None
    tokens_generated: int = 0


class LLMLayer:
    """Main LLM layer providing a facade over the plugin system."""

    def __init__(
        self,
        config: LLMLayerConfig | None = None,
        plugin_directories: list[str] | None = None,
        config_manager=None,
    ):
        """
        Args:
            config: LLM layer configuration
            plugin_directories: Directories to search for LLM plugins
            config_manager: Configuration manager for runtime settings
        """
        self.config = config or LLMLayerConfig()
        self.plugin_manager = LLMPluginManager(plugin_directories, config_manager)

        self.status = LLMLayerStatus.UNINITIALIZED
        self._current_engine: str | None = None

        self._lock = threading.RLock()
        self._logger = logging.getLogger("llm.layer")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, auto_discover: bool = True, auto_load: bool = True) -> bool:
        """
        Initialize the LLM layer and its plugin system.

        Args:
            auto_discover: Automatically discover available plugins
            auto_load: Automatically load discovered plugins

        Returns:
            True if initialization successful
        """
        try:
            self.status = LLMLayerStatus.INITIALIZING
            self._logger.info("Initializing LLM layer...")

            if auto_discover:
                discovered = self.plugin_manager.discover_plugins()
                self._logger.info(f"Discovered {len(discovered)} LLM plugins")

            if auto_load:
                load_results = self.plugin_manager.load_all_plugins()
                successful = sum(1 for ok in load_results.values() if ok)
                self._logger.info(
                    f"Successfully loaded {successful}/{len(load_results)} LLM plugins"
                )

            available_engines = self.plugin_manager.get_available_engines()
            if available_engines:
                if self.config.default_engine in available_engines:
                    self._current_engine = self.config.default_engine
                else:
                    self._current_engine = available_engines[0]
                    self._logger.warning(
                        f"Default engine {self.config.default_engine} not available, "
                        f"using {self._current_engine}"
                    )
            else:
                self._logger.error("No LLM engines available after initialization")
                self.status = LLMLayerStatus.ERROR
                return False

            self.status = LLMLayerStatus.READY
            self._logger.info(
                f"LLM layer initialized with {len(available_engines)} engine(s)"
            )
            return True

        except Exception as e:
            self.status = LLMLayerStatus.ERROR
            self._logger.error(f"Failed to initialize LLM layer: {e}")
            return False

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_text(
        self,
        text: str,
        engine: str | None = None,
        options: LLMProcessingOptions | None = None,
    ) -> str:
        """
        Process text through the configured LLM engine.

        Args:
            text: Input text to process
            engine: Optional engine name override
            options: Optional processing options

        Returns:
            Processed output text

        Raises:
            RuntimeError: If the layer is not ready
            ValueError: If no engine is available
        """
        if self.status != LLMLayerStatus.READY:
            raise RuntimeError(f"LLM layer not ready (status: {self.status})")

        if options is None:
            options = LLMProcessingOptions(
                mode=self.config.default_mode,
                temperature=self.config.default_temperature,
                max_tokens=self.config.default_max_tokens,
            )

        target_engine = engine or self._current_engine
        if not target_engine:
            raise ValueError("No LLM engine specified and no default engine set")

        result = self._process_with_engine(text, target_engine, options)

        if not result.success and self.config.auto_fallback_enabled:
            for fallback in self.config.fallback_engines:
                if fallback != target_engine:
                    self._logger.info(f"Trying fallback LLM engine: {fallback}")
                    result = self._process_with_engine(text, fallback, options)
                    if result.success:
                        break

        if not result.success:
            error_msg = result.error_message or "LLM processing failed"
            self._logger.error(f"LLM processing failed: {error_msg}")
            return text

        return result.output_text

    def process_batch(
        self,
        texts: list[str],
        engine: str | None = None,
        options: LLMProcessingOptions | None = None,
    ) -> list[str]:
        """
        Process multiple texts through the LLM engine.

        Args:
            texts: List of input texts
            engine: Optional engine name override
            options: Optional processing options

        Returns:
            List of processed output texts
        """
        if self.status != LLMLayerStatus.READY:
            raise RuntimeError(f"LLM layer not ready (status: {self.status})")

        if options is None:
            options = LLMProcessingOptions(
                mode=self.config.default_mode,
                temperature=self.config.default_temperature,
                max_tokens=self.config.default_max_tokens,
            )

        target_engine = engine or self._current_engine
        if not target_engine:
            raise ValueError("No LLM engine specified and no default engine set")

        llm_engine = self.plugin_manager.get_engine(target_engine)
        if not llm_engine or not llm_engine.is_ready():
            self._logger.error(f"Engine {target_engine} not available for batch")
            return list(texts)

        try:
            with self._lock:
                self.status = LLMLayerStatus.PROCESSING
            results = llm_engine.process_batch(texts, options)
            with self._lock:
                self.status = LLMLayerStatus.READY
            return results
        except Exception as e:
            with self._lock:
                self.status = LLMLayerStatus.READY
            self._logger.error(f"Batch LLM processing failed: {e}")
            return list(texts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_with_engine(
        self, text: str, engine_name: str, options: LLMProcessingOptions
    ) -> LLMResult:
        start_time = time.perf_counter()

        try:
            llm_engine = self.plugin_manager.get_engine(engine_name)
            if not llm_engine:
                return LLMResult(
                    output_text=text,
                    engine_used=engine_name,
                    processing_time_ms=0,
                    success=False,
                    error_message=f"Engine {engine_name} not available",
                )

            if not llm_engine.is_ready():
                return LLMResult(
                    output_text=text,
                    engine_used=engine_name,
                    processing_time_ms=0,
                    success=False,
                    error_message=(
                        f"Engine {engine_name} not ready "
                        f"(status: {llm_engine.get_status()})"
                    ),
                )

            with self._lock:
                self.status = LLMLayerStatus.PROCESSING
            output = llm_engine.process_text(text, options)
            with self._lock:
                self.status = LLMLayerStatus.READY

            processing_time = (time.perf_counter() - start_time) * 1000

            return LLMResult(
                output_text=output,
                engine_used=engine_name,
                processing_time_ms=processing_time,
                success=True,
            )

        except Exception as e:
            with self._lock:
                self.status = LLMLayerStatus.READY
            processing_time = (time.perf_counter() - start_time) * 1000

            return LLMResult(
                output_text=text,
                engine_used=engine_name,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Engine management
    # ------------------------------------------------------------------

    def set_default_engine(self, engine_name: str) -> bool:
        available = self.get_available_engines()
        if engine_name in available:
            self._current_engine = engine_name
            self.config.default_engine = engine_name
            self._logger.info(f"Set default LLM engine to: {engine_name}")
            return True
        self._logger.error(f"Engine {engine_name} not available")
        return False

    def get_current_engine(self) -> str | None:
        return self._current_engine

    def get_available_engines(self) -> list[str]:
        return self.plugin_manager.get_available_engines()

    def get_engine_info(self, engine_name: str) -> dict[str, Any]:
        engine = self.plugin_manager.get_engine(engine_name)
        if not engine:
            return {}
        capabilities = engine.get_capabilities()
        metrics = engine.get_metrics()
        return {
            "name": engine.engine_name,
            "type": engine.engine_type.value,
            "status": engine.get_status().value,
            "capabilities": capabilities.__dict__,
            "metrics": metrics.__dict__,
            "is_ready": engine.is_ready(),
        }

    def cleanup(self) -> None:
        """Release all engine resources."""
        for name in list(self.plugin_manager.get_loaded_plugins()):
            self.plugin_manager.unload_plugin(name)
        self.status = LLMLayerStatus.UNINITIALIZED
        self._current_engine = None
        self._logger.info("LLM layer cleaned up")
