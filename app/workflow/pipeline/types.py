"""
Pipeline type definitions.

Defines the PipelineStage and ExecutionStrategy protocols, plus
supporting data classes and the ResourceOwner mixin used throughout
the pipeline package.

Requirements: 2.1, 2.2, 9.1
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Protocol


logger = logging.getLogger('optikr.pipeline.types')


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExecutionMode(Enum):
    """Pipeline execution mode."""
    SEQUENTIAL = "sequential"
    ASYNC = "async"
    CUSTOM = "custom"
    SUBPROCESS = "subprocess"


class PipelineState(Enum):
    """Lifecycle state of a pipeline instance."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result produced by a single pipeline stage."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineConfig:
    """Unified pipeline configuration."""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    target_fps: int = 10
    max_consecutive_errors: int = 10
    enable_preprocessing: bool = True
    enable_caching: bool = True
    stop_timeout: float = 5.0
    source_language: str = "ja"
    target_language: str = "en"
    capture_region: Any = None
    overlay_region: Any = None


@dataclass
class PipelineStats:
    """Runtime statistics collected by a pipeline."""
    frames_processed: int = 0
    frames_skipped: int = 0
    frames_dropped: int = 0
    consecutive_errors: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0.0
    average_fps: float = 0.0
    average_latency_ms: float = 0.0
    total_translations: int = 0
    cache_hits: int = 0
    stage_times_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

TranslationCallback = Callable[[dict[str, Any]], None]
ErrorCallback = Callable[[str], None]
StateChangeCallback = Callable[[PipelineState, PipelineState], None]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class PipelineStageProtocol(Protocol):
    """Protocol that every pipeline stage must satisfy."""

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute the stage with *input_data* and return a result."""
        ...

    def cleanup(self) -> None:
        """Release any resources held by the stage."""
        ...


class ExecutionStrategy(Protocol):
    """Protocol for pipeline execution strategies."""

    def run_pipeline(
        self,
        stages: list[PipelineStageProtocol],
        initial_input: dict[str, Any],
    ) -> StageResult:
        """Run *stages* sequentially, threading data through each one."""
        ...


# ---------------------------------------------------------------------------
# ResourceOwner mixin (consolidated from app/utils/resource_management.py)
# ---------------------------------------------------------------------------

class ResourceOwner:
    """Mixin for classes that own resources requiring cleanup.

    Subclasses override ``_do_cleanup()`` with their teardown logic.
    ``cleanup()`` is idempotent — the actual cleanup runs at most once.
    The class also implements the context manager protocol.
    """

    _cleaned_up: bool = False

    def __enter__(self) -> ResourceOwner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Run cleanup logic exactly once. Subsequent calls are no-ops."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self._do_cleanup()
        except Exception as exc:
            logger.warning(
                "Cleanup failed for %s: [%s] %s",
                type(self).__name__,
                type(exc).__name__,
                exc,
            )

    def _do_cleanup(self) -> None:
        """Override in subclasses to perform actual resource teardown."""
