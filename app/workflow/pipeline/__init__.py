"""
Pipeline package for OptikR.

Provides decomposed pipeline components: stages, execution strategies,
a base pipeline engine, and plugin-aware stage wrapping.

Requirements: 2.1, 2.2, 2.3
"""

from .types import (
    PipelineStageProtocol,
    ExecutionStrategy,
    StageResult,
    PipelineConfig,
    PipelineStats,
    PipelineState,
    ExecutionMode,
    TranslationCallback,
    ErrorCallback,
    StateChangeCallback,
)
from .stages import (
    CaptureStage,
    OCRStage,
    TranslationStage,
    OverlayStage,
    AudioCaptureStage,
    SpeechToTextStage,
    TTSStage,
)
from .plugin_stage import PluginAwareStage
from .strategies import (
    SequentialStrategy,
    AsyncStrategy,
    CustomStrategy,
    SubprocessStrategy,
)
from .base_pipeline import BasePipeline

__all__ = [
    # Types & protocols
    "PipelineStageProtocol",
    "ExecutionStrategy",
    "StageResult",
    "PipelineConfig",
    "PipelineStats",
    "PipelineState",
    "ExecutionMode",
    "TranslationCallback",
    "ErrorCallback",
    "StateChangeCallback",
    # Visual stages
    "CaptureStage",
    "OCRStage",
    "TranslationStage",
    "OverlayStage",
    # Audio stages
    "AudioCaptureStage",
    "SpeechToTextStage",
    "TTSStage",
    # Plugin wrapper
    "PluginAwareStage",
    # Strategies
    "SequentialStrategy",
    "AsyncStrategy",
    "CustomStrategy",
    "SubprocessStrategy",
    # Pipeline engine
    "BasePipeline",
]
