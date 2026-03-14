"""
Typed configuration data definitions for OptikR.

Replaces generic dict[str, Any] with TypedDict definitions so that
static analysis tools can catch type errors at development time.
Field names match the keys used in config_schema and user_config.json.

Requirements: 8.1
"""
from typing import Any, TypedDict


class CaptureConfig(TypedDict):
    """Configuration for the screen capture subsystem."""
    method: str
    fps: int
    monitor_index: int


class OCRConfig(TypedDict):
    """Configuration for the OCR subsystem."""
    engine: str
    language: str
    languages: list[str]
    confidence_threshold: float


class TranslationConfig(TypedDict):
    """Configuration for the translation subsystem."""
    engine: str
    source_language: str
    target_language: str


class OverlayConfig(TypedDict):
    """Configuration for the overlay display subsystem."""
    font_size: int
    font_color: str
    background_color: str
    opacity: float
    positioning_mode: str


class PerformanceConfig(TypedDict):
    """Configuration for performance and runtime."""
    runtime_mode: str


class AppConfig(TypedDict):
    """Top-level application configuration."""
    capture: CaptureConfig
    ocr: OCRConfig
    translation: TranslationConfig
    overlay: OverlayConfig
    performance: PerformanceConfig
    plugins: dict[str, Any]
