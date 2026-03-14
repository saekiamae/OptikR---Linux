"""
Base Interfaces and Abstract Classes

Defines the core interfaces that all major components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from .models import (
    Frame, TextBlock, Translation, PerformanceMetrics, CaptureRegion,
    RuntimeMode, PerformanceProfile, Rectangle
)


class CaptureSource(Enum):
    """Source types for screen capture."""
    FULL_SCREEN = "full_screen"
    WINDOW = "window"
    CUSTOM_REGION = "custom_region"


# Core Component Interfaces

class ICaptureLayer(ABC):
    """Interface for screen capture functionality."""

    @abstractmethod
    def capture_frame(self, source: CaptureSource, region: CaptureRegion) -> Frame:
        """Capture a frame from the specified source and region."""
        ...

    @abstractmethod
    def set_capture_mode(self, mode: str) -> bool:
        """Set the capture mode (DirectX, Screenshot, etc.)."""
        ...

    @abstractmethod
    def get_supported_modes(self) -> list[str]:
        """Get list of supported capture modes."""
        ...

    @abstractmethod
    def configure_capture_rate(self, fps: int) -> bool:
        """Configure the capture frame rate."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if capture functionality is available."""
        ...


class IPreprocessingLayer(ABC):
    """Interface for image preprocessing functionality."""

    @abstractmethod
    def preprocess(self, frame: Frame, profile: PerformanceProfile) -> Frame:
        """Preprocess frame for optimal OCR accuracy."""
        ...

    @abstractmethod
    def detect_roi(self, frame: Frame) -> list[Rectangle]:
        """Detect regions of interest containing text."""
        ...



class IOCRLayer(ABC):
    """Interface for OCR functionality."""

    @abstractmethod
    def extract_text(self, frame: Frame, engine: str | None = None, options: dict[str, Any] | None = None) -> list[TextBlock]:
        """Extract text from frame using specified or default OCR engine."""
        ...

    @abstractmethod
    def register_engine(self, engine_name: str, engine_instance: Any) -> bool:
        """Register a new OCR engine."""
        ...

    @abstractmethod
    def get_available_engines(self) -> list[str]:
        """Get list of available OCR engines."""
        ...

    @abstractmethod
    def set_language(self, language: str) -> bool:
        """Set the expected language for OCR processing."""
        ...


class ITranslationLayer(ABC):
    """Interface for translation functionality."""

    @abstractmethod
    def translate(self, text: str, engine: str, src_lang: str,
                  tgt_lang: str, options: dict[str, Any]) -> str:
        """Translate text using specified engine and language pair."""
        ...

    @abstractmethod
    def translate_batch(self, texts: list[str], engine: str,
                        src_lang: str, tgt_lang: str) -> list[str]:
        """Translate multiple texts in batch for efficiency."""
        ...

    @abstractmethod
    def get_supported_languages(self, engine: str) -> list[str]:
        """Get supported languages for specified engine."""
        ...

    @abstractmethod
    def cache_translation(self, source: str, target: str, translation: str) -> None:
        """Cache translation result for future use."""
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear translation cache."""
        ...


class IOverlayRenderer(ABC):
    """Interface for overlay rendering functionality."""

    @abstractmethod
    def render_overlay(self, frame: Frame, translations: list[Translation],
                       mode: str) -> None:
        """Render translated text overlay on screen."""
        ...

    @abstractmethod
    def set_overlay_style(self, style: dict[str, Any]) -> None:
        """Configure overlay appearance and styling."""
        ...

    @abstractmethod
    def toggle_overlay(self, visible: bool) -> None:
        """Show or hide the overlay."""
        ...

    @abstractmethod
    def update_positions(self, translations: list[Translation]) -> None:
        """Update overlay text positions."""
        ...

    @abstractmethod
    def is_overlay_active(self) -> bool:
        """Check if overlay is currently active."""
        ...


class ILogger(ABC):
    """Interface for logging functionality."""

    @abstractmethod
    def log(self, level: str, message: str, context: dict[str, Any] | None = None) -> None:
        """Log a message with specified level and optional context."""
        ...

    @abstractmethod
    def set_log_level(self, level: str) -> None:
        """Set minimum logging level."""
        ...

    @abstractmethod
    def add_handler(self, handler: Any) -> None:
        """Add a log handler."""
        ...

    @abstractmethod
    def get_logs(self, level: str | None = None,
                 limit: int | None = None) -> list[dict[str, Any]]:
        """Retrieve logged messages."""
        ...
