"""
OCR Engine Interface and Plugin System

This module provides the abstract base classes and interfaces for OCR engines,
along with a plugin management system for registering and managing
different OCR implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import time
import threading
import logging

from app.models import Frame, TextBlock
from app.workflow.base.plugin_interface import IPlugin


class OCREngineType(Enum):
    """OCR engine type enumeration."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    ONNX_RUNTIME = "onnx_runtime"
    HYBRID_OCR = "hybrid_ocr"
    WINDOWS_OCR = "windows_ocr"
    RAPIDOCR = "rapidocr"
    DOCTR = "doctr"
    SURYA_OCR = "surya_ocr"
    JUDGE_OCR = "judge_ocr"
    MOKURO = "mokuro"
    CUSTOM = "custom"


class OCREngineStatus(Enum):
    """OCR engine status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class OCREngineCapabilities:
    """OCR engine capabilities and features."""
    supported_languages: list[str] = field(default_factory=list)
    supports_gpu: bool = False
    supports_batch_processing: bool = False
    supports_confidence_scores: bool = True
    supports_text_formatting: bool = False
    supports_text_orientation: bool = False
    has_text_detection: bool = False
    max_image_size: tuple[int, int] = (4096, 4096)
    min_image_size: tuple[int, int] = (32, 32)
    supported_image_formats: list[str] = field(default_factory=lambda: ["RGB", "GRAY"])
    memory_requirements_mb: int = 512
    initialization_time_ms: int = 1000


@dataclass
class OCREngineMetrics:
    """OCR engine performance metrics."""
    total_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    confidence_score: float = 0.0
    last_processing_time_ms: float = 0.0
    
    def update_processing_time(self, processing_time_ms: float) -> None:
        """Update processing time metrics."""
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = self.total_processing_time_ms / self.total_processed
        self.last_processing_time_ms = processing_time_ms
    
    def record_success(self, confidence: float = 0.0) -> None:
        """Record successful OCR operation."""
        self.success_count += 1
        if confidence > 0:
            # Update running average of confidence
            total_confidence = self.confidence_score * (self.success_count - 1) + confidence
            self.confidence_score = total_confidence / self.success_count
    
    def record_error(self) -> None:
        """Record OCR error."""
        self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


@dataclass
class OCRProcessingOptions:
    """OCR processing configuration options."""
    language: str = "en"
    confidence_threshold: float = 0.5
    preprocessing_enabled: bool = True
    gpu_acceleration: bool = False
    batch_size: int = 1
    timeout_ms: int = 5000
    custom_config: dict[str, Any] = field(default_factory=dict)


class IOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def __init__(self, engine_name: str, engine_type: OCREngineType):
        """Initialize OCR engine with name and type."""
        self.engine_name = engine_name
        self.engine_type = engine_type
        self.status = OCREngineStatus.UNINITIALIZED
        self.capabilities = OCREngineCapabilities()
        self.metrics = OCREngineMetrics()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"ocr.{engine_name}")
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> bool:
        """
        Initialize the OCR engine with configuration.
        
        Args:
            config: Engine-specific configuration parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        ...
    
    @abstractmethod
    def extract_text(self, frame: Frame, options: OCRProcessingOptions) -> list[TextBlock]:
        """
        Extract text from frame using this OCR engine.
        
        Args:
            frame: Input frame containing image data
            options: Processing options and configuration
            
        Returns:
            List of extracted text blocks with positions and confidence
        """
        ...
    
    @abstractmethod
    def extract_text_batch(self, frames: list[Frame], options: OCRProcessingOptions) -> list[list[TextBlock]]:
        """
        Extract text from multiple frames in batch.
        
        Args:
            frames: List of input frames
            options: Processing options and configuration
            
        Returns:
            List of text block lists, one for each input frame
        """
        ...
    
    @abstractmethod
    def set_language(self, language: str) -> bool:
        """
        Set the expected language for OCR processing.
        
        Args:
            language: Language code (e.g., 'en', 'zh', 'ja')
            
        Returns:
            True if language is supported and set successfully
        """
        ...
    
    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        ...
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up engine resources and shutdown."""
        ...
    
    def get_capabilities(self) -> OCREngineCapabilities:
        """Get engine capabilities."""
        return self.capabilities
    
    def get_metrics(self) -> OCREngineMetrics:
        """Get engine performance metrics."""
        with self._lock:
            return self.metrics
    
    def get_status(self) -> OCREngineStatus:
        """Get current engine status."""
        return self.status
    
    def is_ready(self) -> bool:
        """Check if engine is ready for processing."""
        return self.status == OCREngineStatus.READY
    
    def validate_frame(self, frame: Frame) -> bool:
        """
        Validate if frame is suitable for this engine.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if frame is valid for processing
        """
        if frame.data is None or frame.data.size == 0:
            return False
        
        height, width = frame.data.shape[:2]
        min_w, min_h = self.capabilities.min_image_size
        max_w, max_h = self.capabilities.max_image_size
        
        return (min_w <= width <= max_w and min_h <= height <= max_h)
    
    def _record_processing_start(self) -> float:
        """Record processing start time."""
        with self._lock:
            self.status = OCREngineStatus.PROCESSING
        return time.perf_counter() * 1000
    
    def _record_processing_end(self, start_time_ms: float, success: bool, confidence: float = 0.0) -> None:
        """Record processing completion."""
        end_time_ms = time.perf_counter() * 1000
        processing_time = end_time_ms - start_time_ms
        
        with self._lock:
            self.metrics.update_processing_time(processing_time)
            if success:
                self.metrics.record_success(confidence)
                self.status = OCREngineStatus.READY
            else:
                self.metrics.record_error()
                self.status = OCREngineStatus.ERROR


class OCREnginePlugin(IPlugin):
    """Base class for OCR engine plugins."""
    
    def __init__(self, engine_class: type, plugin_info: dict[str, Any]):
        """
        Initialize OCR engine plugin.
        
        Args:
            engine_class: OCR engine class that implements IOCREngine
            plugin_info: Plugin metadata and information
        """
        self.engine_class = engine_class
        self.plugin_info = plugin_info
        self._engine_instance: IOCREngine | None = None
    
    def initialize(self, config: dict[str, Any]) -> bool:
        """Initialize the plugin and create engine instance."""
        try:
            # Validate engine class has required methods (duck typing)
            required_methods = ['initialize', 'extract_text', 'get_status', 'is_ready']
            for method in required_methods:
                if not hasattr(self.engine_class, method):
                    raise ValueError(f"Engine class must implement {method} method")
            
            # Create engine instance
            engine_name = self.plugin_info.get("name", "unknown")
            engine_type = self.plugin_info.get("type", "custom")
            
            # Try to create instance with different signatures
            try:
                self._engine_instance = self.engine_class(engine_name, engine_type)
            except TypeError:
                # Try without engine_type
                try:
                    self._engine_instance = self.engine_class(engine_name)
                except TypeError:
                    # Try with no arguments
                    self._engine_instance = self.engine_class()
            
            # Initialize engine
            return self._engine_instance.initialize(config)
            
        except Exception as e:
            logging.error(f"Failed to initialize OCR engine plugin: {e}", exc_info=True)
            return False
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self._engine_instance:
            self._engine_instance.cleanup()
            self._engine_instance = None
    
    def get_info(self) -> dict[str, Any]:
        """Get plugin information."""
        return self.plugin_info.copy()
    
    def get_engine_instance(self) -> IOCREngine | None:
        """Get the OCR engine instance."""
        return self._engine_instance