"""
OCR Layer Implementation

This module provides the main OCR layer that integrates with the plugin system
to provide unified OCR functionality across multiple engines.
"""

import time
import threading
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .ocr_engine_interface import IOCREngine, OCRProcessingOptions
from .ocr_plugin_manager import OCRPluginManager
from app.models import Frame, TextBlock
from app.interfaces import IOCRLayer


class OCRLayerStatus(Enum):
    """OCR layer status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class OCRLayerConfig:
    """OCR layer configuration."""
    default_engine: str = "easyocr"
    fallback_engines: list[str] = field(default_factory=list)
    auto_fallback_enabled: bool = True
    cache_enabled: bool = True
    cache_size_limit: int = 1000
    processing_timeout_ms: int = 10000
    parallel_processing: bool = False
    max_parallel_workers: int = 4


@dataclass
class OCRResult:
    """OCR processing result with metadata."""
    text_blocks: list[TextBlock]
    engine_used: str
    processing_time_ms: float
    success: bool
    error_message: str | None = None
    confidence_score: float = 0.0


class OCRCache:
    """Simple LRU cache for OCR results."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with maximum size."""
        self.max_size = max_size
        self._cache: dict[str, OCRResult] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
    
    def _generate_key(self, frame: Frame, options: OCRProcessingOptions) -> str:
        """Generate cache key from frame content and options."""
        if frame.data is not None:
            d = frame.data
            stride = max(1, d.nbytes // 1024)
            content_hash = hash(d.flat[::stride].tobytes())
        else:
            content_hash = 0
        options_key = f"{options.language}_{options.confidence_threshold}_{options.preprocessing_enabled}"
        return f"{content_hash}_{frame.width}x{frame.height}_{options_key}"
    
    def get(self, frame: Frame, options: OCRProcessingOptions) -> OCRResult | None:
        """Get cached result if available."""
        key = self._generate_key(frame, options)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        
        return None
    
    def put(self, frame: Frame, options: OCRProcessingOptions, result: OCRResult) -> None:
        """Cache OCR result."""
        key = self._generate_key(frame, options)
        
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add to cache
            self._cache[key] = result
            self._access_order.append(key)
            
            # Evict oldest if over limit
            while len(self._cache) > self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0
            }


class OCRLayer(IOCRLayer):
    """Main OCR layer implementation with plugin system integration."""
    
    def __init__(self, config: OCRLayerConfig | None = None, 
                 plugin_directories: list[str] | None = None,
                 config_manager=None):
        """
        Initialize OCR layer.
        
        Args:
            config: OCR layer configuration
            plugin_directories: Directories to search for OCR plugins
            config_manager: Configuration manager for runtime settings
        """
        self.config = config or OCRLayerConfig()
        self.plugin_manager = OCRPluginManager(plugin_directories, config_manager)
        
        self.status = OCRLayerStatus.UNINITIALIZED
        self._current_engine: str | None = None
        self._cache = OCRCache(self.config.cache_size_limit) if self.config.cache_enabled else None
        
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ocr.layer")
    
    def initialize(self, auto_discover: bool = True, auto_load: bool = True) -> bool:
        """
        Initialize OCR layer and plugin system.
        
        Args:
            auto_discover: Automatically discover available plugins
            auto_load: Automatically load discovered plugins
            
        Returns:
            True if initialization successful
        """
        try:
            self.status = OCRLayerStatus.INITIALIZING
            self._logger.info("Initializing OCR layer...")
            
            # Discover plugins
            if auto_discover:
                discovered = self.plugin_manager.discover_plugins()
                self._logger.info(f"Discovered {len(discovered)} OCR plugins")
            
            # Load plugins
            if auto_load:
                load_results = self.plugin_manager.load_all_plugins()
                successful_loads = sum(1 for success in load_results.values() if success)
                self._logger.info(f"Successfully loaded {successful_loads}/{len(load_results)} plugins")
            
            # Set default engine
            available_engines = self.plugin_manager.get_available_engines()
            if available_engines:
                if self.config.default_engine in available_engines:
                    self._current_engine = self.config.default_engine
                else:
                    self._current_engine = available_engines[0]
                    self._logger.warning(f"Default engine {self.config.default_engine} not available, "
                                       f"using {self._current_engine}")
            else:
                self._logger.error("No OCR engines available after initialization")
                self.status = OCRLayerStatus.ERROR
                return False
            
            self.status = OCRLayerStatus.READY
            self._logger.info(f"OCR layer initialized successfully with {len(available_engines)} engines")
            return True
            
        except Exception as e:
            self.status = OCRLayerStatus.ERROR
            self._logger.error(f"Failed to initialize OCR layer: {e}")
            return False
    
    def extract_text(self, frame: Frame, engine: str | None = None, 
                    options: OCRProcessingOptions | None = None) -> list[TextBlock]:
        """
        Extract text from frame using specified or default OCR engine.
        
        Args:
            frame: Input frame containing image data
            engine: Optional specific engine name to use
            options: Optional processing options
            
        Returns:
            List of extracted text blocks
        """
        if self.status != OCRLayerStatus.READY:
            raise RuntimeError(f"OCR layer not ready (status: {self.status})")
        
        # Use default options if not provided
        if options is None:
            options = OCRProcessingOptions()
        
        # Check cache first
        if self._cache:
            cached_result = self._cache.get(frame, options)
            if cached_result and cached_result.success:
                self._logger.debug(f"Returning cached OCR result for frame {frame.timestamp}")
                return cached_result.text_blocks
        
        # Determine engine to use
        target_engine = engine or self._current_engine
        if not target_engine:
            raise ValueError("No OCR engine specified and no default engine set")
        
        # Process with primary engine
        result = self._process_with_engine(frame, target_engine, options)
        
        # Try fallback engines if primary failed and auto-fallback enabled
        if not result.success and self.config.auto_fallback_enabled:
            for fallback_engine in self.config.fallback_engines:
                if fallback_engine != target_engine:
                    self._logger.info(f"Trying fallback engine: {fallback_engine}")
                    result = self._process_with_engine(frame, fallback_engine, options)
                    if result.success:
                        break
        
        # Cache result if successful
        if self._cache and result.success:
            self._cache.put(frame, options, result)
        
        if not result.success:
            error_msg = result.error_message or "OCR processing failed"
            self._logger.error(f"OCR extraction failed: {error_msg}")
        
        return result.text_blocks
    
    def _process_with_engine(self, frame: Frame, engine_name: str, 
                           options: OCRProcessingOptions) -> OCRResult:
        """
        Process frame with specific OCR engine.
        
        Args:
            frame: Input frame
            engine_name: Name of OCR engine to use
            options: Processing options
            
        Returns:
            OCR processing result
        """
        start_time = time.perf_counter()
        
        try:
            # Get engine instance
            engine = self.plugin_manager.get_engine(engine_name)
            if not engine:
                return OCRResult(
                    text_blocks=[],
                    engine_used=engine_name,
                    processing_time_ms=0,
                    success=False,
                    error_message=f"Engine {engine_name} not available"
                )
            
            if not engine.is_ready():
                return OCRResult(
                    text_blocks=[],
                    engine_used=engine_name,
                    processing_time_ms=0,
                    success=False,
                    error_message=f"Engine {engine_name} not ready (status: {engine.get_status()})"
                )
            
            # Validate frame
            if not engine.validate_frame(frame):
                return OCRResult(
                    text_blocks=[],
                    engine_used=engine_name,
                    processing_time_ms=0,
                    success=False,
                    error_message="Frame validation failed for engine"
                )
            
            # Extract text
            with self._lock:
                self.status = OCRLayerStatus.PROCESSING
            text_blocks = engine.extract_text(frame, options)
            with self._lock:
                self.status = OCRLayerStatus.READY
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Calculate confidence score
            confidence_score = 0.0
            if text_blocks:
                confidence_score = sum(block.confidence for block in text_blocks) / len(text_blocks)
            
            return OCRResult(
                text_blocks=text_blocks,
                engine_used=engine_name,
                processing_time_ms=processing_time,
                success=True,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            with self._lock:
                self.status = OCRLayerStatus.READY
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return OCRResult(
                text_blocks=[],
                engine_used=engine_name,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def register_engine(self, engine_name: str, engine_instance: IOCREngine) -> bool:
        """
        Register a new OCR engine instance.
        
        Args:
            engine_name: Name for the engine
            engine_instance: OCR engine instance
            
        Returns:
            True if registration successful
        """
        return self.plugin_manager.registry.register_engine(engine_instance)
    
    def get_available_engines(self) -> list[str]:
        """Get list of available OCR engine names."""
        return self.plugin_manager.get_available_engines()
    
    def set_language(self, language: str) -> bool:
        """
        Set the expected language for OCR processing.
        
        Automatically converts language codes to the format expected by the current engine:
        - Tesseract: 3-letter codes (eng, deu, spa, etc.)
        - EasyOCR/PaddleOCR/ONNX: ISO 639-1 codes (en, de, es, etc.)
        
        Args:
            language: Language code in any format (e.g., 'en', 'eng', 'de', 'deu')
            
        Returns:
            True if language is supported by current engine
        """
        if not self._current_engine:
            return False
        
        # Import language mapper
        try:
            from app.utils.language_mapper import LanguageCodeMapper
            
            # Get current engine name
            engine_name = str(self._current_engine)
            
            # Normalize language code for the current engine
            normalized_language = LanguageCodeMapper.normalize(language, engine_name)
            
            if normalized_language != language:
                self._logger.info(f"Language code converted: {language} -> {normalized_language} (for {engine_name})")
            
            # Set language on engine
            engine = self.plugin_manager.get_engine(self._current_engine)
            if engine:
                return engine.set_language(normalized_language)
            
        except Exception as e:
            self._logger.error(f"Failed to normalize language code: {e}")
            # Fallback to original behavior
            engine = self.plugin_manager.get_engine(self._current_engine)
            if engine:
                return engine.set_language(language)
        
        return False
    
    def set_default_engine(self, engine_name: str) -> bool:
        """
        Set the default OCR engine.
        
        Args:
            engine_name: Name of engine to set as default
            
        Returns:
            True if engine is available and set successfully
        """
        available_engines = self.get_available_engines()
        if engine_name in available_engines:
            self._current_engine = engine_name
            self.config.default_engine = engine_name
            self._logger.info(f"Set default OCR engine to: {engine_name}")
            return True
        
        self._logger.error(f"Engine {engine_name} not available for setting as default")
        return False
    
    def get_current_engine(self) -> str | None:
        """Get the current default engine name."""
        return self._current_engine

    def engine_has_text_detection(self) -> bool:
        """Check if the current engine has built-in text detection.

        Engines with detection (e.g. Mokuro) should receive the full
        frame rather than pre-cropped ROI regions, because their own
        detector outperforms the generic bubble detector.
        """
        if not self._current_engine:
            return False
        engine = self.plugin_manager.get_engine(self._current_engine)
        if engine is None:
            return False
        caps = engine.get_capabilities()
        return getattr(caps, "has_text_detection", False)
    
    def get_engine_info(self, engine_name: str) -> dict[str, Any]:
        """
        Get information about a specific engine.
        
        Args:
            engine_name: Name of engine
            
        Returns:
            Dictionary containing engine information
        """
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
            "is_ready": engine.is_ready()
        }
    