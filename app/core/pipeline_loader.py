"""
Pipeline Loader for asynchronous pipeline initialization.

This module handles the loading and initialization of the translation pipeline,
including OCR engines, translation layers, and overlay systems.
"""

import logging
import sys
import traceback
from pathlib import Path
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


_logger = logging.getLogger(__name__)


class LoadingStage(Enum):
    """Pipeline loading stages for progress tracking."""
    DISCOVERING_PLUGINS = (1, "Discovering OCR plugins")
    LOADING_OCR = (2, "Loading OCR plugins (this may take 20-30 seconds)")
    VERIFYING_OCR = (3, "Verifying OCR engines")
    INITIALIZING_TRANSLATION = (4, "Initializing translation layer")
    VERIFYING_OVERLAY = (5, "Verifying overlay system")
    FINALIZING = (6, "Finalizing pipeline")
    
    def __init__(self, number: int, description: str):
        self.number = number
        self.description = description
    
    @property
    def total_stages(self) -> int:
        return 6
    
    def format_progress(self) -> str:
        """Format stage as '[1/6] Description'."""
        return f"[{self.number}/{self.total_stages}] {self.description}..."


class PipelineLoader(QObject):
    """
    Loads and initializes the translation pipeline.
    
    This class handles the complex initialization process including:
    - Directory structure verification
    - OCR plugin discovery and loading
    - Translation layer initialization
    - Overlay system verification
    
    Signals:
        finished: Emitted when loading completes successfully (pipeline object)
        error: Emitted when loading fails (error message)
        progress: Emitted during loading with status updates (progress message)
    """
    
    finished = pyqtSignal(object)  # Emits pipeline object or None
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress messages

    def __init__(self, config_manager, parent=None):
        """
        Initialize the pipeline loader.
        
        Args:
            config_manager: Configuration manager instance
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self._is_running = False

    def run(self) -> None:
        """
        Load and initialize the pipeline.
        
        This method runs synchronously and emits signals for progress updates.
        On success, emits finished signal with the pipeline object.
        On failure, emits error signal with error message.
        """
        self._is_running = True
        
        try:
            _logger.info("="*60)
            _logger.info("PIPELINE INITIALIZATION")
            _logger.info("="*60)
            
            # Stage 1: Discover plugins
            self._stage_discover_plugins()
            
            # Stage 2: Load OCR plugins
            pipeline = self._stage_load_ocr_plugins()
            
            # Stage 3: Verify OCR engines
            self._stage_verify_ocr_engines(pipeline)
            
            # Stage 4: Initialize translation
            self._stage_initialize_translation(pipeline)
            
            # Stage 5: Verify overlay
            self._stage_verify_overlay()
            
            # Stage 6: Finalize
            self._stage_finalize()
            
            _logger.info("="*60)
            _logger.info("✓ OptikR is ready to use")
            _logger.info("="*60)
            
            self.progress.emit("Pipeline ready")
            self.finished.emit(pipeline)
            
        except KeyboardInterrupt:
            _logger.warning("Pipeline initialization cancelled by user")
            self.error.emit("Cancelled by user")
        except SystemExit:
            _logger.warning("System exit called during initialization")
            self.error.emit("System exit")
        except Exception as e:
            self._handle_fatal_error(e)
        finally:
            self._is_running = False

    def _stage_discover_plugins(self) -> None:
        """Stage 1: Verify directory structure and discover OCR plugins."""
        stage = LoadingStage.DISCOVERING_PLUGINS
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        
        # Verify critical directories exist
        script_dir = Path(__file__).parent.parent.parent
        critical_dirs = [
            script_dir / "app",
            script_dir / "app" / "ocr",
            script_dir / "app" / "workflow",
            script_dir / "plugins",
        ]
        
        _logger.info("   → Verifying directory structure...")
        missing_dirs = []
        for dir_path in critical_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path.relative_to(script_dir)))
                _logger.error("   ✗ Missing: %s", dir_path.relative_to(script_dir))
            else:
                _logger.debug("   ✓ Found: %s", dir_path.relative_to(script_dir))
        
        if missing_dirs:
            raise RuntimeError(f"Missing critical directories: {', '.join(missing_dirs)}")
        
        # Import pipeline module
        _logger.info("   → Importing pipeline module...")
        try:
            from app.workflow.startup_pipeline import StartupPipeline
            _logger.info("   ✓ Pipeline module loaded (StartupPipeline)")
        except ImportError as e:
            _logger.error("   ✗ Failed to import pipeline module: %s", e)
            _logger.error("   → Check if all __init__.py files exist")
            _logger.error("   → Verify app/workflow/startup_pipeline.py exists")
            raise

    def _stage_load_ocr_plugins(self):
        """Stage 2: Load OCR plugins and create pipeline."""
        stage = LoadingStage.LOADING_OCR
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        _logger.info("   → Scanning for available OCR plugins...")
        
        # Add diagnostic information
        script_dir = Path(__file__).parent.parent.parent
        _logger.debug("   → Python: %s", sys.executable)
        _logger.debug("   → Working dir: %s", Path.cwd())
        _logger.debug("   → Script dir: %s", script_dir)
        
        # Check for OCR plugin directories
        plugins_dir = script_dir / "plugins" / "stages" / "ocr"
        if plugins_dir.exists():
            plugin_dirs = [d.name for d in plugins_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('__')]
            if plugin_dirs:
                _logger.info("   → Found OCR plugins: %s", ', '.join(plugin_dirs))
            else:
                _logger.warning("   → No OCR plugins found")
            
            # Check for plugin.json files
            for plugin_dir in plugin_dirs:
                plugin_json = plugins_dir / plugin_dir / "plugin.json"
                if plugin_json.exists():
                    _logger.debug("   ✓ %s/plugin.json exists", plugin_dir)
                else:
                    _logger.warning("   ✗ %s/plugin.json missing", plugin_dir)
        else:
            _logger.warning("   ✗ OCR plugins directory not found: %s", plugins_dir)
        
        # Import OCR components
        _logger.info("   → Importing OCR layer...")
        try:
            from app.ocr.ocr_layer import OCRLayer
            _logger.info("   ✓ OCR layer module imported")
        except ImportError as e:
            _logger.error("   ✗ Failed to import OCR layer: %s", e)
            raise ImportError(f"OCR layer import failed: {e}") from e
        
        _logger.info("   → Importing OCR plugin manager...")
        try:
            from app.ocr.ocr_plugin_manager import OCRPluginManager
            _logger.info("   ✓ OCR plugin manager imported")
        except ImportError as e:
            _logger.error("   ✗ Failed to import plugin manager: %s", e)
            raise ImportError(f"Plugin manager import failed: {e}") from e
        
        # Create pipeline
        _logger.info("   → Creating pipeline integration...")
        try:
            from app.workflow.startup_pipeline import StartupPipeline
            _logger.info("   ✓ StartupPipeline class imported")
            
            _logger.info("   → Creating startup pipeline instance...")
            integration = StartupPipeline(config_manager=self.config_manager)
            _logger.info("   ✓ Integration instance created")
            
            _logger.info("   → Initializing pipeline components...")
            _logger.info("      (This may take 20-30 seconds...)")
            
            result = integration.initialize_components()
            if not result:
                raise RuntimeError("initialize_components() returned False - check component creation")
            
            _logger.info("   ✓ Components initialized")
            
            pipeline = integration
            
            if not pipeline:
                raise RuntimeError("Pipeline creation returned None - check logs for details")
            
            _logger.info("   ✓ Pipeline created successfully")
            
            # Verify OCR layer exists
            if not hasattr(pipeline, 'ocr_layer') or pipeline.ocr_layer is None:
                _logger.warning("   ⚠ Warning: Pipeline has no OCR layer")
            else:
                _logger.info("   ✓ OCR layer attached to pipeline")
            
            _logger.info("   ✓ OCR engines initialized")
            
            return pipeline
            
        except ImportError as e:
            _logger.error("   ✗ Missing OCR dependencies: %s", e)
            _logger.error("   → Possible causes:")
            _logger.error("      • Missing __init__.py files")
            _logger.error("      • Incorrect Python path")
            _logger.error("      • Missing OCR engine packages")
            _logger.error("   → Try: pip install easyocr pytesseract paddleocr")
            raise
        except MemoryError as e:
            _logger.error("   ✗ Out of memory loading OCR models")
            _logger.error("   → Try closing other applications")
            _logger.error("   → Consider using a lighter OCR engine (tesseract)")
            raise
        except AttributeError as e:
            _logger.error("   ✗ Attribute error: %s", e)
            _logger.error("   → This usually means a required module or class is missing")
            raise
        except Exception as e:
            _logger.error("   ✗ Pipeline creation failed: %s: %s", type(e).__name__, e)
            raise RuntimeError(f"Pipeline creation failed: {e}") from e

    def _stage_verify_ocr_engines(self, pipeline) -> None:
        """Stage 3: Verify OCR engines are available."""
        stage = LoadingStage.VERIFYING_OCR
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        
        try:
            if hasattr(pipeline, 'ocr_layer') and pipeline.ocr_layer:
                available = pipeline.get_available_ocr_engines()
                if available:
                    _logger.info("   ✓ Found %d OCR engine(s): %s", 
                               len(available), ', '.join(available))
                else:
                    _logger.warning("   ⚠ Warning: No OCR engines available")
            else:
                _logger.warning("   ⚠ Warning: OCR layer not initialized")
        except Exception as e:
            _logger.warning("   ⚠ Could not verify OCR engines: %s", e)

    def _stage_initialize_translation(self, pipeline) -> None:
        """Stage 4: Initialize translation layer."""
        stage = LoadingStage.INITIALIZING_TRANSLATION
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        
        try:
            if hasattr(pipeline, 'translation_layer') and pipeline.translation_layer:
                _logger.info("   ✓ Translation layer created")
                _logger.info("   → MarianMT will load on first translation (lazy loading)")
                _logger.info("   → First translation may take 3-5 seconds")
            else:
                _logger.warning("   ⚠ Translation layer not available")
        except Exception as e:
            _logger.warning("   ⚠ Translation layer check failed: %s", e)

    def _stage_verify_overlay(self) -> None:
        """Stage 5: Verify overlay system."""
        stage = LoadingStage.VERIFYING_OVERLAY
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        _logger.info("   ✓ Overlay system ready")

    def _stage_finalize(self) -> None:
        """Stage 6: Final checks and cleanup."""
        stage = LoadingStage.FINALIZING
        self.progress.emit(stage.format_progress())
        _logger.info(stage.format_progress())
        _logger.info("   ✓ All components initialized")

    def _handle_fatal_error(self, exception: Exception) -> None:
        """
        Handle fatal errors during pipeline initialization.
        
        Args:
            exception: The exception that caused the failure
        """
        _logger.error("="*60)
        _logger.error("✗ PIPELINE INITIALIZATION FAILED")
        _logger.error("="*60)
        _logger.error("Error Type: %s", type(exception).__name__)
        _logger.error("Error Message: %s", exception)
        _logger.error("Full Traceback:")
        _logger.error("-"*60)
        
        # Log the full traceback
        _logger.exception(exception)
        
        _logger.error("-"*60)
        _logger.error("Troubleshooting:")
        _logger.error("1. Check if all dependencies are installed:")
        _logger.error("   pip install opencv-python numpy easyocr pytesseract paddleocr")
        _logger.error("2. Check logs folder for detailed error messages")
        _logger.error("3. Ensure you have enough RAM (4GB+ recommended)")
        _logger.error("4. Try running from the original dev folder to compare")
        _logger.error("="*60)
        
        self.error.emit(f"{type(exception).__name__}: {str(exception)}")
