"""
UI Manager for the application.

This module manages all UI components including tabs, toolbar, sidebar,
and dialogs. It separates UI concerns from business logic and provides
a clean interface for UI operations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal
from app.localization import tr

if TYPE_CHECKING:
    from .interfaces import MainWindowProtocol

logger = logging.getLogger(__name__)


class UIManager(QObject):
    """
    Manages all UI components for the application.
    
    Responsibilities:
    - Tab creation and lazy loading
    - Toolbar creation and management
    - Sidebar creation and management
    - Dialog management (help, region selector, performance monitor, etc.)
    - Theme switching
    
    The UIManager separates UI concerns from the main window, making the
    codebase more maintainable and testable.
    """
    
    # Signals for UI events
    theme_toggled = pyqtSignal(str)  # Emitted when theme changes (theme_name)
    
    def __init__(self, config_manager=None, parent=None):
        """
        Initialize the UI Manager.
        
        Args:
            config_manager: The configuration manager instance
            parent: The parent QObject (typically the main window)
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.parent_window: 'MainWindowProtocol' | None = parent
        
        # Tab storage for lazy loading
        self._tabs: dict[str, Any] = {}
        self._loaded_tabs: dict[str, bool] = {}
        
        # UI component references
        self._toolbar = None
        self._sidebar = None
        self._tab_widget = None
        
        # Dialog references
        self._active_dialogs: dict[str, Any] = {}
        
        # Theme state
        self._current_theme = "light"
    
    def create_tabs(self, tab_widget) -> None:
        """
        Create all settings tabs with lazy loading support.
        
        Args:
            tab_widget: The QTabWidget to populate with tabs
        
        Consolidated layout — 5 top-level tabs:
          General, Dictionary & Context, Pipeline (hub), Storage, Advanced
        
        Previously separate tabs (Capture, OCR, Translation, LLM, Overlay)
        are now sub-tabs inside the Pipeline hub.
        """
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from app.localization import tr
        
        self._tab_widget = tab_widget
        
        tab_names = [
            "general",
            "dictionary_context",
            "pipeline",
            "overlay",
            "storage",
            "advanced",
        ]
        
        tab_display_names = [
            tr("general"),
            tr("dictionary_context"),
            tr("pipeline"),
            tr("overlay"),
            tr("storage"),
            tr("advanced"),
        ]
        
        # Initialize tab storage
        for tab_name in tab_names:
            self._tabs[tab_name] = None
            self._loaded_tabs[tab_name] = False
        
        # Add placeholder tabs to the tab widget
        for i, (tab_name, display_name) in enumerate(zip(tab_names, tab_display_names)):
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.setContentsMargins(20, 20, 20, 20)
            
            loading_label = QLabel("Loading...")
            loading_label.setStyleSheet("font-size: 14pt; color: #888;")
            placeholder_layout.addWidget(loading_label)
            placeholder_layout.addStretch()
            
            tab_widget.addTab(placeholder, display_name)
        
        # Connect to tab change signal for lazy loading
        if tab_widget:
            tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Load first tab (General) immediately so user sees content
        self._load_tab_on_demand(0)
    
    def _on_tab_changed(self, index):
        """Handle tab change event for lazy loading."""
        if index >= 0:
            self._load_tab_on_demand(index)
    
    def create_toolbar(self, parent) -> Any:
        """
        Create and configure the application toolbar.
        
        Args:
            parent: The parent widget for the toolbar
        
        Returns:
            The created toolbar widget
        
        The toolbar provides quick access to common actions like
        starting/stopping translation, changing themes, and opening dialogs.
        """
        from ui.layout.toolbar.main_toolbar import MainToolbar
        
        # Create toolbar widget
        self._toolbar = MainToolbar(parent)
        
        # Connect to UIManager methods
        self._toolbar.helpClicked.connect(self.show_help)
        self._toolbar.regionOverlayClicked.connect(self.show_region_overlay)
        self._toolbar.themeToggled.connect(self.toggle_theme)
        
        # Connect signals to parent window (MainWindowProtocol)
        if self.parent_window:
            self._toolbar.startClicked.connect(self.parent_window.toggle_translation)
            self._toolbar.captureRegionClicked.connect(self.parent_window.show_capture_region_selector)
            self._toolbar.monitorClicked.connect(self.parent_window.show_performance_monitor)
            self._toolbar.saveClicked.connect(self.parent_window.save_all_settings)
            self._toolbar.importClicked.connect(self.parent_window.import_settings)
            self._toolbar.exportClicked.connect(self.parent_window.export_settings)
        else:
            self._toolbar.captureRegionClicked.connect(self.show_capture_region_selector)
            self._toolbar.monitorClicked.connect(self.show_performance_monitor)
        
        # Load saved theme preference
        is_dark = self.config_manager.get_setting('ui.dark_mode', True)
        self._toolbar.set_theme(is_dark)
        
        return self._toolbar
    
    def create_sidebar(self, parent) -> Any:
        """
        Create and configure the application sidebar.
        
        Args:
            parent: The parent widget for the sidebar
        
        Returns:
            The created sidebar widget
        
        The sidebar displays real-time metrics and status information
        about the translation pipeline.
        """
        from ui.layout.sidebar.sidebar_widget import SidebarWidget
        
        # Create sidebar widget
        self._sidebar = SidebarWidget(self.config_manager, parent)
        
        # Connect signals to UIManager methods
        self._sidebar.logsClicked.connect(self.show_log_viewer)
        self._sidebar.fullTestClicked.connect(self.show_full_test_dialog)
        self._sidebar.languagePackClicked.connect(self.show_language_pack_manager)
        self._sidebar.imageProcessingClicked.connect(self.show_image_processing_dialog)
        self._sidebar.benchmarkClicked.connect(self.show_benchmark_dialog)
        
        # Connect signals to parent window (MainWindowProtocol)
        if self.parent_window:
            self._sidebar.quickOcrSwitchClicked.connect(self.parent_window.show_quick_ocr_switch)
            self._sidebar.presetLoaded.connect(self.parent_window._on_preset_loaded)
            self._sidebar.contentModeChanged.connect(self.parent_window._on_content_mode_changed)
            self._sidebar.pipelineModeChanged.connect(self.parent_window._on_pipeline_mode_changed)
        
        return self._sidebar
    
    def update_sidebar_metrics(self):
        """Update sidebar metrics with real-time system data."""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=0.1)
            
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.used / (1024 ** 3)
            
            gpu = None
            try:
                import pynvml
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu = float(util.gpu)
                finally:
                    pynvml.nvmlShutdown()
            except Exception:
                pass
            
            fps = 0.0
            if self.parent_window and self.parent_window.startup_pipeline and self.parent_window.startup_pipeline.is_running():
                startup = self.parent_window.startup_pipeline
                try:
                    metrics = startup.get_metrics()
                    fps = metrics.average_fps
                except Exception:
                    pass
            
            if self._sidebar:
                self._sidebar.update_metrics(fps=fps, cpu=cpu, gpu=gpu, memory_gb=memory_gb)
                
        except Exception:
            pass
    
    def show_help(self) -> None:
        """
        Display the help dialog.
        
        Shows comprehensive help information about using the application,
        including keyboard shortcuts, feature descriptions, and troubleshooting.
        """
        from ui.dialogs.help_dialog import show_help_dialog
        
        show_help_dialog(self.config_manager, self.parent_window)
    
    def show_region_overlay(self) -> None:
        """
        Display the region overlay for capture area visualization.
        
        Shows a semi-transparent overlay indicating the current capture
        region on the screen.
        """
        from ui.overlays.region_visualizer import RegionVisualizer
        
        # Create visualizer (keep reference to prevent garbage collection)
        if not hasattr(self, 'region_visualizer'):
            self.region_visualizer = RegionVisualizer(self.config_manager)
        
        # Hide any existing overlays first
        self.region_visualizer.hide_all()
        
        # Show all regions (red for OCR capture, blue for translation display, green for boundaries)
        capture_overlays, translation_overlays = self.region_visualizer.show_both_regions()
        
        if capture_overlays or translation_overlays:
            logger.info("Showing region visualizations: %d capture, %d translation",
                        len(capture_overlays), len(translation_overlays))
    
    def show_capture_region_selector(self) -> None:
        """
        Display the capture region selector dialog.
        
        Allows the user to interactively select a screen region for
        text capture and translation.
        """
        if self.parent_window:
            self.parent_window.show_capture_region_selector()
    
    def show_performance_monitor(self) -> None:
        """
        Toggle the performance overlay on/off.
        
        Shows a lightweight, draggable, always-on-top overlay with real-time
        performance metrics (CPU, GPU, memory, FPS). Right-click to configure.
        """
        from ui.overlays.performance_overlay import PerformanceOverlay
        
        # Toggle overlay visibility
        if not hasattr(self, '_perf_overlay') or not self._perf_overlay:
            self._perf_overlay = PerformanceOverlay(config_manager=self.config_manager)
            self._wire_perf_overlay_pipeline()
        
        if self._perf_overlay.isVisible():
            self._perf_overlay.close()
        else:
            self._perf_overlay.show()
            self._perf_overlay.raise_()
            self._perf_overlay.activateWindow()

    def _wire_perf_overlay_pipeline(self) -> None:
        """Pass the startup pipeline to the performance overlay so it can read live metrics."""
        if not hasattr(self, '_perf_overlay') or not self._perf_overlay:
            return
        pipeline = getattr(self.parent_window, 'startup_pipeline', None)
        if pipeline:
            self._perf_overlay.set_pipeline(pipeline)
    
    def show_language_pack_manager(self) -> None:
        """
        Display the localization manager dialog.
        
        Opens a dialog for managing UI language packs
        (app localization, not translation models).
        """
        from ui.dialogs.localization_manager import show_localization_manager
        
        show_localization_manager(self.parent_window)
    
    def show_log_viewer(self) -> None:
        """
        Display the log viewer dialog.
        
        Shows application logs for debugging and troubleshooting purposes.
        """
        from ui.layout.log_viewer import LogViewerDialog
        from app.utils.path_utils import get_logs_dir
        
        # Create and show log viewer (modeless - allows interaction with main window)
        # Keep reference to prevent garbage collection
        if not hasattr(self, 'log_viewer_window') or not self.log_viewer_window:
            self.log_viewer_window = LogViewerDialog(logs_dir=str(get_logs_dir()), parent=self.parent_window)
        
        self.log_viewer_window.show()
        self.log_viewer_window.raise_()
        self.log_viewer_window.activateWindow()
    
    def show_full_test_dialog(self) -> None:
        """
        Display the full test dialog.
        
        Provides a comprehensive testing interface for validating
        OCR, translation, and overlay functionality.
        """
        from ui.dialogs.full_pipeline_test_dialog import show_full_pipeline_test
        
        # Get pipeline reference from parent
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        # Show test dialog
        show_full_pipeline_test(
            parent=self.parent_window,
            pipeline=pipeline,
            config_manager=self.config_manager
        )
    
    def show_image_processing_dialog(self) -> None:
        """
        Display the image processing dialog for batch translating static images.
        """
        from ui.dialogs.image_processing import show_image_processing_dialog
        
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        show_image_processing_dialog(
            parent=self.parent_window,
            pipeline=pipeline,
            config_manager=self.config_manager
        )

    def show_benchmark_dialog(self) -> None:
        """
        Display the benchmark dialog for running pipeline benchmarks.
        """
        try:
            from ui.dialogs.benchmark_dialog import show_benchmark_dialog
        except ImportError:
            logger.warning("Benchmark dialog module not available")
            return

        pipeline = getattr(self.parent_window, 'pipeline', None)
        show_benchmark_dialog(
            parent=self.parent_window,
            pipeline=pipeline,
            config_manager=self.config_manager,
        )
    
    def toggle_theme(self, is_dark_mode=None) -> None:
        """
        Toggle between light and dark themes.
        
        Args:
            is_dark_mode: True for dark mode, False for light mode, None to toggle
        
        Switches the application theme and emits the theme_toggled signal
        to notify other components of the change.
        """
        from pathlib import Path
        from PyQt6.QtWidgets import QApplication, QMessageBox
        
        try:
            # If no mode specified, toggle current theme
            if is_dark_mode is None:
                is_dark_mode = not self._current_theme == "dark"
            
            # Determine which stylesheet to load
            if is_dark_mode:
                stylesheet_path = Path(__file__).parent.parent.parent / "app" / "styles" / "dark.qss"
                theme_name = "dark"
            else:
                stylesheet_path = Path(__file__).parent.parent.parent / "app" / "styles" / "base.qss"
                theme_name = "light"
            
            # Load and apply stylesheet
            if stylesheet_path.exists():
                with open(stylesheet_path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                
                # Apply to application
                app = QApplication.instance()
                app.setStyleSheet(stylesheet)
                
                # Update current theme
                self._current_theme = theme_name
                
                # Save preference
                self.config_manager.set_setting('ui.dark_mode', is_dark_mode)
                self.config_manager.save_config()
                
                self.theme_toggled.emit(theme_name)
                
                logger.info("Theme changed to %s mode", theme_name)
                
                if self.parent_window:
                    self.parent_window.statusBar().showMessage(
                        tr("status_switched_theme", theme=theme_name), 2000)
            else:
                error_msg = f"Could not find {theme_name} mode stylesheet:\n{stylesheet_path}"
                logger.error("Theme file not found: %s", stylesheet_path)
                QMessageBox.warning(
                    self.parent_window,
                    tr("theme_error"),
                    error_msg
                )
        except Exception as e:
            error_msg = f"Failed to change theme:\n\n{str(e)}"
            logger.error("Failed to change theme: %s", e)
            QMessageBox.critical(
                self.parent_window,
                tr("theme_error"),
                error_msg
            )
    
    def _load_tab_on_demand(self, tab_index_or_name) -> None:
        """
        Load a tab's content when it's first accessed (lazy loading).
        
        Args:
            tab_index_or_name: The index or name of the tab to load
        
        This internal method is called when a tab is first accessed,
        creating its content on-demand to improve startup performance.
        """
        tab_names = [
            "general", "dictionary_context", "pipeline",
            "overlay", "storage", "advanced",
        ]
        
        if isinstance(tab_index_or_name, int):
            tab_index = tab_index_or_name
            if 0 <= tab_index < len(tab_names):
                tab_name = tab_names[tab_index]
            else:
                return
        else:
            tab_name = tab_index_or_name
            try:
                tab_index = tab_names.index(tab_name)
            except ValueError:
                return
        
        # Check if already loaded — CRITICAL to prevent infinite loop
        if self._loaded_tabs.get(tab_name, False):
            return
        
        # Mark as loading IMMEDIATELY to prevent re-entry
        self._loaded_tabs[tab_name] = True
        
        logger.debug("Loading tab: %s (index %d)", tab_name, tab_index)
        
        tab_widget = None
        if tab_name == "general":
            tab_widget = self.create_general_tab()
        elif tab_name == "dictionary_context":
            tab_widget = self.create_dictionary_context_tab()
        elif tab_name == "pipeline":
            tab_widget = self.create_pipeline_tab()
        elif tab_name == "overlay":
            tab_widget = self.create_overlay_tab()
        elif tab_name == "storage":
            tab_widget = self.create_storage_tab()
        elif tab_name == "advanced":
            tab_widget = self.create_advanced_tab()
        
        if tab_widget and self._tab_widget:
            logger.debug("Tab widget created: %s", type(tab_widget).__name__)
            
            tab_display_name = self._tab_widget.tabText(tab_index)
            
            self._tab_widget.blockSignals(True)
            
            old_widget = self._tab_widget.widget(tab_index)
            self._tab_widget.removeTab(tab_index)
            self._tab_widget.insertTab(tab_index, tab_widget, tab_display_name)
            
            self._tab_widget.setCurrentIndex(tab_index)
            
            self._tab_widget.blockSignals(False)
            
            if old_widget:
                old_widget.deleteLater()
            
            logger.debug("Tab replaced successfully")
            
            self._tabs[tab_name] = tab_widget
            
            if self.parent_window and self.parent_window.settings_coordinator:
                self.parent_window.settings_coordinator.register_tab(tab_name, tab_widget)
                logger.debug("Registered %s with SettingsCoordinator", tab_name)
        else:
            logger.error("Failed to create tab widget for %s", tab_name)
            self._loaded_tabs[tab_name] = False
        
        logger.debug("Tab %s loading complete", tab_name)
    
    def create_general_tab(self):
        """Create General settings tab using the modular PyQt6 implementation."""
        from ui.settings.general_tab import GeneralSettingsTab
        
        # Create the modular general settings tab
        general_tab = GeneralSettingsTab(config_manager=self.config_manager, parent=self.parent_window)
        
        # Load configuration
        general_tab.load_config()
        
        if self.parent_window:
            general_tab.settingChanged.connect(self.parent_window.on_settings_changed)
            general_tab.source_lang_combo.currentTextChanged.connect(self.parent_window._sync_sidebar_languages)
            general_tab.target_lang_combo.currentTextChanged.connect(self.parent_window._sync_sidebar_languages)
            # Bidirectional language ↔ engine validation
            general_tab.source_lang_combo.currentTextChanged.connect(
                self.parent_window._on_source_language_changed
            )
            general_tab.target_lang_combo.currentTextChanged.connect(
                self.parent_window._on_target_language_changed
            )
            self.parent_window._sync_sidebar_languages()
        
        return general_tab
    
    def create_capture_tab(self):
        """Create Capture settings tab using the modular PyQt6 implementation."""
        from ui.settings.capture_tab import CaptureSettingsTab
        
        # Create the modular capture settings tab
        capture_tab = CaptureSettingsTab(config_manager=self.config_manager, parent=self.parent_window)
        
        capture_tab.load_config()
        
        if self.parent_window:
            capture_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return capture_tab
    
    def create_ocr_tab(self):
        """Create OCR Engines settings tab using the modular PyQt6 implementation."""
        from ui.settings.ocr.ocr_tab import OCRSettingsTab
        
        # Create the modular OCR settings tab
        ocr_tab = OCRSettingsTab(config_manager=self.config_manager, parent=self.parent_window)
        
        if self.parent_window:
            ocr_tab.pipeline = self.parent_window.pipeline
        
        ocr_tab.load_config()
        
        if self.parent_window:
            ocr_tab.settingChanged.connect(self.parent_window.on_settings_changed)
            ocr_tab.settingChanged.connect(self.parent_window.update_sidebar_ocr_display)
            # Bidirectional OCR engine ↔ language validation
            ocr_tab.ocrEngineChanged.connect(self.parent_window._on_ocr_engine_changed)
        
        if self.parent_window and self.parent_window.pipeline:
            ocr_tab._update_engine_statuses()
            # Refresh engine list to show discovered engines
            ocr_tab.refresh_engine_list()
        
        return ocr_tab
    
    def create_translation_tab(self):
        """Create Translation settings tab using the modular PyQt6 implementation."""
        from ui.settings.translation.translation_tab import TranslationSettingsTab
        
        # Get pipeline reference from parent
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        # Create the modular translation settings tab
        translation_tab = TranslationSettingsTab(
            config_manager=self.config_manager,
            pipeline=pipeline,  # Pass pipeline for dictionary management
            parent=self.parent_window
        )
        
        translation_tab.load_config()
        
        if self.parent_window:
            translation_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return translation_tab
    
    def create_llm_tab(self):
        """Create LLM settings tab for the optional LLM pipeline stage."""
        from ui.settings.llm.llm_tab import LLMSettingsTab

        pipeline = getattr(self.parent_window, 'pipeline', None)

        llm_tab = LLMSettingsTab(
            config_manager=self.config_manager,
            pipeline=pipeline,
            parent=self.parent_window
        )

        llm_tab.load_config()

        if self.parent_window:
            llm_tab.settingChanged.connect(self.parent_window.on_settings_changed)

        return llm_tab

    def create_overlay_tab(self):
        """Create Overlay settings tab using the modular PyQt6 implementation."""
        from ui.settings.overlay_tab import OverlaySettingsTab
        
        # Create the modular overlay settings tab
        overlay_tab = OverlaySettingsTab(config_manager=self.config_manager, parent=self.parent_window)
        
        overlay_tab.load_config()
        
        if self.parent_window:
            overlay_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return overlay_tab
    
    def create_smart_dictionary_tab(self):
        """Create Smart Dictionary settings tab using the modular PyQt6 implementation."""
        from ui.settings.dictionary_and_context import SmartDictionaryTab
        
        # Get pipeline reference from parent
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        # Create the modular smart dictionary settings tab
        smart_dictionary_tab = SmartDictionaryTab(
            config_manager=self.config_manager,
            pipeline=pipeline,
            parent=self.parent_window
        )
        
        smart_dictionary_tab.load_config()
        
        if self.parent_window:
            smart_dictionary_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return smart_dictionary_tab
    
    def create_context_tab(self):
        """Create Context Manager tab — domain-aware translation intelligence."""
        from ui.settings.dictionary_and_context import ContextManagerTab

        context_tab = ContextManagerTab(
            config_manager=self.config_manager,
            parent=self.parent_window
        )

        # If pipeline is already loaded, initialise the plugin now
        pipeline = getattr(self.parent_window, 'pipeline', None)
        if pipeline:
            context_tab.set_pipeline(pipeline)

        context_tab.load_config()

        if self.parent_window:
            context_tab.settingChanged.connect(self.parent_window.on_settings_changed)

        return context_tab
    
    def create_dictionary_context_tab(self):
        """Create the merged Dictionary & Context tab."""
        from ui.settings.dictionary_and_context import DictionaryContextTab

        pipeline = getattr(self.parent_window, 'pipeline', None)

        tab = DictionaryContextTab(
            config_manager=self.config_manager,
            pipeline=pipeline,
            parent=self.parent_window,
        )

        if pipeline:
            tab.set_pipeline(pipeline)

        tab.load_config()

        if self.parent_window:
            tab.settingChanged.connect(self.parent_window.on_settings_changed)

        return tab

    def create_pipeline_tab(self):
        """Create Pipeline hub tab — central hub with 8 sub-tabs."""
        from ui.settings.pipeline import PipelineManagementTab
        
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        pipeline_tab = PipelineManagementTab(
            config_manager=self.config_manager,
            pipeline=pipeline,
            parent=self.parent_window
        )
        
        if pipeline:
            pipeline_tab.set_pipeline(pipeline)
        
        pipeline_tab.load_config()
        
        if self.parent_window:
            pipeline_tab.settingChanged.connect(self.parent_window.on_settings_changed)

            # Wire signals from embedded stage sub-tabs.
            # The OCR sub-tab needs extra connections for engine-change
            # validation and sidebar refresh — these are wired lazily
            # when the sub-tab is first accessed via its property accessor.
        
        return pipeline_tab
    
    def create_storage_tab(self):
        """Create Storage settings tab using the modular PyQt6 implementation."""
        from ui.settings.storage import StorageSettingsTab
        
        # Get pipeline reference from parent
        pipeline = getattr(self.parent_window, 'pipeline', None)
        
        # Create the modular storage settings tab
        storage_tab = StorageSettingsTab(
            config_manager=self.config_manager,
            pipeline=pipeline,  # Pass pipeline for dictionary management
            parent=self.parent_window
        )
        
        storage_tab.load_config()
        
        if self.parent_window:
            storage_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return storage_tab
    
    def create_advanced_tab(self):
        """Create Advanced settings tab using the modular PyQt6 implementation."""
        from ui.settings.advanced_tab import AdvancedSettingsTab
        
        # Create the modular advanced settings tab
        advanced_tab = AdvancedSettingsTab(config_manager=self.config_manager, parent=self.parent_window)
        
        advanced_tab.load_config()
        
        if self.parent_window:
            advanced_tab.settingChanged.connect(self.parent_window.on_settings_changed)
        
        return advanced_tab
    
    # Maps old standalone tab names to pipeline hub property accessors
    _EMBEDDED_TAB_NAMES = {
        'capture': 'capture_tab',
        'ocr': 'ocr_tab',
        'translation': 'translation_tab',
        'llm': 'llm_tab',
    }

    def get_tab(self, tab_name: str) -> Any | None:
        """
        Get a reference to a specific tab.
        
        Args:
            tab_name: The name of the tab to retrieve.
                      Accepts both new top-level names (``"pipeline"``,
                      ``"dictionary_context"``) and legacy names
                      (``"ocr"``, ``"translation"``, etc.) — the latter
                      are resolved to the pipeline hub's embedded sub-tabs.
        
        Returns:
            The tab widget, or None if not found
        """
        # Direct top-level tab lookup
        tab = self._tabs.get(tab_name)
        if tab is not None:
            return tab

        # Legacy name → embedded sub-tab inside the pipeline hub
        accessor = self._EMBEDDED_TAB_NAMES.get(tab_name)
        if accessor:
            pipeline_hub = self._tabs.get('pipeline')
            if pipeline_hub is not None:
                return getattr(pipeline_hub, accessor, None)

        # Legacy dictionary / context as standalone names
        if tab_name in ('smart_dictionary', 'context'):
            dc_tab = self._tabs.get('dictionary_context')
            if dc_tab is not None:
                if tab_name == 'smart_dictionary':
                    return getattr(dc_tab, 'dictionary_tab', None)
                return getattr(dc_tab, 'context_tab', None)

        return None

    def is_tab_loaded(self, tab_name: str) -> bool:
        """Check whether a tab has been lazily loaded."""
        return self._loaded_tabs.get(tab_name, False)

    def reload_tab(self, tab_name: str) -> None:
        """
        Force a tab to be recreated (e.g. after pipeline becomes available).

        Resets the loaded flag and triggers on-demand loading so the tab
        is rebuilt with the current application state.
        """
        if tab_name not in self._loaded_tabs:
            return

        if not self._loaded_tabs.get(tab_name, False):
            # Tab was never loaded — nothing to reload
            return

        logger.debug("Reloading tab: %s", tab_name)

        if self.parent_window and self.parent_window.settings_coordinator:
            self.parent_window.settings_coordinator.unregister_tab(tab_name)

        # Reset loaded flag so _load_tab_on_demand will recreate it
        self._loaded_tabs[tab_name] = False
        self._tabs[tab_name] = None

        # Trigger recreation
        self._load_tab_on_demand(tab_name)
    
    def get_all_tabs(self) -> dict[str, Any]:
        """
        Get references to all tabs.
        
        Returns:
            Dictionary mapping tab names to tab widgets
        """
        return self._tabs.copy()
    
    def get_toolbar(self) -> Any | None:
        """
        Get reference to the toolbar.
        
        Returns:
            The toolbar widget, or None if not created
        """
        return self._toolbar
    
    def get_sidebar(self) -> Any | None:
        """
        Get reference to the sidebar.
        
        Returns:
            The sidebar widget, or None if not created
        """
        return self._sidebar
    
