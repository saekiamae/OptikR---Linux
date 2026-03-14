"""
Interface definitions for the refactored architecture.

This module defines protocols and abstract base classes that establish
contracts between components in the application.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class MainWindowProtocol(Protocol):
    """
    Protocol defining what UIManager and SettingsCoordinator expect from
    the main window.

    Using a protocol instead of hasattr() checks makes the cross-component
    contract explicit and statically verifiable.
    """

    config_manager: Any
    startup_pipeline: Any
    settings_coordinator: Any
    ui_manager: Any

    @property
    def pipeline(self) -> Any: ...

    def toggle_translation(self) -> None: ...
    def show_capture_region_selector(self) -> None: ...
    def show_performance_monitor(self) -> None: ...
    def show_quick_ocr_switch(self) -> None: ...
    def save_all_settings(self) -> None: ...
    def import_settings(self) -> None: ...
    def export_settings(self) -> None: ...
    def on_settings_changed(self) -> None: ...
    def update_sidebar_ocr_display(self) -> None: ...
    def _sync_sidebar_languages(self) -> None: ...
    def _on_source_language_changed(self, language_name: str) -> None: ...
    def _on_target_language_changed(self, language_name: str) -> None: ...
    def _on_ocr_engine_changed(self, engine_name: str) -> None: ...
    def _on_preset_loaded(self, preset_name: str) -> None: ...
    def _on_content_mode_changed(self, mode: str) -> None: ...
    def _on_pipeline_mode_changed(self, mode: str) -> None: ...
    def statusBar(self) -> Any: ...


class SettingsTab(Protocol):
    """
    Protocol for settings tab widgets.
    
    All settings tabs must implement this interface to ensure consistent
    behavior across the application. This protocol defines the contract
    for loading, saving, validating, and retrieving state from settings tabs.
    """
    
    def load_config(self) -> None:
        """
        Load configuration from the config manager and update UI elements.
        
        This method should read the current configuration values and
        populate the tab's UI controls with those values.
        """
        ...
    
    def save_config(self) -> tuple[bool, str]:
        """
        Save configuration from UI elements to the config manager.
        
        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if save succeeded, False otherwise
                - error_message (str): Empty string on success, error details on failure
        
        The implementation should:
        1. Read values from UI controls
        2. Validate the values
        3. Update the config manager
        4. Persist to disk
        5. Handle any errors gracefully
        """
        ...
    
    def validate(self) -> bool:
        """
        Validate the current settings in the tab.
        
        Returns:
            bool: True if all settings are valid, False otherwise
        
        This method should check that all user inputs are valid before
        attempting to save. It should display appropriate error messages
        to the user if validation fails.
        """
        ...
    
    def get_state(self) -> dict:
        """
        Get the current state of the tab as a dictionary.
        
        Returns:
            dict: A dictionary containing all current settings in the tab
        
        This method is useful for:
        - Exporting settings
        - Comparing states for change detection
        - Debugging and logging
        """
        ...


class PipelineComponent(ABC):
    """
    Abstract base class for pipeline components.
    
    Pipeline components are responsible for managing the translation pipeline,
    including OCR, translation, and overlay functionality. All pipeline
    components must inherit from this class and implement its abstract methods.
    """
    
    @abstractmethod
    def start(self) -> None:
        """
        Start the pipeline component.
        
        This method should:
        1. Initialize any required resources
        2. Start background threads or processes
        3. Begin processing operations
        4. Emit appropriate signals to notify listeners
        
        Raises:
            RuntimeError: If the component is already running
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the pipeline component.
        
        This method should:
        1. Stop all background operations gracefully
        2. Clean up resources
        3. Emit appropriate signals to notify listeners
        
        The method should be idempotent - calling it multiple times
        should not cause errors.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the pipeline component is currently running.
        
        Returns:
            bool: True if the component is running, False otherwise
        
        This method should return the current operational state of the
        component without side effects.
        """
        pass
