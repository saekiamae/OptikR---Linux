"""
Settings Coordinator for the application.

This module coordinates settings operations across all settings tabs,
including save, load, validation, import, and export functionality.
"""

import logging
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Any
from .interfaces import SettingsTab

logger = logging.getLogger(__name__)


class SettingsCoordinator(QObject):
    """
    Coordinates settings operations across all tabs.
    
    Responsibilities:
    - Coordinating save operations across all tabs
    - Coordinating load/reload operations across all tabs
    - Settings validation before save
    - Import/export functionality
    - Change tracking and notification
    
    The SettingsCoordinator ensures consistent settings management
    across the application and provides a single point of control
    for settings operations.
    """
    
    # Signals for settings events
    settings_changed = pyqtSignal(str, object)  # Emitted when any setting changes (key, value)
    settings_saved = pyqtSignal(bool)  # Emitted when settings save completes (success)
    settings_loaded = pyqtSignal()  # Emitted when all settings are loaded
    save_failed = pyqtSignal(list)  # Emitted when save fails (list of error messages)
    
    def __init__(self, config_manager=None, parent=None):
        """
        Initialize the Settings Coordinator.
        
        Args:
            config_manager: The configuration manager instance
            parent: The parent QObject (typically the main window)
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.parent_window = parent
        
        # Tab registry
        self._tabs: dict[str, SettingsTab] = {}
        
        # Change tracking
        self._has_unsaved_changes = False
        self._original_state: dict[str, Any] | None = None
    
    def register_tab(self, tab_name: str, tab: SettingsTab) -> None:
        """
        Register a settings tab with the coordinator.
        
        Args:
            tab_name: Unique identifier for the tab
            tab: The settings tab instance implementing SettingsTab protocol
        
        All tabs must be registered before save_all or load_all operations.
        """
        self._tabs[tab_name] = tab
    
    def unregister_tab(self, tab_name: str) -> None:
        """
        Unregister a settings tab from the coordinator.
        
        Args:
            tab_name: Unique identifier for the tab to unregister
        """
        if tab_name in self._tabs:
            del self._tabs[tab_name]

    def notify_setting_changed(self, key: str, value: Any, source_tab: str = '') -> None:
        """
        Notify all registered tabs that a setting has changed.

        This enables cross-tab synchronization: when one tab changes a setting,
        all other tabs that display the same setting can update their UI.

        Args:
            key: The config key that changed (dot notation, e.g. 'plugins.context_manager.enabled')
            value: The new value
            source_tab: Name of the tab that made the change (will be skipped during notification)
        """
        for tab_name, tab in self._tabs.items():
            if tab_name == source_tab:
                continue
            if hasattr(tab, 'on_setting_changed'):
                try:
                    tab.on_setting_changed(key, value)
                except Exception as e:
                    logger.warning("Failed to notify %s of setting change %s: %s", tab_name, key, e)

        # Emit the coordinator signal too
        self.settings_changed.emit(key, value)

    
    def save_all(self) -> tuple[bool, list[str]]:
        """
        Save settings from all registered tabs.
        
        Returns:
            tuple[bool, list[str]]: A tuple containing:
                - success (bool): True if all saves succeeded, False otherwise
                - errors (list[str]): List of error messages (empty if success)
        
        This method:
        1. Validates all tabs before saving
        2. Collects save results from all tabs
        3. Aggregates error messages
        4. Emits settings_saved signal on success or save_failed on failure
        
        If any tab fails to save, the method returns False and provides
        detailed error information.
        """
        try:
            # Validate all tabs before saving
            for tab_name, tab in self._tabs.items():
                if not tab.validate():
                    # Validation failed, don't save
                    error_msg = f"{tab_name.replace('_', ' ').title()} settings validation failed"
                    self.save_failed.emit([error_msg])
                    return False, [error_msg]
            
            # All validations passed, proceed with saving
            # Collect save results from all tabs
            errors = []
            
            for tab_name, tab in self._tabs.items():
                result = tab.save_config()
                success, error_msg = result if isinstance(result, tuple) else (True, "")
                if not success:
                    errors.append(f"{tab_name.replace('_', ' ').title()} settings: {error_msg}")
                else:
                    if tab_name == 'pipeline' and self.parent_window:
                        pipeline = self.parent_window.pipeline
                        if pipeline and hasattr(pipeline, 'overlay_system') and pipeline.overlay_system:
                            pipeline.overlay_system.reload_config()
            
            # Check if any saves failed
            if errors:
                self.save_failed.emit(errors)
                return False, errors
            
            # All saves succeeded
            self.mark_saved()
            
            # Capture new baseline state after successful save
            self.capture_original_state()
            
            self.settings_saved.emit(True)
            
            logger.info("All settings saved successfully")
            
            return True, []
            
        except Exception as e:
            error_msg = f"Failed to save settings: {str(e)}"
            logger.error("Failed to save settings: %s", e)
            
            self.save_failed.emit([error_msg])
            return False, [error_msg]
    
    def load_all(self) -> None:
        """
        Load settings into all registered tabs.
        
        Calls load_config() on each registered tab to populate UI
        elements with current configuration values.
        
        Emits settings_loaded signal when complete.
        """
        try:
            # Reload all registered tabs
            for tab_name, tab in self._tabs.items():
                tab.load_config()
                
                if tab_name == 'pipeline' and self.parent_window:
                    pipeline = self.parent_window.pipeline
                    if pipeline and hasattr(pipeline, 'overlay_system') and pipeline.overlay_system:
                        pipeline.overlay_system.reload_config()
            
            if self.parent_window:
                self.parent_window._sync_sidebar_languages()
                self.parent_window.update_sidebar_ocr_display()
                
                pipeline = self.parent_window.pipeline
                if pipeline:
                    if self.config_manager:
                        regions_data = self.config_manager.get_setting('capture.regions', [])
                        
                        if regions_data:
                            from app.models import MultiRegionConfig
                            config = MultiRegionConfig.from_dict({
                                'regions': regions_data,
                            })
                            pipeline.set_multi_region_config(config)
                        else:
                            region_type = self.config_manager.get_setting('capture.region', None)
                            if region_type == 'custom' and hasattr(self.parent_window, '_restore_capture_region_from_config'):
                                self.parent_window._restore_capture_region_from_config()
            
            # Clear unsaved changes flag
            self.mark_saved()
            
            # Capture baseline state after loading
            self.capture_original_state()
            
            # Emit signal
            self.settings_loaded.emit()
            
            logger.info("All settings reloaded")
                
        except Exception as e:
            logger.error("Failed to reload settings: %s", e)
            raise
    

    
    def export_settings(self, file_path: str) -> tuple[bool, str]:
        """
        Export all settings to a file.
        
        Args:
            file_path: Path where settings should be exported
        
        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if export succeeded, False otherwise
                - error_message (str): Empty string on success, error details on failure
        
        Exports the complete configuration state to a JSON file that can
        be imported later or shared with other users.
        
        Emits settings_exported signal on success.
        """
        try:
            import json
            from datetime import datetime
            
            if not self.config_manager:
                return False, "Config manager not available"
            
            # Get current configuration
            config_data = self.config_manager.config
            
            # Add metadata
            import app
            version = app.__version__
            export_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'version': version,
                    'application': 'OptikR'
                },
                'settings': config_data
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Settings exported to %s", file_path)
            
            return True, ""
            
        except Exception as e:
            error_msg = f"Failed to export settings: {str(e)}"
            logger.error("Failed to export settings: %s", e)
            
            return False, error_msg
    
    def import_settings(self, file_path: str) -> tuple[bool, str]:
        """
        Import settings from a file.
        
        Args:
            file_path: Path to the settings file to import
        
        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if import succeeded, False otherwise
                - error_message (str): Empty string on success, error details on failure
        
        Imports configuration from a JSON file and updates all registered
        tabs with the new values.
        
        Emits settings_imported signal on success.
        """
        try:
            import json
            
            if not self.config_manager:
                return False, "Config manager not available"
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Extract settings (support both wrapped and raw formats)
            if 'settings' in import_data and 'metadata' in import_data:
                settings = import_data['settings']
                metadata = import_data.get('metadata', {})
            else:
                settings = import_data
                metadata = {}
            
            logger.info("Importing settings (exported %s)", metadata.get('export_date', 'unknown date'))
            
            # Recursively flatten nested config to dot-notation key/value pairs
            flat = self._flatten_config(settings)
            
            # Schema validation: reject values that violate schema constraints
            schema = getattr(self.config_manager, 'schema', None)
            validation_warnings: list[str] = []
            if schema is not None:
                for key, value in list(flat.items()):
                    is_valid, error_msg = schema.validate(key, value)
                    if not is_valid:
                        validation_warnings.append(f"{key}: {error_msg}")
                        del flat[key]
            
            if validation_warnings:
                logger.warning(
                    "Import skipped %d invalid values: %s",
                    len(validation_warnings),
                    "; ".join(validation_warnings[:10]),
                )
            
            # Apply validated settings
            for key, value in flat.items():
                self.config_manager.set_setting(key, value)
            
            # Save configuration
            self.config_manager.save_config()
            
            # Reload all registered tabs
            self.load_all()
            
            logger.info("Settings imported from %s", file_path)
            
            return True, ""
            
        except KeyError as e:
            error_msg = f"Invalid settings file format (missing key {e})"
            logger.error("Failed to import settings: %s", error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Failed to import settings: {str(e)}"
            logger.error("Failed to import settings: %s", e)
            
            return False, error_msg

    @staticmethod
    def _flatten_config(obj: dict, prefix: str = '') -> dict:
        """Recursively flatten a nested dict into dot-notation key/value pairs.

        Lists and other non-dict values are kept as leaf values.
        """
        items: dict = {}
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(SettingsCoordinator._flatten_config(value, full_key))
            else:
                items[full_key] = value
        return items
    
    def mark_changed(self) -> None:
        """
        Mark that settings have been changed.
        
        Sets the unsaved changes flag and emits settings_changed signal.
        This is typically called when a user modifies any setting in any tab.
        """
        self._has_unsaved_changes = True
        # Emit with empty key and None value for general change notification
        self.settings_changed.emit("", None)
    
    def check_for_changes(self) -> bool:
        """
        Check if any registered tab has actual changes.
        
        Returns:
            bool: True if any tab has changes, False otherwise
        
        This method checks each registered tab for changes by comparing
        current state with original state (if available).
        """
        has_changes = False
        
        # Check each registered tab for actual changes
        for tab_name, tab in self._tabs.items():
            # Check if tab has state tracking methods
            if hasattr(tab, '_get_current_state') and hasattr(tab, '_original_state'):
                current = tab._get_current_state()
                original = tab._original_state
                if current != original:
                    has_changes = True
                    break
        
        if has_changes and not self._has_unsaved_changes:
            logger.info("Settings changed — save button enabled")
        
        self._has_unsaved_changes = has_changes
        
        # Emit signal if state changed
        if has_changes:
            # Emit with empty key and None value for general change notification
            self.settings_changed.emit("", None)
        
        return has_changes
    
    def mark_saved(self) -> None:
        """
        Mark that settings have been saved.
        
        Clears the unsaved changes flag. This is called after a successful
        save_all() operation.
        """
        self._has_unsaved_changes = False
    
    def has_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes.
        
        Returns:
            bool: True if there are unsaved changes, False otherwise
        """
        return self._has_unsaved_changes
    
    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """
        Get the current state of all registered tabs.
        
        Returns:
            dict[str, dict[str, Any]]: Dictionary mapping tab names to their states
        
        This is useful for:
        - Comparing states for change detection
        - Debugging and logging
        - Creating backups before changes
        """
        states = {}
        for tab_name, tab in self._tabs.items():
            try:
                states[tab_name] = tab.get_state()
            except Exception as e:
                # If a tab doesn't support get_state, skip it
                states[tab_name] = {"error": str(e)}
        return states
    
    def capture_original_state(self) -> None:
        """
        Capture the current state as the original state.
        
        This is useful for detecting changes and implementing cancel/revert
        functionality. Call this after loading settings or after a successful save.
        """
        self._original_state = self.get_all_states()
    
    def restore_original_state(self) -> None:
        """
        Restore settings to the original captured state.
        
        Reverts all tabs to the state captured by capture_original_state().
        This is useful for implementing cancel/revert functionality in settings dialogs.
        
        Example usage:
            # When settings dialog opens:
            coordinator.capture_original_state()
            
            # If user clicks "Cancel":
            coordinator.restore_original_state()
            
            # If user clicks "Save":
            coordinator.save_all()  # This will capture new baseline automatically
        """
        if self._original_state is None:
            logger.warning("Cannot restore state: no original state captured")
            return
        
        if not self.config_manager:
            logger.warning("Cannot restore state: config manager not available")
            return
        
        try:
            # Restore config values from captured state
            for tab_name, tab_state in self._original_state.items():
                # Skip tabs that had errors during state capture
                if "error" in tab_state:
                    continue
                
                # Update config manager with original values
                for key, value in tab_state.items():
                    try:
                        self.config_manager.set_setting(key, value)
                    except Exception as e:
                        logger.warning(f"Failed to restore setting {key}: {e}")
            
            # Save restored config to disk
            self.config_manager.save_config()
            
            # Reload all tabs to reflect restored values
            self.load_all()
            
            logger.info("Settings reverted to original state")
            
        except Exception as e:
            logger.error("Failed to restore original state: %s", e)
    
    def get_registered_tabs(self) -> list[str]:
        """
        Get the list of registered tab names.
        
        Returns:
            list[str]: List of registered tab names
        """
        return list(self._tabs.keys())
