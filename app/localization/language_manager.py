"""
Language Manager - Central language change coordination

Provides a signal-based system for notifying all UI components
when the language changes, enabling dynamic translation updates.
"""

import logging

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class LanguageManager(QObject):
    """
    Manages language changes and notifies all registered components.
    
    Emits language_changed signal when language is changed,
    allowing UI components to update their translations dynamically.
    
    Language state is owned by JSONTranslator — this class only
    coordinates the Qt signal broadcast.
    """
    
    language_changed = pyqtSignal(str)  # Emits new language code
    
    def set_language(self, lang_code: str):
        """
        Set the current language and notify all listeners.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'fr')
        """
        from app.localization.json_translator import get_current_language, set_language

        old_language = get_current_language()
        if lang_code == old_language:
            return

        # Update the translation system (single source of truth)
        set_language(lang_code)

        # Only emit if the translator actually changed
        if get_current_language() == lang_code:
            self.language_changed.emit(lang_code)
            logger.info("Language changed: %s -> %s", old_language, lang_code)


# Global singleton instance
_language_manager = None


def get_language_manager() -> LanguageManager:
    """Get the global language manager instance."""
    global _language_manager
    if _language_manager is None:
        _language_manager = LanguageManager()
    return _language_manager
