"""
Translatable Mixin - Auto-updating translation support for widgets

Provides a mixin class that allows widgets to automatically update
their text when the language changes.
"""

import logging
import weakref

logger = logging.getLogger(__name__)

# Supported setter methods in priority order (used when method= is not specified)
_SETTER_PRIORITY = ("setText", "setTitle", "setWindowTitle", "setPlaceholderText")


class TranslatableMixin:
    """
    Mixin for widgets that need automatic translation updates.

    Usage:
        class MyWidget(TranslatableMixin, QWidget):
            def __init__(self):
                super().__init__()

                label = QLabel()
                self.set_translatable_text(label, "my_translation_key")

                # With formatting
                self.set_translatable_text(label, "welcome_message", name="User")

                # Target a specific setter (e.g. placeholder instead of setText)
                self.set_translatable_text(line_edit, "search_hint", method="setPlaceholderText")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each entry: weakref -> (key, format_args, method_name_or_None)
        # Window title uses the sentinel _TITLE_SENTINEL instead of a weakref.
        self._translation_entries: list[tuple] = []
        self._connected_to_language_manager = False

    def _ensure_language_connection(self):
        """Ensure we're connected to language change signals."""
        if not self._connected_to_language_manager:
            from app.localization.language_manager import get_language_manager
            language_manager = get_language_manager()
            language_manager.language_changed.connect(self._update_translations)
            self._connected_to_language_manager = True

    def set_translatable_text(self, widget, key: str, method: str | None = None, **format_args):
        """
        Set text that will auto-update on language change.

        Args:
            widget: The widget to update (QLabel, QPushButton, QGroupBox, etc.)
            key: Translation key
            method: Explicit setter name (e.g. "setPlaceholderText"). If None,
                    auto-detects from _SETTER_PRIORITY.
            **format_args: Optional formatting arguments for the translation
        """
        self._ensure_language_connection()
        ref = weakref.ref(widget)
        # Remove any previous entry for the same widget+method combo
        self._translation_entries = [
            e for e in self._translation_entries
            if not (e[0] is not _TITLE_SENTINEL and e[0]() is widget and e[3] == method)
        ]
        self._translation_entries.append((ref, key, format_args, method))
        self._apply_text(widget, key, format_args, method)

    def set_translatable_title(self, key: str, **format_args):
        """
        Set window title that will auto-update on language change.

        Args:
            key: Translation key
            **format_args: Optional formatting arguments for the translation
        """
        self._ensure_language_connection()
        # Remove any previous title entry
        self._translation_entries = [
            e for e in self._translation_entries if e[0] is not _TITLE_SENTINEL
        ]
        self._translation_entries.append((_TITLE_SENTINEL, key, format_args, None))
        self._apply_title(key, format_args)

    def _apply_text(self, widget, key: str, format_args: dict, method: str | None):
        """Apply translated text to a widget."""
        from app.localization.json_translator import tr

        try:
            text = tr(key, **format_args)
            if method:
                getter = getattr(widget, method, None)
                if getter:
                    getter(text)
                else:
                    logger.debug("Widget %r has no method '%s'", widget, method)
            else:
                for setter_name in _SETTER_PRIORITY:
                    if hasattr(widget, setter_name):
                        getattr(widget, setter_name)(text)
                        break
        except RuntimeError:
            # Widget's C++ object already deleted — skip silently
            pass
        except Exception as e:
            logger.debug("Failed to update widget text for key '%s': %s", key, e)

    def _apply_title(self, key: str, format_args: dict):
        """Update window title."""
        from app.localization.json_translator import tr

        try:
            text = tr(key, **format_args)
            if hasattr(self, "setWindowTitle"):
                self.setWindowTitle(text)
        except Exception as e:
            logger.debug("Failed to update window title for key '%s': %s", key, e)

    def _update_translations(self, lang_code: str):
        """Update all registered widgets when language changes."""
        live_entries = []
        for entry in self._translation_entries:
            ref_or_sentinel, key, format_args, method = entry
            if ref_or_sentinel is _TITLE_SENTINEL:
                self._apply_title(key, format_args)
                live_entries.append(entry)
            else:
                widget = ref_or_sentinel()
                if widget is not None:
                    self._apply_text(widget, key, format_args, method)
                    live_entries.append(entry)
                # else: widget was garbage-collected, drop the entry
        self._translation_entries = live_entries

    def clear_translations(self):
        """Clear all registered translations (useful for cleanup)."""
        self._translation_entries.clear()


# Sentinel object for window title entries (not a weakref)
_TITLE_SENTINEL = object()
