"""
Translation Layer package.

Re-exports TranslationFacade as TranslationLayer for backward compatibility.
"""

from app.text_translation.layer.facade import TranslationFacade, create_translation_layer

# Backward-compatible alias
TranslationLayer = TranslationFacade

__all__ = ["TranslationFacade", "TranslationLayer", "create_translation_layer"]
