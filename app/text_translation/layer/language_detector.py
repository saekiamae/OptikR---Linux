"""
Language Detector for Translation Layer.

Extracts language detection logic into a dedicated module.
Uses LanguageDetector for multi-algorithm detection.

Requirements: 3.1
"""
import logging
from typing import Any

from app.text_translation.translation_engine_interface import (
    LanguageDetectionConfidence,
    LanguageDetectionResult,
    TranslationOptions,
    TranslationQuality,
)
from app.utils.language_detection import LanguageDetector


class LanguageDetectorService:
    """Config-aware language detection using LanguageDetector."""

    def __init__(self, config_manager: Any | None = None) -> None:
        self._logger = logging.getLogger(__name__)
        self._detector = LanguageDetector()
        self._config_manager = config_manager

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language of input text."""
        lang_code, confidence = self._detector.detect_language(text)
        return LanguageDetectionResult(
            language_code=lang_code,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
        )

    def parse_translation_options(
        self, options: dict[str, Any]
    ) -> TranslationOptions:
        """Parse options dictionary into TranslationOptions, reading from config if needed."""
        if self._config_manager:
            preserve_formatting = options.get("preserve_formatting")
            if preserve_formatting is None:
                preserve_formatting = self._config_manager.get_setting(
                    "translation.preserve_formatting", True
                )

            use_cache = options.get("use_cache")
            if use_cache is None:
                use_cache = self._config_manager.get_setting(
                    "translation.cache_enabled", True
                )

            quality = options.get("quality")
            if quality is None:
                quality_level = self._config_manager.get_setting(
                    "translation.quality_level", 70
                )
                if quality_level >= 75:
                    quality = "high"
                elif quality_level >= 25:
                    quality = "medium"
                else:
                    quality = "low"
        else:
            preserve_formatting = options.get("preserve_formatting", True)
            use_cache = options.get("use_cache", True)
            quality = options.get("quality", "medium")

        context = options.get("context")
        if context is None and self._config_manager:
            context = self._config_manager.get_setting("translation.context", "") or None
        if context is not None and isinstance(context, str) and not context.strip():
            context = None

        return TranslationOptions(
            quality=TranslationQuality(quality),
            preserve_formatting=preserve_formatting,
            use_cache=use_cache,
            timeout_seconds=options.get("timeout_seconds", 5.0),
            context=context,
            domain=options.get("domain"),
        )

    @staticmethod
    def _get_confidence_level(confidence: float) -> LanguageDetectionConfidence:
        """Convert numeric confidence to confidence level enum."""
        if confidence >= 0.9:
            return LanguageDetectionConfidence.VERY_HIGH
        elif confidence >= 0.7:
            return LanguageDetectionConfidence.HIGH
        elif confidence >= 0.5:
            return LanguageDetectionConfidence.MEDIUM
        elif confidence >= 0.3:
            return LanguageDetectionConfidence.LOW
        else:
            return LanguageDetectionConfidence.VERY_LOW
