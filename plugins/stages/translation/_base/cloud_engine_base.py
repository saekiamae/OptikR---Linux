"""
Cloud Translation Engine Base

Shared base class for all cloud/API-based translation engine plugins.
Handles the common boilerplate: timing, fallback results, batch-via-loop,
and consistent error handling.

Subclasses only need to implement:
  - _do_translate(text, src_lang, tgt_lang) -> str
  - initialize(config) -> bool
  - cleanup() -> None
  - get_supported_languages() -> list[str]
"""

import logging
import time
from abc import abstractmethod

from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine, TranslationOptions, TranslationResult,
    BatchTranslationResult,
)


class CloudTranslationEngine(AbstractTranslationEngine):
    """
    Base class for cloud/API translation engines.

    Provides the translate_text / translate_batch boilerplate so that
    concrete engines only implement _do_translate().
    """

    # Subclasses should set this to a value between 0.0 and 1.0.
    # Used for engine ranking when multiple engines are available.
    _default_confidence: float = 0.90

    def __init__(self, engine_name: str):
        super().__init__(engine_name)
        self._logger = logging.getLogger(f"{__name__}.{engine_name}")
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Perform the actual translation API call.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Translated text string

        Raises:
            Any exception on failure — the base class catches it.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fallback(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        """Return original text as a zero-confidence fallback."""
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence=0.0,
            engine_used=self.engine_name,
            processing_time_ms=0.0,
            from_cache=False,
        )

    # ------------------------------------------------------------------
    # AbstractTranslationEngine interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._is_initialized

    def translate_text(self, text: str, src_lang: str, tgt_lang: str,
                       options: TranslationOptions | None = None) -> TranslationResult:
        if not self.is_available():
            self._logger.warning("Engine not available")
            return self._make_fallback(text, src_lang, tgt_lang)

        start_time = time.time()
        try:
            translated_text = self._do_translate(text, src_lang, tgt_lang)
            processing_time = (time.time() - start_time) * 1000
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=src_lang,
                target_language=tgt_lang,
                confidence=self._default_confidence,
                engine_used=self.engine_name,
                processing_time_ms=processing_time,
                from_cache=False,
            )
        except Exception as e:
            self._logger.error(f"Translation failed: {e}")
            return self._make_fallback(text, src_lang, tgt_lang)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str,
                        options: TranslationOptions | None = None) -> BatchTranslationResult:
        start_time = time.time()
        results: list[TranslationResult] = []
        failed: list = []

        for i, text in enumerate(texts):
            result = self.translate_text(text, src_lang, tgt_lang, options)
            results.append(result)
            if result.confidence == 0.0:
                failed.append((i, "Translation failed"))

        total_time = (time.time() - start_time) * 1000
        return BatchTranslationResult(
            results=results,
            total_processing_time_ms=total_time,
            cache_hit_rate=0.0,
            failed_translations=failed,
        )

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        supported = self.get_supported_languages()
        return src_lang.lower() in [s.lower() for s in supported] and \
               tgt_lang.lower() in [s.lower() for s in supported]
