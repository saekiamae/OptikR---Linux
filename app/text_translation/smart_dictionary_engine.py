"""
SmartDictionary → AbstractTranslationEngine adapter.

Wraps a ``SmartDictionary`` instance as a registered translation engine
so that ``TranslationFacade.translate()`` can perform dictionary lookups
before falling through to heavy AI engines.
"""

import time
import logging
from typing import Any

from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine,
    TranslationOptions,
    TranslationResult,
)

logger = logging.getLogger(__name__)


class SmartDictionaryEngine(AbstractTranslationEngine):
    """Adapts a ``SmartDictionary`` for use as a ``TranslationFacade`` engine.

    Registered under the name ``"dictionary"`` so that the facade's
    built-in dictionary check (before calling the main AI engine) finds
    and uses it automatically.
    """

    def __init__(self, dictionary: Any = None) -> None:
        super().__init__(engine_name="dictionary")
        self._dictionary = dictionary
        self._is_initialized = dictionary is not None

    def set_dictionary(self, dictionary: Any) -> None:
        """Hot-swap the backing ``SmartDictionary`` instance."""
        self._dictionary = dictionary
        self._is_initialized = dictionary is not None

    def initialize(self, config: dict[str, Any]) -> bool:
        self._is_initialized = self._dictionary is not None
        return self._is_initialized

    def translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        options: TranslationOptions | None = None,
    ) -> TranslationResult:
        start = time.time()

        if not self._dictionary:
            return self._miss(text, src_lang, tgt_lang, 0.0)

        try:
            entry = self._dictionary.lookup(text, src_lang, tgt_lang)
        except Exception as exc:
            logger.debug("Dictionary lookup error: %s", exc)
            return self._miss(text, src_lang, tgt_lang,
                              (time.time() - start) * 1000)

        elapsed_ms = (time.time() - start) * 1000

        if entry and entry.translation and entry.translation != text:
            return TranslationResult(
                original_text=text,
                translated_text=entry.translation,
                source_language=src_lang,
                target_language=tgt_lang,
                confidence=entry.get_effective_confidence(),
                engine_used=self.engine_name,
                processing_time_ms=elapsed_ms,
                from_cache=True,
            )

        return self._miss(text, src_lang, tgt_lang, elapsed_ms)

    def is_available(self) -> bool:
        return self._is_initialized and self._dictionary is not None

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        return True

    def get_supported_languages(self) -> list[str]:
        if not self._dictionary:
            return []
        try:
            pairs = self._dictionary.get_available_language_pairs()
            langs: set[str] = set()
            for src, tgt, _, _ in pairs:
                langs.add(src)
                langs.add(tgt)
            return list(langs)
        except Exception:
            return []

    def cleanup(self) -> None:
        pass

    @staticmethod
    def _miss(
        text: str, src_lang: str, tgt_lang: str, elapsed_ms: float,
    ) -> TranslationResult:
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence=0.0,
            engine_used="dictionary",
            processing_time_ms=elapsed_ms,
        )
