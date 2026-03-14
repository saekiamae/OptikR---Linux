"""
Translation Cache Adapter.

Wraps the existing TranslationCache for use by the TranslationFacade,
providing a thin delegation layer.

Requirements: 3.1
"""
import logging
from pathlib import Path
from typing import Any

from app.text_translation.translation_engine_interface import (
    TranslationCache,
    TranslationOptions,
)


class TranslationCacheAdapter:
    """Adapter around TranslationCache for the translation facade."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self._logger = logging.getLogger(__name__)
        self._cache = TranslationCache(max_size=max_size, ttl_seconds=ttl_seconds)

    # -- public API -----------------------------------------------------------

    def generate_key(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        engine: str,
        options: TranslationOptions | None = None,
    ) -> tuple:
        """Pre-compute a cache key for use with :meth:`get_by_key` / :meth:`put_by_key`."""
        return self._cache.generate_key(text, src_lang, tgt_lang, engine, options)

    def get(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        engine: str,
        options: TranslationOptions | None = None,
    ) -> str | None:
        """Get cached translation if available and not expired."""
        return self._cache.get(text, src_lang, tgt_lang, engine, options)

    def get_by_key(self, cache_key: tuple) -> str | None:
        """Get cached translation using a pre-computed key."""
        return self._cache.get_by_key(cache_key)

    def put(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        engine: str,
        translation: str,
        options: TranslationOptions | None = None,
    ) -> None:
        """Cache a translation result."""
        self._cache.put(text, src_lang, tgt_lang, engine, translation, options)

    def put_by_key(self, cache_key: tuple, translation: str) -> None:
        """Cache a translation result using a pre-computed key."""
        self._cache.put_by_key(cache_key, translation)

    def clear(self) -> None:
        """Clear all cached translations."""
        self._cache.clear()
        self._logger.info("Translation cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def save_to_disk(self, file_path: Path | str) -> bool:
        """Persist cache to a gzip-compressed JSON file."""
        return self._cache.save_to_disk(file_path)

    def load_from_disk(self, file_path: Path | str) -> int:
        """Load cache entries from a gzip-compressed JSON file."""
        return self._cache.load_from_disk(file_path)
