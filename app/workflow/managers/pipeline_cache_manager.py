"""
Pipeline Cache Manager

Manages translation caching (memory + persistent dictionary) and
dictionary persistence.

Uses the unified ``LRUCache`` from ``app.utils.cache`` as the backing store.
"""

import hashlib
import logging
import threading
from typing import Any

from app.utils.cache import LRUCache


class PipelineCacheManager:
    """
    Manages caching for the pipeline.

    Features:
    - Translation caching (memory + persistent dictionary)
    - Smart cache invalidation
    """

    def __init__(self, enable_persistent_dictionary: bool = True, config_manager=None):
        """Initialize cache manager."""
        self.logger = logging.getLogger(__name__)

        _get = (lambda key, default: config_manager.get_setting(key, default)) if config_manager else (lambda _k, d: d)
        trans_cache_size = _get('cache.translation_cache_size', 10000)
        trans_cache_memory = _get('cache.translation_cache_memory_mb', 10.0)

        self.translation_cache: LRUCache[str, str] = LRUCache(max_size=trans_cache_size, max_memory_mb=trans_cache_memory)

        self.persistent_dictionary = None
        if enable_persistent_dictionary:
            try:
                from app.text_translation.smart_dictionary import SmartDictionary
                self.persistent_dictionary = SmartDictionary()
                self.logger.info("Persistent translation dictionary enabled")
            except Exception as e:
                self.logger.warning(f"Failed to load persistent dictionary: {e}")

        self.lock = threading.RLock()

        # When True, cleanup() will NOT auto-save dictionaries to disk.
        # This allows the save dialog to control persistence so the user
        # can choose to discard learned translations.
        self.defer_dictionary_save = False

        self.logger.info("Pipeline Cache Manager initialized")

    # ── Translation operations ────────────────────────────────────

    def cache_translation(self, text: str, source_lang: str, target_lang: str, translation: str,
                         confidence: float = 0.9, save_to_dictionary: bool = True):
        """
        Cache translation in memory AND save to persistent dictionary.

        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            translation: Translated text
            confidence: Translation confidence score
            save_to_dictionary: Whether to save to persistent dictionary
        """
        key = self._make_translation_key(text, source_lang, target_lang)
        self.translation_cache.put(key, translation)

        if save_to_dictionary and self.persistent_dictionary:
            try:
                self.persistent_dictionary.add_entry(
                    source_text=text,
                    translation=translation,
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence=confidence,
                    source_engine="cached"
                )
            except Exception as e:
                self.logger.error(f"Failed to save translation to dictionary: {e}")

    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """
        Get cached translation from memory cache OR persistent dictionary.

        Two-tier lookup: memory LRU first, then persistent SmartDictionary.
        Dictionary hits are promoted to memory cache for subsequent lookups.
        """
        key = self._make_translation_key(text, source_lang, target_lang)
        cached = self.translation_cache.get(key)
        if cached:
            return cached

        if self.persistent_dictionary:
            try:
                entry = self.persistent_dictionary.lookup(text, source_lang, target_lang)
                if entry:
                    self.translation_cache.put(key, entry.translation)
                    return entry.translation
            except Exception as e:
                self.logger.debug(f"Dictionary lookup failed: {e}")

        return None

    @staticmethod
    def _make_translation_key(text: str, source_lang: str, target_lang: str) -> str:
        """Create cache key for translation."""
        normalized = text.strip().lower()
        return f"{source_lang}:{target_lang}:{hashlib.md5(normalized.encode()).hexdigest()}"

    # ── Cache management ──────────────────────────────────────────

    def clear_all(self, clear_dictionary: bool = False):
        """
        Clear all caches.

        Args:
            clear_dictionary: If True, also clear the persistent dictionary
        """
        self.translation_cache.clear()

        if clear_dictionary and self.persistent_dictionary:
            try:
                self.persistent_dictionary.clear_all_entries()
                self.logger.info("Cleared persistent dictionary")
            except AttributeError:
                self.logger.warning("Persistent dictionary does not support clearing")
            except Exception as e:
                self.logger.error(f"Failed to clear dictionary: {e}")

        self.logger.info("All caches cleared")

    # ── Dictionary persistence ────────────────────────────────────

    def save_all_dictionaries(self) -> None:
        """Save all loaded language-pair dictionaries to disk.

        Called on pipeline stop to ensure no in-memory translations are lost.
        """
        if not self.persistent_dictionary:
            return

        pairs = self.persistent_dictionary.get_available_language_pairs()
        if not pairs:
            return

        for source_lang, target_lang, _file_path, entry_count in pairs:
            if entry_count > 0:
                try:
                    self.save_dictionary(source_lang, target_lang)
                except Exception as e:
                    self.logger.error(
                        f"Failed to save dictionary {source_lang}→{target_lang}: {e}"
                    )

    def save_dictionary(self, source_lang: str, target_lang: str):
        """Save persistent dictionary to disk for a language pair."""
        if not self.persistent_dictionary:
            self.logger.warning("Persistent dictionary not enabled")
            return

        try:
            from app.utils.path_utils import get_dictionary_dir

            dict_path = self.persistent_dictionary.get_loaded_dictionary_path(source_lang, target_lang)
            if not dict_path:
                dict_dir = get_dictionary_dir()
                dict_dir.mkdir(parents=True, exist_ok=True)
                dict_path = str(dict_dir / f"{source_lang}_{target_lang}.json.gz")

            self.persistent_dictionary.save_dictionary(dict_path, source_lang, target_lang)
            self.logger.info(f"Saved dictionary: {source_lang}→{target_lang} to {dict_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dictionary: {e}")

    def get_dictionary_stats(self, source_lang: str, target_lang: str) -> dict[str, Any]:
        """Get statistics for persistent dictionary."""
        if not self.persistent_dictionary:
            return {'enabled': False}

        try:
            stats = self.persistent_dictionary.get_stats(source_lang, target_lang)
            return {
                'enabled': True,
                'total_entries': stats.total_entries,
                'total_usage': stats.total_usage,
                'average_usage': stats.average_usage,
                'total_lookups': stats.total_lookups,
                'cache_hits': stats.cache_hits
            }
        except Exception as e:
            self.logger.error(f"Failed to get dictionary stats: {e}")
            return {'enabled': True, 'error': str(e)}

    def cleanup(self) -> None:
        """Save persistent dictionaries and release resources.

        Safe to call multiple times.  When ``defer_dictionary_save`` is
        True, dictionary persistence is skipped so the save dialog can
        control whether learned translations are kept or discarded.
        """
        with self.lock:
            if not self.defer_dictionary_save:
                try:
                    self.save_all_dictionaries()
                except Exception as e:
                    self.logger.warning("Error saving dictionaries during cleanup: %s", e)
            else:
                self.logger.info(
                    "Dictionary save deferred — waiting for user decision"
                )

            try:
                self.translation_cache.clear()
            except Exception as e:
                self.logger.warning("Error clearing translation cache during cleanup: %s", e)

        self.logger.info("PipelineCacheManager cleaned up")
