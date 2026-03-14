"""
Translation Cache Optimizer Plugin
Caches translations for instant lookup of repeated text.

Works as a pre/post plugin on the translation stage:
- Pre (process): looks up each text_block in the in-memory cache and marks
  cache-hit blocks with ``skip_translation=True`` so TranslationStage skips
  them.
- Post (post_process): stores new translations in the cache for future frames.
"""

import logging
import time
import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TranslationCacheOptimizer:
    """In-memory LRU translation cache with TTL expiration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get('max_cache_size', 10000)
        self.ttl = config.get('ttl_seconds', 3600)
        self.fuzzy_match = config.get('enable_fuzzy_match', False)

        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------

    def _make_key(self, text: str, source_lang: str, target_lang: str) -> str:
        key_str = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return (time.time() - self.timestamps[key]) > self.ttl

    def _evict_old_entries(self) -> None:
        current_time = time.time()
        keys_to_remove = [
            k for k, ts in self.timestamps.items()
            if current_time - ts > self.ttl
        ]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            self.evictions += 1

    def _evict_lru(self) -> None:
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            self.timestamps.pop(key, None)
            self.evictions += 1

    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Look up a cached translation."""
        key = self._make_key(text, source_lang, target_lang)

        if self._is_expired(key):
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            self.misses += 1
            return None

        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, text: str, source_lang: str, target_lang: str, translation: str) -> None:
        """Store a translation in the cache."""
        key = self._make_key(text, source_lang, target_lang)

        if len(self.cache) % 100 == 0:
            self._evict_old_entries()

        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = translation
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _block_text(block: Any) -> str:
        """Extract the source text from a text block (dict or object)."""
        if isinstance(block, dict):
            return block.get("text", str(block))
        return getattr(block, "text", str(block))

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-translation: look up each text block in the cache.

        Blocks whose text is found are marked with ``skip_translation=True``
        and ``translated_text=<cached>`` so that ``TranslationStage`` skips
        them without calling the translation engine.
        """
        text_blocks = data.get("text_blocks", [])
        if not text_blocks:
            return data

        source_lang = data.get("source_lang", "auto")
        target_lang = data.get("target_lang", "en")

        for block in text_blocks:
            text = self._block_text(block)
            if not text:
                continue

            cached = self.get(text, source_lang, target_lang)
            if cached is not None and cached != text:
                if isinstance(block, dict):
                    block["skip_translation"] = True
                    block["translated_text"] = cached
                else:
                    block.skip_translation = True
                    block.translated_text = cached

        return data

    def post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-translation: store new translations in the cache.

        Iterates over the output translations and caches any that are not
        already from the dictionary or cache.  Re-caching existing entries
        is a harmless no-op (overwrites with same value and refreshes TTL).
        """
        translations = data.get("translations", [])
        text_blocks = data.get("text_blocks", [])
        source_lang = data.get("source_lang", data.get("source_language", "auto"))
        target_lang = data.get("target_lang", data.get("target_language", "en"))

        for i, trans in enumerate(translations):
            if getattr(trans, "from_dictionary", False):
                continue

            translated_text = (
                getattr(trans, "translated_text", None) or str(trans)
            ) if not isinstance(trans, str) else trans

            if not translated_text:
                continue

            if i < len(text_blocks):
                source_text = self._block_text(text_blocks[i])
            else:
                source_text = ""

            if source_text and translated_text and translated_text != source_text:
                self.put(source_text, source_lang, target_lang, translated_text)

        return data

    # ------------------------------------------------------------------
    # Stats / lifecycle
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.cache),
            'evictions': self.evictions,
        }

    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0


# Plugin interface
def initialize(config: Dict[str, Any]) -> TranslationCacheOptimizer:
    """Initialize the optimizer plugin."""
    return TranslationCacheOptimizer(config)
