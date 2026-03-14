"""
Translation Engine Interface and Base Classes

This module provides the abstract translation engine interface and base implementations
for translation functionality including batch processing, caching, and language detection.
"""

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
from dataclasses import dataclass, field
import gzip
import json
import time
import threading
from collections import defaultdict, OrderedDict
from pathlib import Path
import logging

from app.utils.error_handling import try_import

_models = try_import("app.models", "models")
Translation = _models.Translation
TextBlock = _models.TextBlock


class TranslationQuality(Enum):
    """Translation quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BEST = "best"


class LanguageDetectionConfidence(Enum):
    """Language detection confidence levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TranslationOptions:
    """Options for translation requests."""
    quality: TranslationQuality = TranslationQuality.MEDIUM
    preserve_formatting: bool = True
    use_cache: bool = True
    timeout_seconds: float = 5.0
    context: str | None = None
    domain: str | None = None  # e.g., "technical", "medical", "general"
    
    
@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language_code: str
    confidence: float
    confidence_level: LanguageDetectionConfidence
    alternative_languages: list[tuple[str, float]] = field(default_factory=list)
    

@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    engine_used: str
    processing_time_ms: float
    from_cache: bool = False
    alternatives: list[str] = field(default_factory=list)


@dataclass
class BatchTranslationResult:
    """Result of batch translation operation."""
    results: list[TranslationResult]
    total_processing_time_ms: float
    cache_hit_rate: float
    failed_translations: list[tuple[int, str]] = field(default_factory=list)  # (index, error)


class TranslationCache:
    """High-performance translation cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize translation cache.
        
        Args:
            max_size: Maximum number of cached translations
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[tuple, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def _generate_cache_key(self, text: str, src_lang: str, tgt_lang: str,
                            engine: str, options: TranslationOptions | None = None) -> tuple:
        """Generate cache key as a native tuple (hashable, no serialization cost)."""
        if options:
            return (text, src_lang, tgt_lang, engine, options.quality.value, options.domain or '')
        return (text, src_lang, tgt_lang, engine)

    def generate_key(self, text: str, src_lang: str, tgt_lang: str,
                     engine: str, options: TranslationOptions | None = None) -> tuple:
        """Public wrapper around key generation for callers that need to
        compute a key once and pass it to :meth:`get_by_key` / :meth:`put_by_key`."""
        return self._generate_cache_key(text, src_lang, tgt_lang, engine, options)

    # -- key-based internals ---------------------------------------------------

    def _get_by_key(self, cache_key: tuple) -> str | None:
        with self._lock:
            if cache_key in self._cache:
                translation, timestamp = self._cache[cache_key]
                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[cache_key]
                    self._stats['misses'] += 1
                    self._stats['size'] = len(self._cache)
                    return None
                self._cache.move_to_end(cache_key)
                self._stats['hits'] += 1
                return translation
            self._stats['misses'] += 1
            return None

    def _put_by_key(self, cache_key: tuple, translation: str) -> None:
        with self._lock:
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats['evictions'] += 1
            self._cache[cache_key] = (translation, time.time())
            self._stats['size'] = len(self._cache)

    # -- public convenience API (compute key + lookup in one call) -------------

    def get(self, text: str, src_lang: str, tgt_lang: str, engine: str,
            options: TranslationOptions | None = None) -> str | None:
        """Get cached translation if available and not expired."""
        return self._get_by_key(
            self._generate_cache_key(text, src_lang, tgt_lang, engine, options)
        )

    def get_by_key(self, cache_key: tuple) -> str | None:
        """Get cached translation using a pre-computed key."""
        return self._get_by_key(cache_key)

    def put(self, text: str, src_lang: str, tgt_lang: str, engine: str,
            translation: str, options: TranslationOptions | None = None) -> None:
        """Cache translation result."""
        self._put_by_key(
            self._generate_cache_key(text, src_lang, tgt_lang, engine, options),
            translation,
        )

    def put_by_key(self, cache_key: tuple, translation: str) -> None:
        """Cache translation result using a pre-computed key."""
        self._put_by_key(cache_key, translation)
    
    def clear(self) -> None:
        """Clear all cached translations."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'size': 0
            }
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': self._stats['size'],
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }

    def save_to_disk(self, file_path: Path | str) -> bool:
        """Serialize cache entries to a gzip-compressed JSON file.

        Only non-expired entries are persisted.  The file is written
        atomically (write to a temporary file, then rename) to avoid
        corruption on crash.

        Returns True on success, False on error (logged, never raises).
        """
        file_path = Path(file_path)
        logger = logging.getLogger(__name__)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            now = time.time()
            with self._lock:
                entries: list[dict] = []
                for key, (translation, timestamp) in self._cache.items():
                    if self.ttl_seconds and (now - timestamp) > self.ttl_seconds:
                        continue
                    entries.append({
                        "k": list(key),
                        "v": translation,
                        "t": timestamp,
                    })
                payload = {
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl_seconds,
                    "entries": entries,
                }

            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            tmp_path.replace(file_path)
            logger.info(
                "Translation cache saved to disk: %d entries -> %s",
                len(entries), file_path,
            )
            return True
        except Exception as exc:
            logger.error("Failed to save translation cache to disk: %s", exc)
            return False

    def load_from_disk(self, file_path: Path | str) -> int:
        """Deserialize cache entries from a gzip-compressed JSON file.

        Expired entries (based on the current TTL) are skipped.
        Entries are merged into the live cache; existing in-memory
        entries take priority over on-disk ones.

        Returns the number of entries loaded.
        """
        file_path = Path(file_path)
        logger = logging.getLogger(__name__)
        if not file_path.exists():
            logger.debug("No translation cache file found at %s", file_path)
            return 0
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)

            entries = payload.get("entries", [])
            now = time.time()
            loaded = 0
            with self._lock:
                for entry in entries:
                    key = tuple(entry["k"])
                    translation = entry["v"]
                    timestamp = entry["t"]
                    if self.ttl_seconds and (now - timestamp) > self.ttl_seconds:
                        continue
                    if key in self._cache:
                        continue
                    # Evict LRU if at capacity
                    while len(self._cache) >= self.max_size:
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                        self._stats['evictions'] += 1
                    self._cache[key] = (translation, timestamp)
                    loaded += 1
                self._stats['size'] = len(self._cache)

            logger.info(
                "Translation cache loaded from disk: %d entries from %s "
                "(%d skipped as expired or duplicate)",
                loaded, file_path, len(entries) - loaded,
            )
            return loaded
        except Exception as exc:
            logger.error("Failed to load translation cache from disk: %s", exc)
            return 0




class AbstractTranslationEngine(ABC):
    """Abstract base class for translation engines."""
    
    def __init__(self, engine_name: str):
        """
        Initialize translation engine.
        
        Args:
            engine_name: Name identifier for this engine
        """
        self.engine_name = engine_name
        self._logger = logging.getLogger(f"{__name__}.{engine_name}")
        self._is_initialized = False
        self._supported_languages: set[str] = set()
        self._performance_stats = defaultdict(list)
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> bool:
        """
        Initialize the translation engine with configuration.
        
        Args:
            config: Engine-specific configuration
            
        Returns:
            True if initialization successful, False otherwise
        """
        raise NotImplementedError("Translation engine plugins must implement initialize()")
    
    @abstractmethod
    def translate_text(self, text: str, src_lang: str, tgt_lang: str,
                      options: TranslationOptions | None = None) -> TranslationResult:
        """
        Translate single text.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            options: Translation options
            
        Returns:
            TranslationResult with translation and metadata
        """
        raise NotImplementedError("Translation engine plugins must implement translate_text()")
    
    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str,
                       options: TranslationOptions | None = None) -> BatchTranslationResult:
        """
        Translate multiple texts in batch.
        
        Default implementation translates individually. Engines should override
        for true batch processing optimization.
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            options: Translation options
            
        Returns:
            BatchTranslationResult with all translations and metadata
        """
        start_time = time.time()
        results = []
        failed_translations = []
        
        for i, text in enumerate(texts):
            try:
                result = self.translate_text(text, src_lang, tgt_lang, options)
                results.append(result)
            except Exception as e:
                self._logger.error(f"Failed to translate text at index {i}: {e}")
                failed_translations.append((i, str(e)))
        
        total_time_ms = (time.time() - start_time) * 1000
        cache_hits = sum(1 for r in results if r.from_cache)
        cache_hit_rate = cache_hits / len(results) if results else 0.0
        
        return BatchTranslationResult(
            results=results,
            total_processing_time_ms=total_time_ms,
            cache_hit_rate=cache_hit_rate,
            failed_translations=failed_translations
        )
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(self._supported_languages)
    
    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        """Check if engine supports specific language pair."""
        return src_lang in self._supported_languages and tgt_lang in self._supported_languages
    
    def is_available(self) -> bool:
        """Check if engine is available and initialized."""
        return self._is_initialized
    
    def get_engine_info(self) -> dict[str, Any]:
        """Get engine information and capabilities."""
        return {
            'name': self.engine_name,
            'initialized': self._is_initialized,
            'supported_languages': list(self._supported_languages),
            'supports_batch': True,
            'supports_quality_levels': True,
            'supports_domains': False
        }
    
    def record_performance(self, operation: str, duration_ms: float) -> None:
        """Record performance metrics for monitoring."""
        self._performance_stats[operation].append(duration_ms)
        
        # Keep only last 100 measurements per operation
        if len(self._performance_stats[operation]) > 100:
            self._performance_stats[operation] = self._performance_stats[operation][-100:]
    
    def get_performance_stats(self) -> dict[str, dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for operation, times in self._performance_stats.items():
            if times:
                stats[operation] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'count': len(times)
                }
        return stats
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up engine resources."""
        raise NotImplementedError("Translation engine plugins must implement cleanup()")


class TranslationEngineRegistry:
    """Registry for managing translation engines."""
    
    def __init__(self):
        """Initialize engine registry."""
        self._engines: dict[str, AbstractTranslationEngine] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_engine(self, engine: AbstractTranslationEngine) -> bool:
        """
        Register a translation engine.
        
        Args:
            engine: Translation engine instance
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if engine.engine_name in self._engines:
                self._logger.warning(f"Engine {engine.engine_name} already registered, replacing")
            
            self._engines[engine.engine_name] = engine
            self._logger.info(f"Registered translation engine: {engine.engine_name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to register engine {engine.engine_name}: {e}")
            return False
    
    def unregister_engine(self, engine_name: str) -> bool:
        """
        Unregister a translation engine.
        
        Args:
            engine_name: Name of engine to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if engine_name in self._engines:
                engine = self._engines[engine_name]
                engine.cleanup()
                del self._engines[engine_name]
                self._logger.info(f"Unregistered translation engine: {engine_name}")
                return True
            else:
                self._logger.warning(f"Engine {engine_name} not found for unregistration")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to unregister engine {engine_name}: {e}")
            return False
    
    def get_engine(self, engine_name: str) -> AbstractTranslationEngine | None:
        """Get engine by name."""
        return self._engines.get(engine_name)
    
    def get_available_engines(self) -> list[str]:
        """Get list of available engine names."""
        return [name for name, engine in self._engines.items() if engine.is_available()]
    
    def get_engines_for_language_pair(self, src_lang: str, tgt_lang: str) -> list[str]:
        """Get engines that support specific language pair."""
        compatible_engines = []
        for name, engine in self._engines.items():
            if engine.is_available() and engine.supports_language_pair(src_lang, tgt_lang):
                compatible_engines.append(name)
        return compatible_engines
    
    def cleanup_all(self) -> None:
        """Clean up all registered engines."""
        for engine in self._engines.values():
            try:
                engine.cleanup()
            except Exception as e:
                self._logger.error(f"Error cleaning up engine {engine.engine_name}: {e}")
        self._engines.clear()