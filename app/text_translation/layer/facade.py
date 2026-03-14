"""
Translation Facade.

Exposes the same public API as the original TranslationLayer,
delegating to EngineManager, TranslationCacheAdapter,
LanguageDetectorService, and DictionaryOps.

Requirements: 3.2
"""
import time
import logging
import threading
from typing import Any

from app.interfaces import ITranslationLayer
from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine,
    TranslationOptions,
    LanguageDetectionResult,
)
from app.text_translation.layer.engine_manager import EngineManager
from app.text_translation.layer.cache_adapter import TranslationCacheAdapter
from app.text_translation.layer.language_detector import LanguageDetectorService
from app.text_translation.layer.dictionary_ops import DictionaryOps


class TranslationFacade(ITranslationLayer):
    """
    Facade that delegates to focused sub-modules while preserving
    the original TranslationLayer public API.
    """

    def __init__(
        self,
        cache_size: int | None = None,
        cache_ttl: int | None = None,
        config_manager: Any | None = None,
    ) -> None:
        self._logger = logging.getLogger("optikr.pipeline.translation")
        self._config_manager = config_manager
        self._lock = threading.RLock()

        # Resolve cache settings from config
        if cache_size is None and config_manager:
            cache_size = config_manager.get_setting(
                "cache.translation_cache_size", 10000
            )
        elif cache_size is None:
            cache_size = 10000

        if cache_ttl is None and config_manager:
            cache_ttl = config_manager.get_setting(
                "cache.translation_cache_ttl", 3600
            )
        elif cache_ttl is None:
            cache_ttl = 3600

        # Sub-modules
        self._engine_mgr = EngineManager(config_manager=config_manager)
        self._cache_adapter = TranslationCacheAdapter(
            max_size=cache_size, ttl_seconds=cache_ttl
        )
        self._lang_detector = LanguageDetectorService(config_manager=config_manager)
        self._dict_ops = DictionaryOps(
            engine_registry=self._engine_mgr.engine_registry,
            config_manager=config_manager,
        )

        # Performance tracking
        self._performance_stats = {
            "total_translations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_translations": 0,
            "avg_translation_time_ms": 0.0,
        }

        # Log-throttle: avoid flooding console with identical errors
        self._last_error_msg: str = ""
        self._last_error_time: float = 0.0
        self._suppressed_error_count: int = 0
        self._ERROR_LOG_COOLDOWN: float = 10.0

        # Expose plugin_manager for backward compat
        self.plugin_manager = self._engine_mgr.plugin_manager

        self._logger.info("Translation layer initialized with plugin support")

    # -- Expose internal registries for backward compat -----------------------

    @property
    def _engine_registry(self) -> Any:
        return self._engine_mgr.engine_registry

    @property
    def _cache(self) -> Any:
        return self._cache_adapter._cache

    @property
    def _language_detector(self) -> Any:
        return self._lang_detector

    @property
    def _default_engine(self) -> str | None:
        return self._engine_mgr.default_engine

    @_default_engine.setter
    def _default_engine(self, value: str | None) -> None:
        self._engine_mgr.set_default_engine_unchecked(value)

    @property
    def _fallback_engines(self) -> list[str]:
        return self._engine_mgr.fallback_engines

    # -- Engine management (delegates to EngineManager) -----------------------

    def register_engine(
        self,
        engine: AbstractTranslationEngine,
        is_default: bool = False,
        is_fallback: bool = True,
    ) -> bool:
        return self._engine_mgr.register_engine(engine, is_default, is_fallback)

    def load_engine(
        self, engine_name: str, config: dict[str, Any] | None = None
    ) -> bool:
        return self._engine_mgr.load_engine(engine_name, config)

    def unload_engine(self, engine_name: str) -> bool:
        return self._engine_mgr.unload_engine(engine_name)

    def set_default_engine(self, engine_name: str) -> bool:
        return self._engine_mgr.set_default_engine(engine_name)

    def get_available_engines(self) -> list[str]:
        return self._engine_mgr.get_available_engines()

    def get_engines_for_language_pair(
        self, src_lang: str, tgt_lang: str
    ) -> list[str]:
        return self._engine_mgr.get_engines_for_language_pair(src_lang, tgt_lang)

    def preload_models(self, src_lang: str, tgt_lang: str) -> bool:
        return self._engine_mgr.preload_models(src_lang, tgt_lang)

    # -- Translation (ITranslationLayer) ----------------------------------------

    def translate(
        self,
        text: str,
        engine: str,
        src_lang: str,
        tgt_lang: str,
        options: dict[str, Any],
    ) -> str:
        if not text or not text.strip():
            return text

        start_time = time.time()

        try:
            translation_options = self._lang_detector.parse_translation_options(options)

            if src_lang == "auto":
                detection_result = self._lang_detector.detect_language(text)
                src_lang = detection_result.language_code
                self._logger.debug(
                    f"Auto-detected language: {src_lang} "
                    f"(confidence: {detection_result.confidence})"
                )

            # Pre-compute cache key once for the initial engine
            initial_cache_key = None
            if translation_options.use_cache:
                initial_cache_key = self._cache_adapter.generate_key(
                    text, src_lang, tgt_lang, engine, translation_options
                )
                cached = self._cache_adapter.get_by_key(initial_cache_key)
                if cached and cached != text:
                    self._update_performance_stats(True, time.time() - start_time)
                    return cached

            # Try dictionary engine first
            dict_engine = self._engine_mgr.get_engine("dictionary")
            if dict_engine and dict_engine.is_available():
                try:
                    dict_result = dict_engine.translate_text(
                        text, src_lang, tgt_lang, translation_options
                    )
                    if dict_result.confidence > 0:
                        self._logger.debug(
                            f"Using dictionary translation: {text} -> "
                            f"{dict_result.translated_text}"
                        )
                        if translation_options.use_cache:
                            self._cache_adapter.put(
                                text,
                                src_lang,
                                tgt_lang,
                                "dictionary",
                                dict_result.translated_text,
                                translation_options,
                            )
                        self._update_performance_stats(
                            False, time.time() - start_time
                        )
                        return dict_result.translated_text
                except Exception as e:
                    self._logger.debug(
                        f"Dictionary lookup failed, falling back to {engine}: {e}"
                    )

            # Get translation engine
            engine_name = engine
            translation_engine = self._engine_mgr.get_engine(engine_name)

            # Config may store "marianmt_gpu"/"marianmt_cpu" but engine
            # registers under its base name "marianmt".  Try stripped variant.
            if not translation_engine or not translation_engine.is_available():
                for suffix in ("_gpu", "_cpu"):
                    if engine_name.endswith(suffix):
                        base_name = engine_name[: -len(suffix)]
                        candidate = self._engine_mgr.get_engine(base_name)
                        if candidate and candidate.is_available():
                            translation_engine = candidate
                            engine_name = base_name
                            break

            # Fall back to the default engine if the requested one isn't loaded
            if not translation_engine or not translation_engine.is_available():
                default = self._engine_mgr.default_engine
                if default and default != engine_name:
                    candidate = self._engine_mgr.get_engine(default)
                    if candidate and candidate.is_available():
                        self._logger.info(
                            "Engine '%s' not available, using default '%s'",
                            engine_name, default,
                        )
                        translation_engine = candidate
                        engine_name = default

            if not translation_engine or not translation_engine.is_available():
                translation_engine = self._engine_mgr.get_fallback_engine(
                    src_lang, tgt_lang
                )
                if not translation_engine:
                    raise RuntimeError(
                        f"No available translation engine for {src_lang} -> {tgt_lang}"
                    )
                engine_name = translation_engine.engine_name

            result = translation_engine.translate_text(
                text, src_lang, tgt_lang, translation_options
            )

            if translation_options.use_cache and result.translated_text and result.translated_text != text:
                if engine_name == engine and initial_cache_key is not None:
                    self._cache_adapter.put_by_key(
                        initial_cache_key, result.translated_text
                    )
                else:
                    self._cache_adapter.put(
                        text,
                        src_lang,
                        tgt_lang,
                        engine_name,
                        result.translated_text,
                        translation_options,
                    )

            self._update_performance_stats(False, time.time() - start_time)
            return result.translated_text

        except Exception as e:
            self._performance_stats["failed_translations"] += 1
            err_str = str(e)
            now = time.time()
            if err_str != self._last_error_msg or (now - self._last_error_time) >= self._ERROR_LOG_COOLDOWN:
                if self._suppressed_error_count > 0:
                    self._logger.error(
                        f"Translation failed: {e} "
                        f"(repeated {self._suppressed_error_count} more time(s) since last log)"
                    )
                else:
                    self._logger.error(f"Translation failed: {e}")
                self._last_error_msg = err_str
                self._last_error_time = now
                self._suppressed_error_count = 0
            else:
                self._suppressed_error_count += 1
            return text

    def translate_batch(
        self,
        texts: list[str],
        engine: str,
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        if not texts:
            return []

        start_time = time.time()

        try:
            if src_lang == "auto" and texts:
                detection_result = self._lang_detector.detect_language(texts[0])
                src_lang = detection_result.language_code
                self._logger.debug(
                    f"Auto-detected language for batch: {src_lang}"
                )

            engine_name = engine
            translation_engine = self._engine_mgr.get_engine(engine_name)

            if not translation_engine or not translation_engine.is_available():
                for suffix in ("_gpu", "_cpu"):
                    if engine_name.endswith(suffix):
                        base_name = engine_name[: -len(suffix)]
                        candidate = self._engine_mgr.get_engine(base_name)
                        if candidate and candidate.is_available():
                            translation_engine = candidate
                            engine_name = base_name
                            break

            if not translation_engine or not translation_engine.is_available():
                default = self._engine_mgr.default_engine
                if default and default != engine_name:
                    candidate = self._engine_mgr.get_engine(default)
                    if candidate and candidate.is_available():
                        translation_engine = candidate
                        engine_name = default

            if not translation_engine or not translation_engine.is_available():
                translation_engine = self._engine_mgr.get_fallback_engine(
                    src_lang, tgt_lang
                )
                if not translation_engine:
                    raise RuntimeError(
                        f"No available translation engine for {src_lang} -> {tgt_lang}"
                    )

            cached_results = {}
            texts_to_translate = []
            indices_to_translate = []
            miss_cache_keys: dict[int, tuple] = {}

            for i, text in enumerate(texts):
                if not text or not text.strip():
                    cached_results[i] = text
                    continue

                key = self._cache_adapter.generate_key(
                    text, src_lang, tgt_lang, translation_engine.engine_name
                )
                cached = self._cache_adapter.get_by_key(key)
                if cached and cached != text:
                    cached_results[i] = cached
                else:
                    miss_cache_keys[i] = key
                    texts_to_translate.append(text)
                    indices_to_translate.append(i)

            translated_results = []
            if texts_to_translate:
                options = self._parse_translation_options({})
                batch_result = translation_engine.translate_batch(
                    texts_to_translate, src_lang, tgt_lang, options=options
                )
                translated_results = batch_result.results

                for idx, result in zip(indices_to_translate, translated_results):
                    original_text = texts[idx] if idx < len(texts) else ""
                    if result.translated_text and result.translated_text != original_text:
                        self._cache_adapter.put_by_key(
                            miss_cache_keys[idx], result.translated_text
                        )

            final_results = [""] * len(texts)
            for i, translation in cached_results.items():
                final_results[i] = translation
            for i, result in enumerate(translated_results):
                original_index = indices_to_translate[i]
                final_results[original_index] = result.translated_text

            cache_hits = len(cached_results)
            total_requests = len(texts)
            for _ in range(cache_hits):
                self._update_performance_stats(True, 0)
            for _ in range(total_requests - cache_hits):
                self._update_performance_stats(
                    False,
                    (time.time() - start_time) / max(total_requests - cache_hits, 1),
                )

            return final_results

        except Exception as e:
            self._logger.error(f"Batch translation failed: {e}")
            self._performance_stats["failed_translations"] += len(texts)
            return texts

    def get_supported_languages(self, engine: str) -> list[str]:
        try:
            translation_engine = self._engine_mgr.get_engine(engine)
            if translation_engine:
                return translation_engine.get_supported_languages()
            return []
        except Exception as e:
            self._logger.error(f"Failed to get supported languages: {e}")
            return []

    def cache_translation(
        self,
        source: str,
        target: str,
        translation: str,
        source_lang: str | None = None,
    ) -> None:
        try:
            if source_lang is None:
                source_lang = getattr(self, "_current_source_lang", None) or "auto"
            engine_name = self._engine_mgr.default_engine or "unknown"
            self._cache_adapter.put(source, source_lang, target, engine_name, translation)
        except Exception as e:
            self._logger.error(f"Failed to cache translation: {e}")

    def clear_cache(self) -> None:
        try:
            self._cache_adapter.clear()
        except Exception as e:
            self._logger.error(f"Failed to clear cache: {e}")

    # -- Language detection -----------------------------------------------------

    def detect_language(self, text: str) -> LanguageDetectionResult:
        return self._lang_detector.detect_language(text)

    # -- Dictionary operations (delegates to DictionaryOps) -------------------

    def get_available_language_pairs(self) -> list[Any]:
        return self._dict_ops.get_available_language_pairs()

    def get_current_language_pair(self) -> Any:
        return self._dict_ops.get_current_language_pair()

    def set_language_pair(self, source_lang: str, target_lang: str) -> None:
        self._dict_ops.set_language_pair(source_lang, target_lang)
        # Keep instance attrs for backward compat
        self._current_source_lang = source_lang
        self._current_target_lang = target_lang

    def get_dictionary_stats(self) -> dict[str, Any]:
        return self._dict_ops.get_dictionary_stats()

    def clear_dictionary(self) -> None:
        self._dict_ops.clear_dictionary()

    def reload_dictionary_from_file(
        self,
        file_path: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> None:
        self._dict_ops.reload_dictionary_from_file(
            file_path, source_lang, target_lang
        )

    def get_loaded_dictionary_path(
        self,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str | None:
        return self._dict_ops.get_loaded_dictionary_path(
            source_lang, target_lang
        )

    def export_dictionary_wordbook(
        self,
        output_path: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str | None:
        return self._dict_ops.export_dictionary_wordbook(
            output_path, source_lang, target_lang
        )

    # -- Performance stats ----------------------------------------------------

    def get_performance_stats(self) -> dict[str, Any]:
        cache_stats = self._cache_adapter.get_stats()
        return {
            "translation_stats": self._performance_stats.copy(),
            "cache_stats": cache_stats,
            "available_engines": self.get_available_engines(),
            "default_engine": self._engine_mgr.default_engine,
            "fallback_engines": self._engine_mgr.fallback_engines,
        }

    def get_benchmark_metadata(self) -> dict[str, Any]:
        """
        Lightweight snapshot of translation-layer settings for benchmark reports.

        Returns:
            Dict with engine/cache/performance metadata suitable for embedding in
            benchmark JSON headers. Kept intentionally small and JSON‑serializable.
        """
        stats = self.get_performance_stats()

        cache_cfg: dict[str, Any] = {}
        try:
            cache_obj = getattr(self._cache_adapter, "_cache", None)
            if cache_obj is not None:
                cache_cfg = {
                    "max_size": getattr(cache_obj, "max_size", None),
                    "ttl_seconds": getattr(cache_obj, "ttl_seconds", None),
                }
        except Exception:
            cache_cfg = {}

        engines_summary: list[dict[str, Any]] = []
        try:
            available = self.get_available_engines()
            for name in available:
                eng = self._engine_mgr.engine_registry.get_engine(name)
                if not eng:
                    continue
                info = {}
                try:
                    info = eng.get_engine_info()
                except Exception:
                    info = {"name": name}
                try:
                    perf = eng.get_performance_stats()
                except Exception:
                    perf = {}
                engines_summary.append(
                    {
                        "name": name,
                        "info": info,
                        "performance": perf,
                    }
                )
        except Exception:
            engines_summary = []

        config_snapshot: dict[str, Any] = {}
        if self._config_manager is not None:
            try:
                config_snapshot = {
                    "engine": self._config_manager.get_setting(
                        "translation.engine", None
                    ),
                    "source_language": self._config_manager.get_setting(
                        "translation.source_language", None
                    ),
                    "target_language": self._config_manager.get_setting(
                        "translation.target_language", None
                    ),
                    "fallback_enabled": self._config_manager.get_setting(
                        "translation.fallback_enabled", True
                    ),
                    "cache_size": self._config_manager.get_setting(
                        "cache.translation_cache_size", None
                    ),
                    "cache_ttl": self._config_manager.get_setting(
                        "cache.translation_cache_ttl", None
                    ),
                }
            except Exception:
                config_snapshot = {}

        return {
            "translation_stats": stats.get("translation_stats", {}),
            "cache_stats": stats.get("cache_stats", {}),
            "cache_config": cache_cfg,
            "engines": engines_summary,
            "config": config_snapshot,
        }

    # -- Internal helpers (kept private, same logic as original) --------------

    def _parse_translation_options(self, options: dict[str, Any]) -> TranslationOptions:
        return self._lang_detector.parse_translation_options(options)

    def _get_fallback_engine(
        self, src_lang: str, tgt_lang: str
    ) -> AbstractTranslationEngine | None:
        return self._engine_mgr.get_fallback_engine(src_lang, tgt_lang)

    def _update_performance_stats(self, cache_hit: bool, duration: float) -> None:
        with self._lock:
            self._performance_stats["total_translations"] += 1
            if cache_hit:
                self._performance_stats["cache_hits"] += 1
            else:
                self._performance_stats["cache_misses"] += 1
            duration_ms = duration * 1000
            current_avg = self._performance_stats["avg_translation_time_ms"]
            alpha = 0.1
            self._performance_stats["avg_translation_time_ms"] = (
                alpha * duration_ms + (1 - alpha) * current_avg
            )

    # -- Cache persistence -----------------------------------------------------

    def save_cache_to_disk(self, file_path: str | None = None) -> bool:
        """Persist the translation cache to disk.

        Args:
            file_path: Target path.  When *None*, uses the default location
                       ``system_data/cache/translation_cache.json.gz``.

        Returns:
            True on success, False on error.
        """
        if file_path is None:
            from app.utils.path_utils import ensure_dir
            file_path = str(ensure_dir("cache") / "translation_cache.json.gz")
        return self._cache_adapter.save_to_disk(file_path)

    def load_cache_from_disk(self, file_path: str | None = None) -> int:
        """Load cached translations from disk into memory.

        Args:
            file_path: Source path.  When *None*, uses the default location
                       ``system_data/cache/translation_cache.json.gz``.

        Returns:
            Number of entries loaded.
        """
        if file_path is None:
            from app.utils.path_utils import get_cache_dir
            file_path = str(get_cache_dir() / "translation_cache.json.gz")
        return self._cache_adapter.load_from_disk(file_path)

    # -- Cleanup --------------------------------------------------------------

    def cleanup(self) -> None:
        try:
            self._engine_mgr.cleanup()
            self._cache_adapter.clear()
            self._logger.info("Translation layer cleaned up")
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")


def create_translation_layer(config: dict[str, Any] | None = None) -> TranslationFacade:
    """
    Factory function to create and configure translation layer.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured TranslationFacade instance
    """
    if config is None:
        config = {}

    cache_size = config.get("cache_size", 10000)
    cache_ttl = config.get("cache_ttl", 3600)

    return TranslationFacade(cache_size=cache_size, cache_ttl=cache_ttl)
