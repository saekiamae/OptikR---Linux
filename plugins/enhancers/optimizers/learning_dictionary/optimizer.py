"""
Smart Dictionary Optimizer Plugin
Persistent learned translations for instant lookup using SmartDictionary.

Works as both a pre-plugin and post-plugin for the translation stage:
- Pre (process): Looks up text_blocks in the dictionary, removes blocks
  with known translations so the translation engine can skip them.
- Post (post_process): Merges dictionary translations back into the
  final translations list and learns new translations from engine output.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from app.text_translation.translation_quality_filter import (
        default_quality_filter,
        strict_quality_filter,
    )
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    default_quality_filter = None  # type: ignore[assignment]
    strict_quality_filter = None   # type: ignore[assignment]
    logger.info("[LEARNING_DICT] Translation quality filter not available")


class _DictionaryTranslation:
    """Lightweight translation result from dictionary lookup.

    Mirrors the attributes of ``TranslationResult`` so the overlay system
    and downstream consumers can handle it identically.
    """

    __slots__ = (
        'original_text', 'translated_text', 'confidence',
        'source_language', 'target_language', 'from_dictionary',
        'position', 'engine_used', 'from_cache', 'processing_time_ms',
        'alternatives',
    )

    def __init__(
        self,
        original_text: str,
        translated_text: str,
        confidence: float,
        source_language: str,
        target_language: str,
        position: Any = None,
    ) -> None:
        self.original_text = original_text
        self.translated_text = translated_text
        self.confidence = confidence
        self.source_language = source_language
        self.target_language = target_language
        self.from_dictionary = True
        self.position = position
        self.engine_used = 'dictionary'
        self.from_cache = True
        self.processing_time_ms = 0.0
        self.alternatives: list[str] = []

    def __str__(self) -> str:
        return self.translated_text


class SmartDictionaryOptimizer:
    """Provides instant lookup for learned translations using SmartDictionary.

    The dictionary engine is injected after construction via
    ``set_dictionary_engine()`` so it can share the application-wide
    ``SmartDictionary`` instance.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.auto_save: bool = config.get('auto_save', True)
        self.min_confidence: float = config.get('min_confidence', 0.8)
        self.validate_sentences: bool = config.get('validate_sentences', True)
        self._quality_filter_mode: int = 0
        self._learn_words: bool = True
        self._learn_sentences: bool = True

        self._dictionary_engine: Any = None
        self._context_manager: Any = None
        self._source_lang: str = config.get('source_lang', 'ja')
        self._target_lang: str = config.get('target_lang', 'en')

        self._pending_pre_translated: list[_DictionaryTranslation] = []

        self.total_lookups = 0
        self.cache_hits = 0
        self.context_hits = 0
        self.cache_misses = 0
        self.saved_translations = 0
        self.rejected_translations = 0

        logger.info(
            "[LEARNING_DICT] Initialized (auto_save=%s, min_confidence=%s, validate=%s)",
            'on' if self.auto_save else 'off',
            self.min_confidence,
            self.validate_sentences,
        )

    def set_dictionary_engine(self, engine: Any) -> None:
        """Inject the SmartDictionary instance used for lookups and learning."""
        self._dictionary_engine = engine

    def set_context_manager(self, context_manager: Any) -> None:
        """Inject the Context Manager plugin for locked-term priority lookup.

        When set, ``process()`` checks ``context_manager.lookup_term()``
        *before* the dictionary, enforcing the priority order:
        Context Manager > Smart Dictionary > Translation engine.
        """
        self._context_manager = context_manager

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """Set the default source/target languages for dictionary lookups."""
        self._source_lang = source_lang
        self._target_lang = target_lang

    def set_quality_filter_config(
        self, enabled: bool, mode: int = 0,
    ) -> None:
        """Apply app-level quality-filter settings.

        Parameters
        ----------
        enabled:
            Maps to ``translation.quality_filter_enabled`` in the schema.
            When *False*, ``_should_save`` skips quality filtering entirely.
        mode:
            Maps to ``translation.quality_filter_mode`` (0 = balanced,
            1 = strict).
        """
        self.validate_sentences = enabled
        self._quality_filter_mode = mode

    def set_learn_config(
        self, learn_words: bool = True, learn_sentences: bool = True,
    ) -> None:
        """Apply dictionary learning scope from ``dictionary.*`` config keys."""
        self._learn_words = learn_words
        self._learn_sentences = learn_sentences

    # ------------------------------------------------------------------
    # Pre-plugin: dictionary lookup on text_blocks
    # ------------------------------------------------------------------

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process: look up OCR text blocks in the dictionary.

        Blocks whose text is found in the dictionary are removed from
        ``data['text_blocks']`` (so the translation engine skips them)
        and stored internally.  The companion ``SmartDictionaryPostProcessor``
        merges them back into the final ``translations`` list.

        Priority order: Context Manager locked terms > Smart Dictionary >
        Translation engine.
        """
        text_blocks = data.get('text_blocks', [])
        dict_engine = self._dictionary_engine
        ctx_mgr = self._context_manager

        if not text_blocks or (not dict_engine and not ctx_mgr):
            self._pending_pre_translated = []
            return data

        source_lang = self._source_lang
        target_lang = self._target_lang

        pre_translated: list[_DictionaryTranslation] = []

        for block in text_blocks:
            source_text = getattr(block, 'text', str(block))
            self.total_lookups += 1

            # --- Priority 1: Context Manager locked terms ---
            if ctx_mgr is not None:
                try:
                    locked = ctx_mgr.lookup_term(
                        source_text, source_lang, target_lang,
                    )
                    if locked:
                        if isinstance(block, dict):
                            block['skip_translation'] = True
                            block['translated_text'] = locked
                        else:
                            block.skip_translation = True
                            block.translated_text = locked
                        pre_translated.append(_DictionaryTranslation(
                            original_text=source_text,
                            translated_text=locked,
                            confidence=1.0,
                            source_language=source_lang,
                            target_language=target_lang,
                            position=getattr(block, 'position', None),
                        ))
                        self.context_hits += 1
                        self.cache_hits += 1
                        continue
                except Exception as exc:
                    logger.debug("[LEARNING_DICT] Context lookup error: %s", exc)

            # --- Priority 2: Smart Dictionary ---
            if dict_engine is not None:
                try:
                    entry = dict_engine.lookup(source_text, source_lang, target_lang)
                    if entry and entry.translation and entry.translation != source_text:
                        if isinstance(block, dict):
                            block['skip_translation'] = True
                            block['translated_text'] = entry.translation
                        else:
                            block.skip_translation = True
                            block.translated_text = entry.translation
                        pre_translated.append(_DictionaryTranslation(
                            original_text=source_text,
                            translated_text=entry.translation,
                            confidence=entry.get_effective_confidence(),
                            source_language=source_lang,
                            target_language=target_lang,
                            position=getattr(block, 'position', None),
                        ))
                        self.cache_hits += 1
                        try:
                            entry.update_usage(success=True)
                        except Exception:
                            pass
                        continue
                except Exception as exc:
                    logger.debug("[LEARNING_DICT] Lookup error: %s", exc)

            # --- Priority 3: pass through to translation engine ---
            self.cache_misses += 1

        self._pending_pre_translated = pre_translated

        if pre_translated:
            logger.debug(
                "[LEARNING_DICT] %d/%d blocks resolved (context=%d, dict=%d)",
                len(pre_translated),
                len(text_blocks),
                self.context_hits,
                self.cache_hits - self.context_hits,
            )

        return data

    # ------------------------------------------------------------------
    # Post-plugin: merge dict results and learn new translations
    # ------------------------------------------------------------------

    def post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Post-process: merge dictionary translations and learn new ones.

        Called by ``SmartDictionaryPostProcessor.process()`` after the
        translation engine has run.
        """
        translations = list(data.get('translations', []))
        text_blocks = data.get('text_blocks', [])
        dict_engine = self._dictionary_engine

        # 1. Merge pre-translated dictionary results back in
        if self._pending_pre_translated:
            translations.extend(self._pending_pre_translated)
            self._pending_pre_translated = []

        # 2. Learn new translations from the engine output
        if dict_engine and self.auto_save:
            for i, tr in enumerate(translations):
                if getattr(tr, 'from_dictionary', False):
                    continue

                original = getattr(tr, 'original_text', '')
                translated = getattr(tr, 'translated_text', '')
                confidence = getattr(tr, 'confidence', 0.0)

                # TranslationStage stores plain strings — pair with text_block
                if not original and isinstance(tr, str) and tr:
                    translated = tr
                    if i < len(text_blocks):
                        block = text_blocks[i]
                        original = (
                            block.get('text', str(block))
                            if isinstance(block, dict)
                            else getattr(block, 'text', str(block))
                        )
                    if not confidence:
                        confidence = 0.9

                if not original or not translated:
                    continue
                if original == translated:
                    continue

                if self._should_save(original, translated, confidence):
                    try:
                        dict_engine.learn_from_translation(
                            source_text=original,
                            translation=translated,
                            source_language=self._source_lang,
                            target_language=self._target_lang,
                            confidence=confidence,
                        )
                        self.saved_translations += 1
                    except Exception as exc:
                        logger.debug("[LEARNING_DICT] Save error: %s", exc)
                else:
                    self.rejected_translations += 1

        data['translations'] = translations
        return data

    # ------------------------------------------------------------------

    def _should_save(self, source: str, translated: str, confidence: float) -> bool:
        """Validate a translation before saving to the dictionary."""
        word_count = len(source.split())
        is_single_word = word_count <= 1
        is_sentence = word_count >= 2

        if is_single_word and not self._learn_words:
            return False
        if is_sentence and not self._learn_sentences:
            return False

        if self.validate_sentences and VALIDATOR_AVAILABLE:
            qf = (
                strict_quality_filter
                if self._quality_filter_mode == 1
                else default_quality_filter
            )
            is_valid, _reason = qf.should_save(
                source, translated, confidence,
                self._source_lang, self._target_lang,
            )
            return is_valid

        return bool(translated) and confidence >= self.min_confidence

    def create_post_processor(self) -> 'SmartDictionaryPostProcessor':
        """Create the companion post-plugin that delegates to ``post_process()``."""
        return SmartDictionaryPostProcessor(self)

    # ------------------------------------------------------------------
    # Statistics / lifecycle
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return optimizer statistics."""
        hit_rate = (self.cache_hits / self.total_lookups * 100) if self.total_lookups > 0 else 0
        total_save_attempts = self.saved_translations + self.rejected_translations
        save_rate = (self.saved_translations / total_save_attempts * 100) if total_save_attempts > 0 else 0

        return {
            'total_lookups': self.total_lookups,
            'cache_hits': self.cache_hits,
            'context_hits': self.context_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'saved_translations': self.saved_translations,
            'rejected_translations': self.rejected_translations,
            'save_rate': f"{save_rate:.1f}%",
            'auto_save': self.auto_save,
            'validation_enabled': self.validate_sentences,
        }

    def reset(self) -> None:
        """Reset optimizer statistics."""
        self.total_lookups = 0
        self.cache_hits = 0
        self.context_hits = 0
        self.cache_misses = 0
        self.saved_translations = 0
        self.rejected_translations = 0

    def cleanup(self) -> None:
        """Persist learned translations to disk before shutdown."""
        if not self._dictionary_engine:
            return
        try:
            pairs = self._dictionary_engine.get_available_language_pairs()
            for source_lang, target_lang, file_path, entry_count in pairs:
                if file_path and entry_count > 0:
                    self._dictionary_engine.save_dictionary(
                        file_path, source_lang, target_lang,
                    )
            logger.info("[LEARNING_DICT] Dictionary saved on cleanup")
        except Exception as exc:
            logger.warning("[LEARNING_DICT] Failed to save dictionary on cleanup: %s", exc)


class SmartDictionaryPostProcessor:
    """Post-plugin wrapper: delegates to ``SmartDictionaryOptimizer.post_process()``.

    ``PluginAwareStage`` calls ``.process()`` on every plugin in its list.
    This thin wrapper adapts that call to the optimizer's ``post_process()``
    method, keeping the pre/post logic cleanly separated.
    """

    def __init__(self, optimizer: SmartDictionaryOptimizer) -> None:
        self._optimizer = optimizer

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._optimizer.post_process(data)

    def cleanup(self) -> None:
        pass

    def reset(self) -> None:
        pass


# Plugin interface
def initialize(config: dict[str, Any]) -> SmartDictionaryOptimizer:
    """Initialize the optimizer plugin."""
    return SmartDictionaryOptimizer(config)
