"""
MarianMT Translation Engine Plugin

Neural machine translation using MarianMT (Helsinki-NLP models).
This is the plugin wrapper that implements the AbstractTranslationEngine interface.

This is the CANONICAL MarianMT implementation — all other MarianMT code
(subprocess wrappers, app/ engine) should be removed in favor of this.
"""

import logging
import threading
import time

try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    MarianMTModel = None
    MarianTokenizer = None

from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine, TranslationOptions, TranslationResult,
    BatchTranslationResult
)


# Maximum number of cached models to prevent unbounded memory growth.
# Each model is ~300MB. With a limit of 3, worst case is ~900MB.
_MAX_CACHED_MODELS = 3


class TranslationEngine(AbstractTranslationEngine):
    """
    MarianMT translation engine plugin.

    This is the main class that the plugin manager will instantiate.
    Must be named 'TranslationEngine' for plugin discovery.
    """

    def __init__(self):
        """Initialize MarianMT engine."""
        super().__init__("marianmt")
        self._logger = logging.getLogger("optikr.pipeline.marianmt")
        self._is_initialized = TRANSFORMERS_AVAILABLE
        self._loaded_models: dict[tuple[str, str], tuple] = {}
        self._model_load_order: list[tuple[str, str]] = []  # LRU tracking
        self._model_lock = threading.Lock()
        self._device = None  # Set during initialize()
        self._max_length = 512
        self._num_beams = 4

        if not TRANSFORMERS_AVAILABLE:
            self._logger.warning("transformers library not available")
        else:
            self._logger.info("MarianMT engine plugin ready (call initialize() to configure device)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, config: dict) -> bool:
        """
        Initialize engine with configuration.

        Args:
            config: Configuration dictionary with optional keys:
                   - source_language: Source language code
                   - target_language: Target language code
                   - gpu: Use GPU acceleration if available
                   - runtime_mode: 'auto', 'gpu', or 'cpu'
                   - max_length: Maximum token length (default 512)
                   - num_beams: Beam search width (default 4)

        Returns:
            True if initialization successful
        """
        if not TRANSFORMERS_AVAILABLE:
            self._logger.error("transformers library not available")
            return False

        try:
            import torch

            self._config = config
            self._max_length = int(config.get('max_length', 512))
            self._num_beams = int(config.get('num_beams', 4))

            # Determine device
            use_gpu = config.get('gpu', True)
            runtime_mode = config.get('runtime_mode', 'auto')

            if runtime_mode == 'cpu':
                use_gpu = False
            elif runtime_mode == 'gpu':
                use_gpu = True

            if use_gpu and torch.cuda.is_available():
                self._device = torch.device('cuda')
                self._logger.info("MarianMT using GPU acceleration (CUDA)")
            else:
                self._device = torch.device('cpu')
                if use_gpu and not torch.cuda.is_available():
                    self._logger.warning("GPU requested but CUDA not available, falling back to CPU")
                else:
                    self._logger.info("MarianMT using CPU")

            # PyTorch 2.6+ safe-globals workaround (call once)
            try:
                torch.serialization.add_safe_globals([])
            except AttributeError:
                pass  # Older PyTorch versions don't have this

            self._is_initialized = True
            self._logger.info(f"MarianMT engine initialized on {self._device}")
            return True

        except Exception as e:
            self._logger.exception(f"Failed to initialize: {e}")
            return False

    def is_available(self) -> bool:
        """Check if engine is available and initialized."""
        return self._is_initialized and TRANSFORMERS_AVAILABLE

    def cleanup(self) -> None:
        """Clean up engine resources."""
        self.unload_all_models()
        self._is_initialized = False
        self._logger.info("MarianMT engine cleaned up")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _get_model_name(self, src_lang: str, tgt_lang: str) -> str:
        """Get HuggingFace model name for language pair."""
        src = src_lang.lower()
        tgt = tgt_lang.lower()
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    def _load_model(self, src_lang: str, tgt_lang: str) -> tuple | None:
        """
        Load MarianMT model for language pair (thread-safe, LRU-cached).

        Args:
            src_lang: Source language
            tgt_lang: Target language

        Returns:
            Tuple of (model, tokenizer) or None if loading fails
        """
        lang_pair = (src_lang, tgt_lang)

        with self._model_lock:
            if lang_pair in self._loaded_models:
                # Move to end of LRU list
                if lang_pair in self._model_load_order:
                    self._model_load_order.remove(lang_pair)
                self._model_load_order.append(lang_pair)
                return self._loaded_models[lang_pair]

        # Load outside the lock to avoid blocking other threads during download
        try:
            model_name = self._get_model_name(src_lang, tgt_lang)
            self._logger.info(f"Loading MarianMT model: {model_name} on {self._device}")

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            try:
                model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
            except Exception:
                model = MarianMTModel.from_pretrained(model_name, use_safetensors=False)

            # Move to configured device (defaults to CPU if initialize() wasn't called)
            import torch
            device = self._device or torch.device('cpu')
            model = model.to(device)

            with self._model_lock:
                # Double-check after acquiring lock
                if lang_pair not in self._loaded_models:
                    # Evict oldest model if at capacity
                    evicted_any = False
                    while len(self._loaded_models) >= _MAX_CACHED_MODELS and self._model_load_order:
                        evict_pair = self._model_load_order.pop(0)
                        evicted = self._loaded_models.pop(evict_pair, None)
                        if evicted:
                            evicted_any = True
                            self._logger.info(f"Evicted model: {evict_pair[0]}->{evict_pair[1]}")
                    if evicted_any:
                        from app.utils.pytorch_manager import release_gpu_memory
                        release_gpu_memory()

                    self._loaded_models[lang_pair] = (model, tokenizer)
                    self._model_load_order.append(lang_pair)

                self._logger.info(f"Model loaded: {model_name} on {self._device}")
                return self._loaded_models[lang_pair]

        except Exception as e:
            self._logger.exception(f"Failed to load model for {src_lang}->{tgt_lang}: {e}")
            return None

    def preload_model(self, src_lang: str, tgt_lang: str) -> bool:
        """
        Pre-load a model so the first translation is fast.

        Call from the main thread before starting the pipeline.

        Args:
            src_lang: Source language
            tgt_lang: Target language

        Returns:
            True if model loaded successfully
        """
        return self._load_model(src_lang, tgt_lang) is not None

    def unload_model(self, src_lang: str, tgt_lang: str) -> None:
        """Unload a specific model to free memory."""
        lang_pair = (src_lang, tgt_lang)
        with self._model_lock:
            if lang_pair in self._loaded_models:
                del self._loaded_models[lang_pair]
                if lang_pair in self._model_load_order:
                    self._model_load_order.remove(lang_pair)
                self._logger.info(f"Unloaded model: {src_lang}->{tgt_lang}")

        from app.utils.pytorch_manager import release_gpu_memory
        release_gpu_memory()

    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        with self._model_lock:
            self._loaded_models.clear()
            self._model_load_order.clear()
            self._logger.info("Unloaded all MarianMT models")

        from app.utils.pytorch_manager import release_gpu_memory
        release_gpu_memory()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _make_fallback_result(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
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

    def translate_text(self, text: str, src_lang: str, tgt_lang: str,
                       options: TranslationOptions | None = None) -> TranslationResult:
        """
        Translate text using MarianMT.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            options: Translation options (optional)

        Returns:
            TranslationResult with translated text
        """
        if not self.is_available():
            self._logger.warning("Engine not available, returning original text")
            return self._make_fallback_result(text, src_lang, tgt_lang)

        start_time = time.time()

        try:
            model_tuple = self._load_model(src_lang, tgt_lang)
            if model_tuple is None:
                self._logger.warning(f"Model loading failed for {src_lang}->{tgt_lang}")
                return self._make_fallback_result(text, src_lang, tgt_lang)

            model, tokenizer = model_tuple

            # Tokenize and move to device
            import torch
            device = self._device or torch.device('cpu')
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=self._max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            translated = model.generate(
                **inputs,
                max_length=self._max_length,
                num_beams=self._num_beams,
                early_stopping=True,
            )

            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            processing_time = (time.time() - start_time) * 1000

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=src_lang,
                target_language=tgt_lang,
                confidence=0.90,
                engine_used=self.engine_name,
                processing_time_ms=processing_time,
                from_cache=False,
            )

        except Exception as e:
            self._logger.error(f"Translation failed: {e}")
            return self._make_fallback_result(text, src_lang, tgt_lang)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str,
                        options: TranslationOptions | None = None) -> BatchTranslationResult:
        """
        Translate multiple texts in a single batch.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            options: Translation options (optional)

        Returns:
            BatchTranslationResult with all translations
        """
        start_time = time.time()
        results: list[TranslationResult] = []
        failed: list = []

        try:
            model_tuple = self._load_model(src_lang, tgt_lang)

            if model_tuple is None:
                for i, text in enumerate(texts):
                    failed.append((i, "Model loading failed"))
                    results.append(self._make_fallback_result(text, src_lang, tgt_lang))
            else:
                model, tokenizer = model_tuple

                inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self._max_length)
                import torch
                device = self._device or torch.device('cpu')
                inputs = {k: v.to(device) for k, v in inputs.items()}

                translated = model.generate(
                    **inputs,
                    max_length=self._max_length,
                    num_beams=self._num_beams,
                    early_stopping=True,
                )

                for i, (original, translation) in enumerate(zip(texts, translated)):
                    try:
                        translated_text = tokenizer.decode(translation, skip_special_tokens=True)
                        results.append(TranslationResult(
                            original_text=original,
                            translated_text=translated_text,
                            source_language=src_lang,
                            target_language=tgt_lang,
                            confidence=0.90,
                            engine_used=self.engine_name,
                            processing_time_ms=0.0,
                            from_cache=False,
                        ))
                    except Exception as e:
                        failed.append((i, str(e)))
                        results.append(self._make_fallback_result(original, src_lang, tgt_lang))

        except Exception as e:
            self._logger.error(f"Batch translation failed: {e}")
            for i, text in enumerate(texts):
                failed.append((i, str(e)))
                results.append(self._make_fallback_result(text, src_lang, tgt_lang))

        total_time = (time.time() - start_time) * 1000

        return BatchTranslationResult(
            results=results,
            total_processing_time_ms=total_time,
            cache_hit_rate=0.0,
            failed_translations=failed,
        )

    # ------------------------------------------------------------------
    # Language support
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'nl', 'pl', 'tr', 'sv', 'no', 'da', 'fi', 'cs', 'el',
            'he', 'id', 'ro', 'uk', 'bg', 'hr', 'sr', 'sk', 'sl', 'et',
            'lv', 'lt', 'ca', 'gl', 'eu', 'is', 'mk', 'sq', 'af', 'sw',
        ]

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        """Check if engine supports specific language pair."""
        supported = self.get_supported_languages()
        return src_lang.lower() in supported and tgt_lang.lower() in supported
