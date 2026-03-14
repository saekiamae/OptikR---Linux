"""
NLLB-200 Multilingual Translation Worker

Translation plugin for NLLB-200 (200+ languages).
Uses Meta AI's No Language Left Behind model for high-quality translation.
Language pair is determined at runtime from pipeline config.
"""

import logging
import time
from typing import Any

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine, TranslationOptions, TranslationResult,
    BatchTranslationResult,
)

logger = logging.getLogger("optikr.pipeline.nllb200")

# NLLB-200 uses BCP-47-like language codes, not ISO 639-1.
_NLLB_LANG_MAP: dict[str, str] = {
    "ja": "jpn_Jpan", "en": "eng_Latn", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ko": "kor_Hang", "ar": "arb_Arab", "nl": "nld_Latn",
    "pl": "pol_Latn", "tr": "tur_Latn",
}


def _to_nllb(lang: str) -> str:
    return _NLLB_LANG_MAP.get(lang, lang)


class TranslationEngine(AbstractTranslationEngine):
    """NLLB-200 translation engine plugin."""

    def __init__(self):
        super().__init__("nllb200")
        self._logger = logger
        self.model = None
        self.tokenizer = None
        self.model_name = "facebook/nllb-200-1.3B"
        self._device = None
        self._max_length = 512

    def initialize(self, config: dict) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            self._logger.error("transformers library not available")
            return False
        try:
            self.model_name = config.get("model_name", self.model_name)
            use_gpu = config.get("gpu", True) or config.get("use_gpu", True)
            self._device = (
                torch.device("cuda")
                if use_gpu and torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._logger.info("Loading NLLB model: %s on %s", self.model_name, self._device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, use_safetensors=True,
                ).to(self._device)
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, use_safetensors=False,
                ).to(self._device)
            self._logger.info("NLLB model loaded on %s", self._device)
            return True
        except Exception as e:
            self._logger.error("Failed to initialize NLLB: %s", e)
            return False

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def cleanup(self) -> None:
        self.model = None
        self.tokenizer = None
        from app.utils.pytorch_manager import release_gpu_memory
        release_gpu_memory()
        self._logger.info("NLLB engine cleaned up")

    def _make_fallback(self, text, src, tgt):
        return TranslationResult(
            original_text=text, translated_text=text,
            source_language=src, target_language=tgt,
            confidence=0.0, engine_used=self.engine_name,
            processing_time_ms=0.0, from_cache=False,
        )

    def translate_text(self, text: str, src_lang: str, tgt_lang: str,
                       options: TranslationOptions | None = None) -> TranslationResult:
        if not self.is_available():
            return self._make_fallback(text, src_lang, tgt_lang)
        start = time.time()
        try:
            src_nllb = _to_nllb(src_lang)
            tgt_nllb = _to_nllb(tgt_lang)
            self.tokenizer.src_lang = src_nllb
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self._max_length,
            ).to(self._device)
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, forced_bos_token_id=tgt_token_id,
                    max_new_tokens=256,
                )
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = (time.time() - start) * 1000
            return TranslationResult(
                original_text=text, translated_text=translated,
                source_language=src_lang, target_language=tgt_lang,
                confidence=0.90, engine_used=self.engine_name,
                processing_time_ms=elapsed, from_cache=False,
            )
        except Exception as e:
            self._logger.error("NLLB translation failed: %s", e)
            return self._make_fallback(text, src_lang, tgt_lang)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str,
                        options: TranslationOptions | None = None) -> BatchTranslationResult:
        start = time.time()
        results = []
        failed = []
        try:
            src_nllb = _to_nllb(src_lang)
            tgt_nllb = _to_nllb(tgt_lang)
            self.tokenizer.src_lang = src_nllb
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self._max_length,
            ).to(self._device)
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, forced_bos_token_id=tgt_token_id,
                    max_new_tokens=256,
                )
            for i, (orig, out) in enumerate(zip(texts, outputs)):
                try:
                    translated = self.tokenizer.decode(out, skip_special_tokens=True)
                    results.append(TranslationResult(
                        original_text=orig, translated_text=translated,
                        source_language=src_lang, target_language=tgt_lang,
                        confidence=0.90, engine_used=self.engine_name,
                        processing_time_ms=0.0, from_cache=False,
                    ))
                except Exception as e:
                    failed.append((i, str(e)))
                    results.append(self._make_fallback(orig, src_lang, tgt_lang))
        except Exception as e:
            self._logger.error("NLLB batch translation failed: %s", e)
            for i, t in enumerate(texts):
                failed.append((i, str(e)))
                results.append(self._make_fallback(t, src_lang, tgt_lang))
        elapsed = (time.time() - start) * 1000
        return BatchTranslationResult(
            results=results, total_processing_time_ms=elapsed,
            cache_hit_rate=0.0, failed_translations=failed,
        )

    def get_supported_languages(self) -> list[str]:
        return list(_NLLB_LANG_MAP.keys())

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        return src_lang in _NLLB_LANG_MAP and tgt_lang in _NLLB_LANG_MAP
