"""Built-in metadata catalog for all supported model families.

Defines BUILTIN_MODELS — a dictionary keyed by model_id containing
ModelMetadata for 8 families (4 translation, 4 OCR).  Translation entries
are per-language-pair for MarianMT and per-variant for multilingual families.
OCR entries are per-engine.

This module is pure data with no I/O; it is imported by ModelCatalog at
startup to seed the in-memory catalog.
"""

from app.core.model_catalog_types import ModelMetadata

# ---------------------------------------------------------------------------
# MarianMT language pairs  (Helsinki-NLP/opus-mt-*)
# ---------------------------------------------------------------------------
_MARIANMT_PAIRS: dict[str, dict] = {
    "en-de": {"repo": "Helsinki-NLP/opus-mt-en-de", "size": 298, "bleu": 41.2},
    "de-en": {"repo": "Helsinki-NLP/opus-mt-de-en", "size": 301, "bleu": 43.1},
    "en-es": {"repo": "Helsinki-NLP/opus-mt-en-es", "size": 301, "bleu": 42.8},
    "es-en": {"repo": "Helsinki-NLP/opus-mt-es-en", "size": 298, "bleu": 44.3},
    "en-fr": {"repo": "Helsinki-NLP/opus-mt-en-fr", "size": 290, "bleu": 40.5},
    "fr-en": {"repo": "Helsinki-NLP/opus-mt-fr-en", "size": 287, "bleu": 42.7},
    "en-it": {"repo": "Helsinki-NLP/opus-mt-en-it", "size": 285, "bleu": 38.9},
    "it-en": {"repo": "Helsinki-NLP/opus-mt-it-en", "size": 283, "bleu": 41.2},
    "en-pt": {"repo": "Helsinki-NLP/opus-mt-en-pt", "size": 295, "bleu": 39.7},
    "pt-en": {"repo": "Helsinki-NLP/opus-mt-pt-en", "size": 292, "bleu": 42.1},
    "en-ja": {"repo": "Helsinki-NLP/opus-mt-en-jap", "size": 312, "bleu": 28.5},
    "ja-en": {"repo": "Helsinki-NLP/opus-mt-jap-en", "size": 315, "bleu": 31.2},
    "en-zh": {"repo": "Helsinki-NLP/opus-mt-en-zh", "size": 325, "bleu": 26.8},
    "zh-en": {"repo": "Helsinki-NLP/opus-mt-zh-en", "size": 328, "bleu": 29.4},
    "en-ko": {"repo": "Helsinki-NLP/opus-mt-en-ko", "size": 308, "bleu": 24.3},
    "ko-en": {"repo": "Helsinki-NLP/opus-mt-ko-en", "size": 310, "bleu": 27.8},
    "en-ru": {"repo": "Helsinki-NLP/opus-mt-en-ru", "size": 318, "bleu": 32.1},
    "ru-en": {"repo": "Helsinki-NLP/opus-mt-ru-en", "size": 320, "bleu": 35.6},
    "en-ar": {"repo": "Helsinki-NLP/opus-mt-en-ar", "size": 295, "bleu": 27.9},
    "ar-en": {"repo": "Helsinki-NLP/opus-mt-ar-en", "size": 298, "bleu": 30.2},
    "en-nl": {"repo": "Helsinki-NLP/opus-mt-en-nl", "size": 288, "bleu": 39.4},
    "nl-en": {"repo": "Helsinki-NLP/opus-mt-nl-en", "size": 285, "bleu": 41.8},
    "en-pl": {"repo": "Helsinki-NLP/opus-mt-en-pl", "size": 292, "bleu": 33.7},
    "pl-en": {"repo": "Helsinki-NLP/opus-mt-pl-en", "size": 290, "bleu": 36.2},
    "en-tr": {"repo": "Helsinki-NLP/opus-mt-en-tr", "size": 285, "bleu": 31.5},
    "tr-en": {"repo": "Helsinki-NLP/opus-mt-tr-en", "size": 283, "bleu": 34.1},
}

_WEIGHTS_FILES = [
    "config.json",
    "pytorch_model.bin",
]

_SAFETENSORS_OR_WEIGHTS = [
    "config.json",
]


def _marianmt_entries() -> dict[str, ModelMetadata]:
    """Generate one ModelMetadata per MarianMT language pair."""
    entries: dict[str, ModelMetadata] = {}
    for pair, info in _MARIANMT_PAIRS.items():
        src, tgt = pair.split("-")
        model_id = f"marianmt-{pair}"
        bleu = info["bleu"]
        quality = "high" if bleu >= 40 else ("good" if bleu >= 30 else "basic")
        entries[model_id] = ModelMetadata(
            family="MarianMT",
            category="translation",
            languages=[src, tgt],
            size_mb=info["size"],
            speed="fast",
            quality=quality,
            gpu_required=False,
            rationale=f"MarianMT {src.upper()}\u2192{tgt.upper()} — BLEU {bleu:.1f}, ~{info['size']}MB, runs on CPU",
            hf_repo=info["repo"],
            required_files=list(_WEIGHTS_FILES),
        )
    return entries


# ---------------------------------------------------------------------------
# NLLB-200 variants  (facebook/nllb-200-*)
# ---------------------------------------------------------------------------
_NLLB_VARIANTS: dict[str, dict] = {
    "distilled-600M": {
        "repo": "facebook/nllb-200-distilled-600M",
        "size": 600,
        "langs": 200,
        "bleu": 38.5,
    },
    "1.3B": {
        "repo": "facebook/nllb-200-1.3B",
        "size": 1300,
        "langs": 200,
        "bleu": 42.1,
    },
    "3.3B": {
        "repo": "facebook/nllb-200-3.3B",
        "size": 3300,
        "langs": 200,
        "bleu": 44.8,
    },
}

_NLLB_COMMON_LANGS = [
    "en", "de", "es", "fr", "it", "pt", "ja", "zh", "ko", "ru",
    "ar", "nl", "pl", "tr", "hi", "th", "vi", "id", "uk", "cs",
]


def _nllb_entries() -> dict[str, ModelMetadata]:
    entries: dict[str, ModelMetadata] = {}
    for variant, info in _NLLB_VARIANTS.items():
        model_id = f"nllb200-{variant}"
        bleu = info["bleu"]
        quality = "high" if bleu >= 42 else "good"
        speed = "fast" if info["size"] <= 700 else ("medium" if info["size"] <= 1500 else "slow")
        entries[model_id] = ModelMetadata(
            family="NLLB-200",
            category="translation",
            languages=list(_NLLB_COMMON_LANGS),
            size_mb=info["size"],
            speed=speed,
            quality=quality,
            gpu_required=info["size"] > 1500,
            rationale=f"NLLB-200 {variant} — BLEU {bleu:.1f}, {info['langs']} languages, ~{info['size']}MB",
            hf_repo=info["repo"],
            required_files=list(_SAFETENSORS_OR_WEIGHTS),
        )
    return entries


# ---------------------------------------------------------------------------
# M2M-100 variants  (facebook/m2m100_*)
# ---------------------------------------------------------------------------
_M2M100_VARIANTS: dict[str, dict] = {
    "418M": {
        "repo": "facebook/m2m100_418M",
        "size": 418,
        "langs": 100,
        "bleu": 35.2,
    },
    "1.2B": {
        "repo": "facebook/m2m100_1.2B",
        "size": 1200,
        "langs": 100,
        "bleu": 39.7,
    },
}


def _m2m100_entries() -> dict[str, ModelMetadata]:
    entries: dict[str, ModelMetadata] = {}
    for variant, info in _M2M100_VARIANTS.items():
        model_id = f"m2m100-{variant}"
        bleu = info["bleu"]
        quality = "good" if bleu >= 35 else "basic"
        speed = "fast" if info["size"] <= 500 else "medium"
        entries[model_id] = ModelMetadata(
            family="M2M-100",
            category="translation",
            languages=list(_NLLB_COMMON_LANGS),
            size_mb=info["size"],
            speed=speed,
            quality=quality,
            gpu_required=info["size"] > 1000,
            rationale=f"M2M-100 {variant} — BLEU {bleu:.1f}, {info['langs']} languages, ~{info['size']}MB",
            hf_repo=info["repo"],
            required_files=list(_SAFETENSORS_OR_WEIGHTS),
        )
    return entries


# ---------------------------------------------------------------------------
# mBART  (facebook/mbart-large-50-many-to-many-mmt)
# ---------------------------------------------------------------------------

def _mbart_entries() -> dict[str, ModelMetadata]:
    return {
        "mbart-large-50": ModelMetadata(
            family="mBART",
            category="translation",
            languages=list(_NLLB_COMMON_LANGS),
            size_mb=2400,
            speed="slow",
            quality="high",
            gpu_required=True,
            rationale="mBART large-50 — BLEU 41.3, 50 languages, 2.4GB, GPU recommended",
            hf_repo="facebook/mbart-large-50-many-to-many-mmt",
            required_files=list(_SAFETENSORS_OR_WEIGHTS),
        ),
    }


# ---------------------------------------------------------------------------
# Qwen3 variants  (Qwen/Qwen3-*)
# ---------------------------------------------------------------------------
_QWEN3_VARIANTS: dict[str, dict] = {
    "0.6B": {
        "repo": "Qwen/Qwen3-0.6B",
        "size": 1200,
        "speed": "fast",
        "quality": "basic",
        "gpu_required": False,
    },
    "1.7B": {
        "repo": "Qwen/Qwen3-1.7B",
        "size": 3400,
        "speed": "medium",
        "quality": "good",
        "gpu_required": False,
    },
    "4B": {
        "repo": "Qwen/Qwen3-4B",
        "size": 8000,
        "speed": "slow",
        "quality": "high",
        "gpu_required": True,
    },
    "8B": {
        "repo": "Qwen/Qwen3-8B",
        "size": 16000,
        "speed": "slow",
        "quality": "high",
        "gpu_required": True,
    },
}


def _qwen3_entries() -> dict[str, ModelMetadata]:
    entries: dict[str, ModelMetadata] = {}
    for variant, info in _QWEN3_VARIANTS.items():
        model_id = f"qwen3-{variant}"
        entries[model_id] = ModelMetadata(
            family="Qwen3",
            category="llm",
            languages=list(_NLLB_COMMON_LANGS),
            size_mb=info["size"],
            speed=info["speed"],
            quality=info["quality"],
            gpu_required=info["gpu_required"],
            rationale=(
                f"Qwen3 {variant} — LLM-based prompt translation, "
                f"~{info['size']}MB"
                f"{', GPU recommended' if info['gpu_required'] else ', runs on CPU'}"
            ),
            hf_repo=info["repo"],
            required_files=list(_SAFETENSORS_OR_WEIGHTS),
        )
    return entries


# ---------------------------------------------------------------------------
# Qwen3-VL variants  (Qwen/Qwen3-VL-*)
# ---------------------------------------------------------------------------
_QWEN3_VL_VARIANTS: dict[str, dict] = {
    "2B": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "size": 4000,
        "speed": "medium",
        "quality": "good",
        "gpu_required": True,
    },
    "4B": {
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "size": 8000,
        "speed": "medium",
        "quality": "high",
        "gpu_required": True,
    },
    "8B": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "size": 16000,
        "speed": "slow",
        "quality": "high",
        "gpu_required": True,
    },
}


def _qwen3_vl_entries() -> dict[str, ModelMetadata]:
    entries: dict[str, ModelMetadata] = {}
    for variant, info in _QWEN3_VL_VARIANTS.items():
        model_id = f"qwen3-vl-{variant}"
        entries[model_id] = ModelMetadata(
            family="Qwen3-VL",
            category="vision",
            languages=list(_NLLB_COMMON_LANGS),
            size_mb=info["size"],
            speed=info["speed"],
            quality=info["quality"],
            gpu_required=info["gpu_required"],
            rationale=(
                f"Qwen3-VL {variant} — vision-language model combining OCR + translation, "
                f"~{info['size']}MB, GPU required"
            ),
            hf_repo=info["repo"],
            required_files=list(_SAFETENSORS_OR_WEIGHTS),
        )
    return entries


# ---------------------------------------------------------------------------
# OCR engines — pre-built plugins, no HF download needed for most
# ---------------------------------------------------------------------------

def _ocr_entries() -> dict[str, ModelMetadata]:
    return {
        "easyocr": ModelMetadata(
            family="EasyOCR",
            category="ocr",
            languages=[
                "en", "de", "es", "fr", "it", "pt", "ja", "zh", "ko", "ru",
                "ar", "nl", "pl", "tr", "hi", "th", "vi", "id",
            ],
            size_mb=150,
            speed="medium",
            quality="good",
            gpu_required=False,
            rationale="EasyOCR — 80+ languages, good accuracy, works on CPU and GPU",
            hf_repo=None,
            required_files=[],
        ),
        "tesseract": ModelMetadata(
            family="Tesseract",
            category="ocr",
            languages=[
                "en", "de", "es", "fr", "it", "pt", "ja", "zh", "ko", "ru",
                "ar", "nl", "pl", "tr",
            ],
            size_mb=30,
            speed="fast",
            quality="basic",
            gpu_required=False,
            rationale="Tesseract — fast, lightweight, CPU-only, best for Latin scripts",
            hf_repo=None,
            install_url="https://github.com/UB-Mannheim/tesseract/releases",
            required_files=[],
        ),
        "paddleocr": ModelMetadata(
            family="PaddleOCR",
            category="ocr",
            languages=[
                "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru",
            ],
            size_mb=100,
            speed="fast",
            quality="good",
            gpu_required=False,
            rationale="PaddleOCR — excellent CJK support, fast, optional GPU acceleration",
            hf_repo=None,
            required_files=[],
        ),
        "mokuro": ModelMetadata(
            family="Mokuro",
            category="ocr",
            languages=["ja"],
            size_mb=500,
            speed="medium",
            quality="high",
            gpu_required=False,
            rationale="Mokuro — manga page OCR with text detection and bounding boxes",
            hf_repo=None,
            required_files=[],
        ),
    }


# ---------------------------------------------------------------------------
# Public constant: union of all built-in models
# ---------------------------------------------------------------------------

BUILTIN_MODELS: dict[str, ModelMetadata] = {
    **_marianmt_entries(),
    **_nllb_entries(),
    **_m2m100_entries(),
    **_mbart_entries(),
    **_qwen3_entries(),
    **_qwen3_vl_entries(),
    **_ocr_entries(),
}
