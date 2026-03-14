"""Import handler for the unified ModelCatalog.

Validates, copies, and registers manually imported model directories.
Family detection inspects ``config.json``'s ``model_type`` field plus
structural markers (e.g. ``.traineddata`` for Tesseract, ``inference.pdmodel``
for PaddleOCR).
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Required files per family (beyond config.json which is checked separately).
_FAMILY_REQUIRED: dict[str, list[str]] = {
    "MarianMT": ["tokenizer_config.json"],
    "NLLB-200": ["tokenizer_config.json"],
    "M2M-100": ["tokenizer_config.json"],
    "mBART": ["tokenizer_config.json"],
    "MangaOCR": [],
    "EasyOCR": [],
    "PaddleOCR": [],
    "Tesseract": [],
}

# Mapping from config.json ``model_type`` values to family names.
_MODEL_TYPE_MAP: dict[str, str] = {
    "marian": "MarianMT",
    "mbart": "mBART",
    "m2m_100": "M2M-100",  # NLLB also reports m2m_100; disambiguated below
}

# Category inferred from family.
_FAMILY_CATEGORY: dict[str, str] = {
    "MarianMT": "translation",
    "NLLB-200": "translation",
    "M2M-100": "translation",
    "mBART": "translation",
    "EasyOCR": "ocr",
    "Tesseract": "ocr",
    "PaddleOCR": "ocr",
    "MangaOCR": "ocr",
}


class ImportHandler:
    """Handles manual model import: detect family, validate, copy, register."""

    # ------------------------------------------------------------------
    # Family detection
    # ------------------------------------------------------------------

    def detect_family(self, source_path: Path) -> str | None:
        """Inspect files to determine the model family.

        Returns the family name string (e.g. ``"MarianMT"``) or ``None`` if
        unrecognised.
        """
        source = Path(source_path)
        if not source.is_dir():
            logger.warning("Source is not a directory: %s", source)
            return None

        # --- Tesseract (*.traineddata) ---
        if any(source.glob("*.traineddata")):
            return "Tesseract"

        # --- PaddleOCR (inference.pdmodel or ppocr config) ---
        if (source / "inference.pdmodel").exists():
            return "PaddleOCR"
        if any(source.glob("**/inference.pdmodel")):
            return "PaddleOCR"

        # --- EasyOCR structural marker ---
        if (source / "easyocr").exists() or (source / "model.pth").exists():
            if not (source / "config.json").exists():
                return "EasyOCR"

        # --- HF-style models with config.json ---
        config_path = source / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Cannot read config.json in %s: %s", source, exc)
                return None

            model_type = cfg.get("model_type", "").lower()

            # MangaOCR: vision-encoder-decoder with manga indicators
            if model_type == "vision-encoder-decoder":
                name_hint = cfg.get("_name_or_path", "").lower()
                if "manga" in name_hint or "manga" in source.name.lower():
                    return "MangaOCR"

            # m2m_100 can be either NLLB or plain M2M-100
            if model_type == "m2m_100":
                name_hint = cfg.get("_name_or_path", "").lower()
                if "nllb" in name_hint or "nllb" in source.name.lower():
                    return "NLLB-200"
                return "M2M-100"

            family = _MODEL_TYPE_MAP.get(model_type)
            if family:
                return family

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, source_path: Path, family: str) -> tuple[bool, str]:
        """Check required files for the detected family.

        Returns ``(True, "")`` on success or ``(False, error_message)`` on
        failure.
        """
        source = Path(source_path)
        if not source.is_dir():
            return False, f"Source directory not found: {source}"

        missing: list[str] = []

        # Tesseract has no config.json; others do
        if family != "Tesseract":
            needs_config = family not in ("EasyOCR", "PaddleOCR")
            if needs_config and not (source / "config.json").exists():
                missing.append("config.json")

        # Check for at least one weights file (skip for Tesseract / PaddleOCR / EasyOCR)
        if family in ("MarianMT", "NLLB-200", "M2M-100", "mBART", "MangaOCR"):
            has_weights = (
                (source / "pytorch_model.bin").exists()
                or (source / "model.safetensors").exists()
                or any(source.glob("*.pth"))
                or any(source.glob("pytorch_model*.bin"))
                or any(source.glob("model*.safetensors"))
            )
            if not has_weights:
                missing.append("weights file (pytorch_model.bin / model.safetensors)")

        # Family-specific extras
        for req in _FAMILY_REQUIRED.get(family, []):
            if not (source / req).exists():
                missing.append(req)

        if missing:
            return False, f"Missing required files for {family}: {', '.join(missing)}"
        return True, ""

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy_to_cache(self, source_path: Path, family: str, cache_dir: Path) -> Path:
        """Copy model files into the appropriate cache directory structure.

        Returns the destination ``Path``.  On failure raises ``OSError``.
        """
        source = Path(source_path)
        dest = cache_dir / source.name
        if dest.exists():
            logger.info("Destination already exists, removing: %s", dest)
            shutil.rmtree(dest)
        try:
            shutil.copytree(source, dest)
        except OSError:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            raise
        logger.info("Copied %s -> %s", source, dest)
        return dest

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_category(family: str) -> str:
        """Return ``'translation'`` or ``'ocr'`` for a known family."""
        return _FAMILY_CATEGORY.get(family, "translation")
