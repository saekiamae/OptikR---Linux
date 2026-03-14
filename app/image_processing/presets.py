"""Built-in and user-custom presets for image processing settings.

Presets are named bundles of image-processing configuration that can be
applied with a single click.  Three categories exist:

* **Content presets** -- auto-configure OCR, translation, font style,
  erasure, and rendering for common content types (manga, game UI, etc.).
* **Style presets** -- only affect rendering appearance (font, colors,
  background, border) without changing OCR/translation behaviour.
* **Custom presets** -- user-defined, persisted as individual JSON files
  under ``user_data/image_processing_presets/``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.utils.path_utils import get_image_processing_presets_dir

logger = logging.getLogger(__name__)

# Config keys written/read by presets (without the ``image_processing.`` prefix).
_STYLE_KEYS: list[str] = [
    "font_family",
    "font_size",
    "auto_font_size",
    "text_color",
    "background_color",
    "background_enabled",
    "background_opacity",
    "border_enabled",
    "padding",
]

_ERASURE_KEYS: list[str] = [
    "erase_original_text",
    "inpaint_method",
]

_OUTPUT_KEYS: list[str] = [
    "output_format",
    "naming_pattern",
    "naming_suffix",
    "jpg_quality",
]

_OCR_TRANSLATION_KEYS: list[str] = [
    "use_main_ocr_settings",
    "use_main_translation_settings",
]

ALL_PRESET_KEYS: list[str] = (
    _STYLE_KEYS + _ERASURE_KEYS + _OUTPUT_KEYS + _OCR_TRANSLATION_KEYS
)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class ImageProcessingPreset:
    """A named bundle of image-processing settings."""

    name: str
    type: str  # "content", "style", or "custom"
    description: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "created": self.created,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageProcessingPreset:
        return cls(
            name=data.get("name", "Unnamed"),
            type=data.get("type", "custom"),
            description=data.get("description", ""),
            settings=dict(data.get("settings", {})),
            created=data.get("created", ""),
        )


# ------------------------------------------------------------------
# Built-in content-type presets
# ------------------------------------------------------------------

def _content_presets() -> list[ImageProcessingPreset]:
    """Return the built-in content-type presets."""
    return [
        ImageProcessingPreset(
            name="Manga / Comics",
            type="content",
            description=(
                "Optimised for manga and comic pages: auto-sized bold text, "
                "original text erased via smart inpainting, white on black."
            ),
            settings={
                "auto_font_size": True,
                "erase_original_text": True,
                "inpaint_method": "inpaint",
                "font_family": "Arial Black",
                "font_size": 18,
                "text_color": "#FFFFFF",
                "background_color": "#000000",
                "background_enabled": True,
                "background_opacity": 0.9,
                "border_enabled": True,
                "padding": 8,
                "use_main_ocr_settings": False,
                "use_main_translation_settings": False,
            },
        ),
        ImageProcessingPreset(
            name="Game UI / Screenshots",
            type="content",
            description=(
                "For game user-interface screenshots: keeps original graphics, "
                "semi-transparent background, smaller font."
            ),
            settings={
                "auto_font_size": True,
                "erase_original_text": False,
                "inpaint_method": "solid_fill",
                "font_family": "Segoe UI",
                "font_size": 14,
                "text_color": "#FFFFFF",
                "background_color": "#1A1A2E",
                "background_enabled": True,
                "background_opacity": 0.75,
                "border_enabled": False,
                "padding": 4,
                "use_main_ocr_settings": False,
                "use_main_translation_settings": False,
            },
        ),
        ImageProcessingPreset(
            name="Documents / Formal",
            type="content",
            description=(
                "Clean document translation: erases original text with solid "
                "fill, serif font, black on white."
            ),
            settings={
                "auto_font_size": True,
                "erase_original_text": True,
                "inpaint_method": "solid_fill",
                "font_family": "Times New Roman",
                "font_size": 14,
                "text_color": "#000000",
                "background_color": "#FFFFFF",
                "background_enabled": True,
                "background_opacity": 1.0,
                "border_enabled": False,
                "padding": 4,
                "use_main_ocr_settings": False,
                "use_main_translation_settings": False,
            },
        ),
        ImageProcessingPreset(
            name="Subtitles / Captions",
            type="content",
            description=(
                "Subtitle-style: large centred text, white on dark shadow, "
                "no background rectangle, original text preserved."
            ),
            settings={
                "auto_font_size": False,
                "erase_original_text": False,
                "inpaint_method": "solid_fill",
                "font_family": "Segoe UI",
                "font_size": 24,
                "text_color": "#FFFFFF",
                "background_color": "#000000",
                "background_enabled": False,
                "background_opacity": 0.0,
                "border_enabled": False,
                "padding": 6,
                "use_main_ocr_settings": True,
                "use_main_translation_settings": True,
            },
        ),
        ImageProcessingPreset(
            name="Web Pages",
            type="content",
            description=(
                "Web-page translation: erases original text, auto-sized "
                "sans-serif font, solid fill."
            ),
            settings={
                "auto_font_size": True,
                "erase_original_text": True,
                "inpaint_method": "solid_fill",
                "font_family": "Segoe UI",
                "font_size": 14,
                "text_color": "#222222",
                "background_color": "#FFFFFF",
                "background_enabled": True,
                "background_opacity": 1.0,
                "border_enabled": False,
                "padding": 4,
                "use_main_ocr_settings": True,
                "use_main_translation_settings": True,
            },
        ),
        ImageProcessingPreset(
            name="Novel / Book",
            type="content",
            description=(
                "Clean book-style: serif font, black text, no background "
                "or border, solid fill erasure."
            ),
            settings={
                "auto_font_size": True,
                "erase_original_text": True,
                "inpaint_method": "solid_fill",
                "font_family": "Times New Roman",
                "font_size": 14,
                "text_color": "#000000",
                "background_color": "#FFFFFF",
                "background_enabled": False,
                "background_opacity": 0.0,
                "border_enabled": False,
                "padding": 4,
                "use_main_ocr_settings": True,
                "use_main_translation_settings": True,
            },
        ),
    ]


# ------------------------------------------------------------------
# Built-in visual style presets
# ------------------------------------------------------------------

def _style_presets() -> list[ImageProcessingPreset]:
    """Return the built-in visual-style presets."""
    return [
        ImageProcessingPreset(
            name="Clean / Minimal",
            type="style",
            description="White text, no background, no border, subtle and clean.",
            settings={
                "font_family": "Segoe UI",
                "auto_font_size": True,
                "font_size": 16,
                "text_color": "#FFFFFF",
                "background_color": "#000000",
                "background_enabled": False,
                "background_opacity": 0.0,
                "border_enabled": False,
                "padding": 4,
            },
        ),
        ImageProcessingPreset(
            name="Comic Style",
            type="style",
            description="Bold font, white text, black background, thick border, high opacity.",
            settings={
                "font_family": "Arial Black",
                "auto_font_size": True,
                "font_size": 18,
                "text_color": "#FFFFFF",
                "background_color": "#000000",
                "background_enabled": True,
                "background_opacity": 1.0,
                "border_enabled": True,
                "padding": 8,
            },
        ),
        ImageProcessingPreset(
            name="Professional",
            type="style",
            description="Dark text, light translucent background, thin border, padded.",
            settings={
                "font_family": "Segoe UI",
                "auto_font_size": True,
                "font_size": 14,
                "text_color": "#1A1A1A",
                "background_color": "#F0F0F0",
                "background_enabled": True,
                "background_opacity": 0.8,
                "border_enabled": True,
                "padding": 6,
            },
        ),
        ImageProcessingPreset(
            name="High Contrast",
            type="style",
            description="Bright yellow text on solid black background, full opacity.",
            settings={
                "font_family": "Segoe UI",
                "auto_font_size": True,
                "font_size": 16,
                "text_color": "#FFFF00",
                "background_color": "#000000",
                "background_enabled": True,
                "background_opacity": 1.0,
                "border_enabled": False,
                "padding": 6,
            },
        ),
        ImageProcessingPreset(
            name="Transparent",
            type="style",
            description="White text, low-opacity dark background, no border.",
            settings={
                "font_family": "Segoe UI",
                "auto_font_size": True,
                "font_size": 16,
                "text_color": "#FFFFFF",
                "background_color": "#000000",
                "background_enabled": True,
                "background_opacity": 0.35,
                "border_enabled": False,
                "padding": 4,
            },
        ),
        ImageProcessingPreset(
            name="Outline Only",
            type="style",
            description="White text with dark outline, no background fill.",
            settings={
                "font_family": "Segoe UI",
                "auto_font_size": True,
                "font_size": 16,
                "text_color": "#FFFFFF",
                "background_color": "#333333",
                "background_enabled": False,
                "background_opacity": 0.0,
                "border_enabled": True,
                "padding": 4,
            },
        ),
    ]


# ------------------------------------------------------------------
# Preset manager
# ------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Turn a human-readable preset name into a safe filename stem."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = safe.strip(". ")
    return safe or "preset"


class PresetManager:
    """Manages built-in and user-custom image-processing presets.

    Parameters
    ----------
    presets_dir : str | Path, optional
        Directory where custom preset JSON files are stored.
        Defaults to the canonical path from ``path_utils``
        (``user_data/image_processing_presets/``).
    """

    def __init__(self, presets_dir: str | None = None) -> None:
        self._presets_dir = str(presets_dir) if presets_dir else str(get_image_processing_presets_dir())
        self._content_presets = _content_presets()
        self._style_presets = _style_presets()

    # ------------------------------------------------------------------
    # Built-in preset access
    # ------------------------------------------------------------------

    def get_content_presets(self) -> list[ImageProcessingPreset]:
        """Return all built-in content-type presets."""
        return list(self._content_presets)

    def get_style_presets(self) -> list[ImageProcessingPreset]:
        """Return all built-in visual-style presets."""
        return list(self._style_presets)

    # ------------------------------------------------------------------
    # Custom preset CRUD
    # ------------------------------------------------------------------

    def get_custom_presets(self) -> list[ImageProcessingPreset]:
        """Load all user-defined custom presets from disk."""
        presets: list[ImageProcessingPreset] = []
        if not os.path.isdir(self._presets_dir):
            return presets

        for filename in sorted(os.listdir(self._presets_dir)):
            if not filename.lower().endswith(".json"):
                continue
            filepath = os.path.join(self._presets_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                preset = ImageProcessingPreset.from_dict(data)
                preset.type = "custom"
                presets.append(preset)
            except Exception as exc:
                logger.warning("Failed to load preset '%s': %s", filepath, exc)

        return presets

    def save_custom_preset(
        self, name: str, settings: dict[str, Any]
    ) -> ImageProcessingPreset:
        """Save a custom preset to disk and return it.

        Parameters
        ----------
        name :
            Human-readable preset name.
        settings :
            Dict of image-processing setting values (keys without the
            ``image_processing.`` prefix).
        """
        os.makedirs(self._presets_dir, exist_ok=True)
        preset = ImageProcessingPreset(
            name=name,
            type="custom",
            description="User-defined preset",
            settings=dict(settings),
            created=datetime.now(timezone.utc).isoformat(),
        )
        filepath = os.path.join(self._presets_dir, f"{_safe_filename(name)}.json")
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(preset.to_dict(), fh, indent=4, ensure_ascii=False)
        logger.info("Saved custom preset '%s' -> %s", name, filepath)
        return preset

    def delete_custom_preset(self, name: str) -> bool:
        """Delete a custom preset JSON file by name.

        Returns ``True`` if the file was found and removed.
        """
        if not os.path.isdir(self._presets_dir):
            return False

        target = f"{_safe_filename(name)}.json"
        filepath = os.path.join(self._presets_dir, target)
        if os.path.isfile(filepath):
            os.remove(filepath)
            logger.info("Deleted custom preset '%s'", filepath)
            return True

        # Fallback: scan by preset name stored inside the JSON
        for filename in os.listdir(self._presets_dir):
            if not filename.lower().endswith(".json"):
                continue
            fpath = os.path.join(self._presets_dir, filename)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("name") == name:
                    os.remove(fpath)
                    logger.info("Deleted custom preset '%s'", fpath)
                    return True
            except Exception:
                continue

        logger.warning("Custom preset '%s' not found for deletion", name)
        return False

    # ------------------------------------------------------------------
    # Apply / create from config
    # ------------------------------------------------------------------

    def apply_preset(self, preset: ImageProcessingPreset, config_manager: Any) -> None:
        """Write a preset's settings into the live configuration.

        Content presets write all keys; style presets write only the
        visual-style subset.
        """
        keys_to_write = (
            ALL_PRESET_KEYS if preset.type != "style" else _STYLE_KEYS
        )
        for key in keys_to_write:
            if key in preset.settings:
                config_manager.set_setting(
                    f"image_processing.{key}", preset.settings[key]
                )

        config_manager.set_setting("image_processing.active_preset", preset.name)
        config_manager.set_setting("image_processing.last_used_preset_type", preset.type)

    def create_preset_from_current(
        self, name: str, config_manager: Any
    ) -> ImageProcessingPreset:
        """Read the current image-processing settings and persist them as
        a new custom preset."""
        settings: dict[str, Any] = {}
        for key in ALL_PRESET_KEYS:
            val = config_manager.get_setting(f"image_processing.{key}")
            if val is not None:
                settings[key] = val
        return self.save_custom_preset(name, settings)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_all_presets(self) -> list[ImageProcessingPreset]:
        """Return content + style + custom presets as a flat list."""
        return (
            self.get_content_presets()
            + self.get_style_presets()
            + self.get_custom_presets()
        )

    def find_preset(self, name: str) -> ImageProcessingPreset | None:
        """Find a preset by name across all categories."""
        for preset in self.get_all_presets():
            if preset.name == name:
                return preset
        return None
