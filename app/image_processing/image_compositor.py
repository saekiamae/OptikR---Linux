"""Image compositor for rendering translated text onto images.

Uses Pillow for TrueType font rendering with anti-aliasing and optional
OpenCV inpainting for background reconstruction when erasing original text.
"""

import logging
import os
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models import Rectangle

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "erase_original_text": True,
    "inpaint_method": "solid_fill",
    "font_family": "Segoe UI",
    "font_size": 16,
    "auto_font_size": True,
    "text_color": "#FFFFFF",
    "background_color": "#000000",
    "background_enabled": True,
    "background_opacity": 0.85,
    "border_enabled": False,
    "padding": 6,
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an ``(R, G, B)`` tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    try:
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    except ValueError:
        return (255, 255, 255)


def _get_position(block: Any) -> Rectangle:
    """Extract a :class:`Rectangle` from a ``TextBlock`` or dict."""
    if isinstance(block, dict):
        pos = block.get("position", {})
        if isinstance(pos, Rectangle):
            return pos
        return Rectangle(
            x=pos.get("x", 0),
            y=pos.get("y", 0),
            width=pos.get("width", 100),
            height=pos.get("height", 30),
        )
    return getattr(block, "position", Rectangle(0, 0, 100, 30))


@lru_cache(maxsize=64)
def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font with a cascade of candidate paths."""
    candidates: list[str] = [family, f"{family}.ttf", f"{family}.TTF"]

    win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for variant in (family, family.replace(" ", ""), family.replace(" ", "").lower()):
        candidates.append(os.path.join(win_fonts, f"{variant}.ttf"))

    candidates.extend([
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        f"/usr/share/fonts/truetype/{family.lower().replace(' ', '')}.ttf",
        f"/System/Library/Fonts/{family}.ttf",
    ])

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (IOError, OSError):
            continue

    for fallback in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except (IOError, OSError):
            continue

    logger.warning("Could not load TrueType font '%s', using default", family)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_text(text: str, font: Any, max_width: int) -> str:
    """Wrap *text* to fit within *max_width* pixels.

    Uses word-level splitting when spaces are present, otherwise falls
    back to character-level wrapping (useful for CJK text).
    """
    if not text:
        return text

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox((0, 0), text, font=font)
    if (bbox[2] - bbox[0]) <= max_width:
        return text

    words = text.split()
    if len(words) > 1:
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            bbox = draw.textbbox((0, 0), test, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines)

    lines = []
    current = ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return "\n".join(lines)


class ImageCompositor:
    """Renders translated text onto images using Pillow.

    Parameters
    ----------
    config : dict, optional
        Default style configuration.  Individual calls to
        :meth:`composite` can override any key.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = {**_DEFAULTS, **(config or {})}

    def composite(
        self,
        image: np.ndarray,
        text_blocks: list[Any],
        translations: list[str],
        config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Composite translated text onto a BGR image.

        Parameters
        ----------
        image :
            BGR ``numpy.ndarray``.
        text_blocks :
            ``TextBlock`` objects (or dicts) with ``position`` info.
        translations :
            Translated strings, one per text block.
        config :
            Per-call style overrides.

        Returns
        -------
        numpy.ndarray
            BGR image with translated text rendered.
        """
        cfg = {**self._config, **(config or {})}
        result = image.copy()

        pairs = [(b, t) for b, t in zip(text_blocks, translations) if t]
        if not pairs:
            return result

        # --- Phase 1: erase original text (numpy / OpenCV) ---
        if cfg.get("erase_original_text", True):
            positions = [_get_position(b) for b, _ in pairs]
            result = self._erase_regions(result, positions, cfg)

        # --- Phase 2: draw translated text (PIL / Pillow) ---
        is_bgr = len(result.shape) == 3 and result.shape[2] >= 3
        if is_bgr:
            pil_image = Image.fromarray(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            ).convert("RGBA")
        else:
            pil_image = Image.fromarray(result).convert("RGBA")

        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        text_rgb = _hex_to_rgb(cfg.get("text_color", "#FFFFFF"))
        bg_rgb = _hex_to_rgb(cfg.get("background_color", "#000000"))
        bg_alpha = int(cfg.get("background_opacity", 0.85) * 255)
        bg_enabled = cfg.get("background_enabled", True)
        border_enabled = cfg.get("border_enabled", False)
        padding = cfg.get("padding", 6)
        font_family = cfg.get("font_family", "Segoe UI")
        auto_font = cfg.get("auto_font_size", True)
        base_font_size = cfg.get("font_size", 16)

        for block, translation in pairs:
            try:
                pos = _get_position(block)
                available_w = max(pos.width - 2 * padding, 10)

                if auto_font:
                    font_size = self._auto_font_size(
                        translation, pos, font_family, padding,
                    )
                else:
                    font_size = base_font_size

                font = _load_font(font_family, font_size)
                wrapped = _wrap_text(translation, font, available_w)

                bbox = draw.textbbox((0, 0), wrapped, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                tx = pos.x + (pos.width - text_w) // 2
                ty = pos.y + (pos.height - text_h) // 2

                if bg_enabled:
                    draw.rectangle(
                        (pos.x, pos.y, pos.x + pos.width, pos.y + pos.height),
                        fill=(*bg_rgb, bg_alpha),
                    )

                if border_enabled:
                    draw.rectangle(
                        (pos.x, pos.y,
                         pos.x + pos.width - 1, pos.y + pos.height - 1),
                        outline=(*text_rgb, 255),
                        width=1,
                    )

                draw.text(
                    (tx, ty), wrapped,
                    fill=(*text_rgb, 255),
                    font=font,
                )
            except Exception as exc:
                logger.warning("Failed to composite block: %s", exc)
                continue

        composited = Image.alpha_composite(pil_image, overlay).convert("RGB")
        result = np.array(composited)
        if is_bgr:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return result

    # ------------------------------------------------------------------
    # Text erasure
    # ------------------------------------------------------------------

    @staticmethod
    def _erase_regions(
        image: np.ndarray,
        positions: list[Rectangle],
        cfg: dict[str, Any],
    ) -> np.ndarray:
        """Erase text at the given bounding-box positions."""
        method = cfg.get("inpaint_method", "solid_fill")
        result = image.copy()
        h, w = result.shape[:2]

        if method == "inpaint":
            mask = np.zeros((h, w), dtype=np.uint8)
            for pos in positions:
                x1 = max(0, pos.x)
                y1 = max(0, pos.y)
                x2 = min(w, pos.x + pos.width)
                y2 = min(h, pos.y + pos.height)
                mask[y1:y2, x1:x2] = 255
            if mask.any():
                result = cv2.inpaint(
                    result, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA,
                )
        else:
            bg_rgb = _hex_to_rgb(cfg.get("background_color", "#000000"))
            fill_bgr = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
            for pos in positions:
                x1 = max(0, pos.x)
                y1 = max(0, pos.y)
                x2 = min(w, pos.x + pos.width)
                y2 = min(h, pos.y + pos.height)
                result[y1:y2, x1:x2] = fill_bgr

        return result

    # ------------------------------------------------------------------
    # Auto font sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_font_size(
        text: str,
        rect: Rectangle,
        font_family: str,
        padding: int,
    ) -> int:
        """Binary-search for the largest font size that fits *text* inside *rect*."""
        available_w = max(rect.width - 2 * padding, 10)
        available_h = max(rect.height - 2 * padding, 10)

        dummy = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy)

        lo, hi = 8, min(available_h, 72)
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = _load_font(font_family, mid)
            wrapped = _wrap_text(text, font, available_w)
            bbox = dummy_draw.textbbox((0, 0), wrapped, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            if tw <= available_w and th <= available_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return max(best, 8)
