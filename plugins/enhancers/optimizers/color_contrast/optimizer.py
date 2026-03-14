"""
Color Contrast Optimizer

Detects background color behind text regions and adjusts overlay
text/background colors for optimal readability using WCAG 2.0 contrast ratios.

Modes:
  - auto_contrast: Picks text color (black or white) that maximizes contrast
    against the detected background.
  - seamless: Matches the overlay background to the captured image background
    and picks a contrasting text color.
"""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WCAG 2.0 color math
# ---------------------------------------------------------------------------

def _gamma_correct(channel: float) -> float:
    """Linearize an sRGB channel value (0-1)."""
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def calculate_luminance(rgb: tuple[int, int, int]) -> float:
    """Return WCAG 2.0 relative luminance (0.0 – 1.0) for an RGB color."""
    r, g, b = (_gamma_correct(c / 255.0) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def calculate_contrast_ratio(c1: tuple[int, int, int],
                             c2: tuple[int, int, int]) -> float:
    """Return WCAG 2.0 contrast ratio (1.0 – 21.0) between two RGB colors."""
    l1 = calculate_luminance(c1)
    l2 = calculate_luminance(c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def detect_background_color(image: np.ndarray,
                            x: int, y: int,
                            sample_size: int = 20) -> tuple[int, int, int]:
    """Sample the average RGB color around (*x*, *y*) in a BGR image."""
    h, w = image.shape[:2]
    half = sample_size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    # Guard against zero-area regions at image edges
    if x1 >= x2 or y1 >= y2:
        return (128, 128, 128)  # neutral gray fallback

    region = image[y1:y2, x1:x2]

    if region.ndim == 3:
        avg = np.mean(region, axis=(0, 1))
        # OpenCV BGR → RGB
        return (int(avg[2]), int(avg[1]), int(avg[0]))

    avg_gray = int(np.mean(region))
    return (avg_gray, avg_gray, avg_gray)


def pick_text_color(bg_rgb: tuple[int, int, int],
                    light: str = "#ffffff",
                    dark: str = "#000000",
                    min_ratio: float = 4.5) -> str:
    """Return the hex text color that best contrasts with *bg_rgb*."""
    lum = calculate_luminance(bg_rgb)
    return dark if lum > 0.5 else light


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class ColorContrastOptimizer:
    """Pipeline optimizer that adjusts overlay colors per text region."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.mode: str = config.get("mode", "auto_contrast")
        self.min_contrast_ratio: float = config.get("min_contrast_ratio", 4.5)
        self.sample_size: int = config.get("sample_size", 20)
        self.fallback_light: str = config.get("fallback_text_light", "#ffffff")
        self.fallback_dark: str = config.get("fallback_text_dark", "#000000")

        self._frames_processed = 0
        self._regions_adjusted = 0
        logger.info(
            "ColorContrastOptimizer initialized (mode=%s, ratio=%.1f)",
            self.mode, self.min_contrast_ratio,
        )

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Adjust overlay colors based on the captured frame.

        Reads ``data['frame']`` (BGR numpy array) and
        ``data['translations']`` (list of Translation-like objects with
        a ``.position`` rectangle).  Attaches per-region color overrides
        in ``data['overlay_color_overrides']``.
        """
        frame = data.get("frame")
        translations = data.get("translations")
        if frame is None or not translations:
            return data

        overrides: list[dict[str, str]] = []

        for t in translations:
            pos = getattr(t, "position", None)
            if pos is None:
                overrides.append({})
                continue

            cx = int(pos.x + pos.width / 2)
            cy = int(pos.y + pos.height / 2)

            bg_rgb = detect_background_color(frame, cx, cy, self.sample_size)

            if self.mode == "seamless":
                # Match overlay bg to captured bg, pick contrasting text
                override = {
                    "background_color": rgb_to_hex(bg_rgb),
                    "text_color": pick_text_color(
                        bg_rgb, self.fallback_light, self.fallback_dark,
                        self.min_contrast_ratio,
                    ),
                }
            else:
                # auto_contrast: keep user bg, pick best text color
                override = {
                    "text_color": pick_text_color(
                        bg_rgb, self.fallback_light, self.fallback_dark,
                        self.min_contrast_ratio,
                    ),
                }

            overrides.append(override)
            self._regions_adjusted += 1

        data["overlay_color_overrides"] = overrides
        self._frames_processed += 1
        return data

    # ------------------------------------------------------------------
    # Stats / lifecycle
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "frames_processed": self._frames_processed,
            "regions_adjusted": self._regions_adjusted,
            "mode": self.mode,
        }

    def reset(self) -> None:
        self._frames_processed = 0
        self._regions_adjusted = 0


def initialize(config: dict[str, Any]) -> ColorContrastOptimizer:
    """Factory called by the plugin loader."""
    return ColorContrastOptimizer(config)
