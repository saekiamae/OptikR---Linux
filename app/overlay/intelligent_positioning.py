"""
Intelligent Positioning Engine

Provides smart overlay positioning with collision avoidance.
Uses a spiral search pattern to handle dense text layouts (e.g. manga pages
with many speech bubbles in close proximity).
"""

from dataclasses import dataclass
from enum import Enum

from app.models import Rectangle, Translation


class PositioningMode(Enum):
    """Positioning modes for overlay engine."""
    INTELLIGENT = "intelligent"
    SIMPLE = "simple"


@dataclass
class PositioningContext:
    """Context for positioning decisions."""
    screen_width: int = 1920
    screen_height: int = 1080


class IntelligentPositioningEngine:
    """
    Positioning engine with collision avoidance for translation overlays.
    """

    _MAX_SPIRAL_RINGS = 4

    def __init__(self, context: PositioningContext | None = None,
                 collision_padding: int = 5, screen_margin: int = 10):
        self.context = context or PositioningContext()
        self.collision_padding = collision_padding
        self.screen_margin = screen_margin
        self.overlay_region = None  # Optional dict {'x','y','width','height'} constraining overlay bounds

    def calculate_optimal_positions(self, translations, frame=None,
                                    mode=PositioningMode.INTELLIGENT):
        """
        Calculate optimal positions for translations.

        Args:
            translations: List of translations to position
            frame: Optional frame for context
            mode: Positioning mode to use

        Returns:
            List of translations with optimized positions
        """
        if mode == PositioningMode.SIMPLE:
            return translations

        return self._apply_intelligent_positioning(translations)

    def _apply_intelligent_positioning(self, translations):
        """Apply intelligent positioning with collision avoidance."""
        positioned = []
        existing_rects = []

        for translation in translations:
            original_rect = translation.position

            best_rect = self._find_best_position(original_rect, existing_rects)

            positioned_translation = Translation(
                original_text=translation.original_text,
                translated_text=translation.translated_text,
                source_language=translation.source_language,
                target_language=translation.target_language,
                position=best_rect,
                confidence=translation.confidence,
                engine_used=translation.engine_used
            )

            # Preserve optional attributes
            if hasattr(translation, 'metadata'):
                positioned_translation.metadata = translation.metadata
            if hasattr(translation, 'estimated_font_size'):
                positioned_translation.estimated_font_size = translation.estimated_font_size

            positioned.append(positioned_translation)
            existing_rects.append(best_rect)

        return positioned

    def _find_best_position(self, original_rect, existing_rects):
        """Find best position avoiding collisions using spiral search."""
        if not self._has_collision(original_rect, existing_rects):
            return original_rect

        gap = self.collision_padding + 5

        for ring in range(1, self._MAX_SPIRAL_RINGS + 1):
            candidates = self._generate_spiral_candidates(original_rect, gap, ring)
            for candidate in candidates:
                if self._is_on_screen(candidate) and not self._has_collision(candidate, existing_rects):
                    return candidate

        return original_rect

    def _generate_spiral_candidates(self, rect, gap, ring):
        """Generate candidate positions in a spiral ring around *rect*.

        Ring 1 produces the 4 cardinal + 4 diagonal neighbours (8 total).
        Each subsequent ring extends the offset further out.
        """
        dx = (rect.width + gap) * ring
        dy = (rect.height + gap) * ring
        cx, cy = rect.x, rect.y
        w, h = rect.width, rect.height

        return [
            Rectangle(cx, cy - dy, w, h),          # above
            Rectangle(cx, cy + dy, w, h),           # below
            Rectangle(cx - dx, cy, w, h),           # left
            Rectangle(cx + dx, cy, w, h),           # right
            Rectangle(cx + dx, cy - dy, w, h),      # top-right
            Rectangle(cx - dx, cy - dy, w, h),      # top-left
            Rectangle(cx + dx, cy + dy, w, h),      # bottom-right
            Rectangle(cx - dx, cy + dy, w, h),      # bottom-left
        ]

    def _has_collision(self, rect, existing_rects):
        """Check if rectangle collides with any existing rectangles."""
        p = self.collision_padding
        for existing in existing_rects:
            if not (rect.x + rect.width + p < existing.x or
                    existing.x + existing.width + p < rect.x or
                    rect.y + rect.height + p < existing.y or
                    existing.y + existing.height + p < rect.y):
                return True
        return False

    def _is_on_screen(self, rect):
        """Check if rectangle is within the overlay region (if set) or screen bounds."""
        m = self.screen_margin
        ov = self.overlay_region
        if ov and ov.get('width', 0) > 0 and ov.get('height', 0) > 0:
            return (rect.x >= ov.get('x', 0) + m and
                    rect.y >= ov.get('y', 0) + m and
                    rect.x + rect.width <= ov['x'] + ov['width'] - m and
                    rect.y + rect.height <= ov['y'] + ov['height'] - m)
        return (rect.x >= m and
                rect.y >= m and
                rect.x + rect.width <= self.context.screen_width - m and
                rect.y + rect.height <= self.context.screen_height - m)

    def update_context(self, context: PositioningContext):
        """Update positioning context."""
        self.context = context
