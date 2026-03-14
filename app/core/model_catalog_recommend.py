"""Recommendation engine for the unified ModelCatalog.

Scores and ranks models based on hardware constraints and language needs
using a composite weighted formula:

    score = language_coverage * 0.4
          + quality_score     * 0.3
          + speed_score       * 0.2
          + size_score        * 0.1

Models requiring GPU are excluded when ``gpu_available=False``.
"""

import logging

from app.core.model_catalog_types import ModelEntry, Recommendation

logger = logging.getLogger(__name__)

_QUALITY_SCORES = {"basic": 0.33, "good": 0.66, "high": 1.0}
_SPEED_SCORES = {"slow": 0.33, "medium": 0.66, "fast": 1.0}


class RecommendationEngine:
    """Scores and ranks models for recommendations."""

    def recommend(
        self,
        models: list[ModelEntry],
        gpu_available: bool,
        languages_needed: list[str],
    ) -> list[Recommendation]:
        """Filter by GPU, score, and return a descending-ranked list.

        Parameters
        ----------
        models:
            Candidate ``ModelEntry`` objects (typically from ``list_available``).
        gpu_available:
            Whether the user's system has a usable GPU.
        languages_needed:
            ISO-639 language codes the user cares about (e.g. ``["en", "de"]``).

        Returns
        -------
        list[Recommendation]
            Sorted best-first.  Empty when no candidates survive filtering.
        """
        if not models:
            return []

        candidates = [
            m for m in models
            if gpu_available or not m.metadata.gpu_required
        ]

        if not candidates:
            return []

        max_size = max(m.metadata.size_mb for m in candidates) or 1.0

        results: list[Recommendation] = []
        for entry in candidates:
            meta = entry.metadata

            lang_coverage = self._language_coverage(meta.languages, languages_needed)
            quality = _QUALITY_SCORES.get(meta.quality, 0.5)
            speed = _SPEED_SCORES.get(meta.speed, 0.5)
            size = 1.0 - (meta.size_mb / max_size) if max_size > 0 else 0.5

            score = (
                lang_coverage * 0.4
                + quality * 0.3
                + speed * 0.2
                + size * 0.1
            )

            explanation = self._build_explanation(
                entry, lang_coverage, languages_needed
            )
            results.append(Recommendation(model=entry, score=round(score, 4), explanation=explanation))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _language_coverage(model_langs: list[str], needed: list[str]) -> float:
        if not needed:
            return 1.0
        model_set = set(model_langs)
        covered = sum(1 for lang in needed if lang in model_set)
        return covered / len(needed)

    @staticmethod
    def _build_explanation(
        entry: ModelEntry,
        lang_coverage: float,
        languages_needed: list[str],
    ) -> str:
        meta = entry.metadata
        parts: list[str] = []

        if languages_needed:
            covered = int(lang_coverage * len(languages_needed))
            parts.append(f"Covers {covered}/{len(languages_needed)} requested languages")
        else:
            parts.append(f"{meta.family} model")

        parts.append(f"{meta.quality} quality")
        parts.append(f"{meta.speed} speed")

        if meta.gpu_required:
            parts.append("GPU required")
        else:
            parts.append("runs on CPU")

        return ", ".join(parts)
