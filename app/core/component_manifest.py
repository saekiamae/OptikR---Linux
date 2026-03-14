"""ComponentManifest — maps built-in model IDs to wizard categories.

A pure-data module with no I/O and no Qt dependency.  It reads
BUILTIN_MODELS and classifies each model as essential or optional,
deriving display names and descriptions from ModelMetadata fields.
"""

from dataclasses import dataclass

from app.core.model_catalog_metadata import BUILTIN_MODELS
from app.core.model_catalog_types import ModelMetadata


@dataclass
class ComponentInfo:
    """A component entry for the wizard selector."""

    model_id: str
    display_name: str
    description: str
    size_mb: float
    category: str  # "essential" or "optional"
    component_type: str  # "translation", "ocr", "llm", or "vision"
    languages: list[str]
    gpu_required: bool


# Essential model IDs — the minimum viable set
ESSENTIAL_IDS: set[str] = {"marianmt-en-de", "easyocr"}


def _derive_display_name(meta: ModelMetadata) -> str:
    """Build a human-readable display name from model metadata."""
    if meta.category == "ocr":
        return meta.family
    if meta.category == "llm":
        return f"{meta.family} ({len(meta.languages)} languages)"
    if meta.category == "vision":
        return f"{meta.family} ({len(meta.languages)} languages)"
    # Translation models
    if len(meta.languages) == 2:
        src, tgt = meta.languages
        return f"{meta.family} {src.upper()}\u2192{tgt.upper()}"
    return f"{meta.family} ({len(meta.languages)} languages)"


def get_all_components() -> list[ComponentInfo]:
    """Return all wizard-selectable components derived from BUILTIN_MODELS."""
    components: list[ComponentInfo] = []
    for model_id, meta in BUILTIN_MODELS.items():
        components.append(
            ComponentInfo(
                model_id=model_id,
                display_name=_derive_display_name(meta),
                description=meta.rationale,
                size_mb=meta.size_mb,
                category="essential" if model_id in ESSENTIAL_IDS else "optional",
                component_type=meta.category,
                languages=list(meta.languages),
                gpu_required=meta.gpu_required,
            )
        )
    return components


def get_components_by_category(category: str) -> list[ComponentInfo]:
    """Filter components by ``'essential'`` or ``'optional'``."""
    return [c for c in get_all_components() if c.category == category]


def compute_total_size(model_ids: list[str]) -> float:
    """Sum *size_mb* for the given model IDs.  Unknown IDs are skipped."""
    lookup = {c.model_id: c.size_mb for c in get_all_components()}
    return sum(lookup.get(mid, 0.0) for mid in model_ids)


def format_size(size_mb: float) -> str:
    """Format a size value: ``'X MB'`` when < 1024, else ``'X.X GB'``."""
    if size_mb < 1024:
        return f"{round(size_mb)} MB"
    return f"{size_mb / 1024:.1f} GB"
