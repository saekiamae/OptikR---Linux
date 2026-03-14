"""Unified data types for the ModelCatalog system.

Replaces the disconnected types from ModelDownloader (ModelInfo),
UniversalModelManager (TranslationModel), and OCRModelManager (OCRModel)
with a single set of dataclasses used across all model lifecycle operations.
"""

from dataclasses import dataclass, field


@dataclass
class ModelMetadata:
    """Static metadata describing a model family member."""

    family: str
    category: str  # "translation", "ocr", or "llm"
    languages: list[str]
    size_mb: float
    speed: str  # "fast", "medium", "slow"
    quality: str  # "basic", "good", "high"
    gpu_required: bool
    rationale: str
    hf_repo: str | None = None
    install_url: str | None = None
    required_files: list[str] = field(default_factory=lambda: ["config.json"])


@dataclass
class ModelStatus:
    """Mutable state tracked per model in the registry."""

    downloaded: bool = False
    plugin_registered: bool = False
    enabled: bool = False
    download_timestamp: str | None = None  # ISO 8601
    local_path: str | None = None


@dataclass
class ModelEntry:
    """A complete model record combining identity, metadata, and status."""

    model_id: str
    family: str
    category: str  # "translation", "ocr", or "llm"
    metadata: ModelMetadata
    status: ModelStatus = field(default_factory=ModelStatus)


@dataclass
class Recommendation:
    """A ranked recommendation returned by the RecommendationEngine."""

    model: ModelEntry
    score: float
    explanation: str


@dataclass
class CacheInfo:
    """Aggregate statistics about the model cache."""

    total_models: int
    total_size_mb: float
    available_space_gb: float
