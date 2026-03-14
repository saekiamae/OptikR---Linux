"""Unified Model Catalog — singleton service for model lifecycle management.

Replaces ``ModelDownloader``, ``UniversalModelManager``, and
``OCRModelManager`` with a single API consumed by the Wizard, Translation
UI, and OCR UI.

Threading model
---------------
* Public methods are called from the UI thread.
* ``download()`` runs the HF download on a background ``threading.Thread``
  and invokes *progress_callback* from that thread.  UI consumers must
  marshal callbacks to the Qt event loop themselves.
* ``cancel()`` sets a ``threading.Event`` checked by the download thread.
* Registry writes are protected by an internal ``threading.Lock``.
"""
from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.model_catalog_types import (
    CacheInfo,
    ModelEntry,
    ModelMetadata,
    ModelStatus,
    Recommendation,
)
from app.core.model_catalog_metadata import BUILTIN_MODELS
from app.core.model_catalog_registry import RegistryManager
from app.core.model_catalog_download import DownloadManager
from app.core.model_catalog_recommend import RecommendationEngine
from app.core.model_catalog_import import ImportHandler

logger = logging.getLogger(__name__)


class ModelCatalog:
    """Singleton service for unified model lifecycle management."""

    _instance: "ModelCatalog" | None = None
    _init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "ModelCatalog":
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Tear down the singleton (test-only)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        from app.utils.path_utils import get_models_dir, get_hf_cache_dir

        self._hf_cache = get_hf_cache_dir()

        registry_path = get_models_dir() / "model_registry.json"

        self._registry = RegistryManager(
            registry_path=registry_path,
            hf_cache_dir=self._hf_cache,
        )

        self._cancel_event = threading.Event()
        self._downloader = DownloadManager(cancel_event=self._cancel_event)
        self._recommender = RecommendationEngine()
        self._importer = ImportHandler()

        self._custom_models: dict[str, ModelEntry] = {}

        self._seed_registry()
        self._registry.scan_local_models()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_registry(self) -> None:
        """Ensure every built-in model has a registry entry."""
        for model_id, meta in BUILTIN_MODELS.items():
            if self._registry.get_entry(model_id) is None:
                self._registry.update_entry(
                    model_id,
                    family=meta.family,
                    category=meta.category,
                    hf_repo=meta.hf_repo,
                    languages=meta.languages,
                    size_mb=meta.size_mb,
                    downloaded=False,
                    plugin_registered=False,
                    enabled=False,
                    custom=False,
                )

    # ------------------------------------------------------------------
    # Public API — queries
    # ------------------------------------------------------------------

    def list_available(self, category: str) -> list[ModelEntry]:
        """List all models for *category* (``'translation'`` or ``'ocr'``).

        Returns ``ModelEntry`` objects with merged metadata + status.
        """
        results: list[ModelEntry] = []

        for model_id, meta in BUILTIN_MODELS.items():
            if meta.category != category:
                continue
            status = self._status_from_registry(model_id)
            results.append(ModelEntry(
                model_id=model_id,
                family=meta.family,
                category=meta.category,
                metadata=meta,
                status=status,
            ))

        for model_id, entry in self._custom_models.items():
            if entry.category == category:
                status = self._status_from_registry(model_id)
                results.append(ModelEntry(
                    model_id=model_id,
                    family=entry.family,
                    category=entry.category,
                    metadata=entry.metadata,
                    status=status,
                ))

        return results

    def get_status(self, model_id: str) -> ModelStatus:
        """Get download/plugin/enabled status for a model."""
        return self._status_from_registry(model_id)

    def is_downloaded(self, model_id: str) -> bool:
        """Return True if the model has been downloaded."""
        return self._status_from_registry(model_id).downloaded

    # ------------------------------------------------------------------
    # Public API — download / cancel
    # ------------------------------------------------------------------

    def download(
        self,
        model_id: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """Download a model from HF Hub with retry.  Returns True on success."""
        meta = self._resolve_metadata(model_id)
        if meta is None:
            logger.error("Unknown model_id: %s", model_id)
            return False

        current_status = self._status_from_registry(model_id)
        if current_status.downloaded:
            logger.info("Model %s already downloaded", model_id)
            if progress_callback:
                progress_callback(1.0, "Already downloaded")
            return True

        if not meta.hf_repo and meta.install_url:
            if progress_callback:
                progress_callback(
                    1.0,
                    f"Manual install required — download from: {meta.install_url}",
                )
            self._registry.update_entry(
                model_id,
                downloaded=True,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.register_plugin(model_id)
            return True

        # Built-in OCR engines are package/plugin backed and have no dedicated
        # HF repository download step. Treat them as available and register.
        if not meta.hf_repo and meta.category == "ocr":
            if progress_callback:
                progress_callback(1.0, "No model download required for this OCR engine")
            self._registry.update_entry(
                model_id,
                downloaded=True,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.register_plugin(model_id)
            return True

        if not meta.hf_repo:
            logger.error("No HF repo for model %s — nothing to download", model_id)
            return False

        self._cancel_event.clear()

        success = self._downloader.download(
            model_id=model_id,
            hf_repo=meta.hf_repo,
            progress_callback=progress_callback,
        )

        if success:
            local_path = self._find_hf_snapshot(meta.hf_repo)
            self._registry.update_entry(
                model_id,
                downloaded=True,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                local_path=str(local_path) if local_path else None,
            )

        return success

    def cancel(self) -> None:
        """Cancel any in-progress download within ~2 seconds."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Public API — plugin registration
    # ------------------------------------------------------------------

    def register_plugin(self, model_id: str, *, source_lang: str | None = None, target_lang: str | None = None) -> bool:
        """Generate or enable the plugin for a downloaded model."""
        meta = self._resolve_metadata(model_id)
        if meta is None:
            logger.error("Unknown model_id for plugin registration: %s", model_id)
            return False

        status = self._status_from_registry(model_id)
        if not status.downloaded:
            logger.error("Model %s not downloaded — cannot register plugin", model_id)
            return False

        try:
            if meta.category == "translation":
                success = self._register_translation_plugin(model_id, meta, source_lang=source_lang, target_lang=target_lang)
            else:
                success = self._register_ocr_plugin(model_id, meta)
        except Exception as exc:
            logger.error("Plugin registration failed for %s: %s", model_id, exc)
            return False

        if success:
            self._registry.update_entry(model_id, plugin_registered=True, enabled=True)

        return success

    def _register_translation_plugin(self, model_id: str, meta: ModelMetadata, *, source_lang: str | None = None, target_lang: str | None = None) -> bool:
        try:
            from app.workflow.universal_plugin_generator import PluginGenerator
        except ImportError:
            logger.error("PluginGenerator not available")
            return False

        family_prefix = meta.family.lower().replace("-", "")

        # Multilingual models (NLLB, M2M-100, mBART) use a single model for
        # all language pairs — the plugin should be language-agnostic.
        _MULTILINGUAL_FAMILIES = {"nllb200", "m2m100", "mbart"}
        is_multilingual = family_prefix in _MULTILINGUAL_FAMILIES

        if is_multilingual:
            plugin_name = family_prefix
        else:
            # MarianMT and similar: one model per language pair
            if source_lang and target_lang:
                src, tgt = source_lang, target_lang
            else:
                langs = meta.languages
                if len(langs) >= 2:
                    src, tgt = langs[0], langs[1]
                else:
                    src = tgt = langs[0] if langs else "xx"
            plugin_name = f"{family_prefix}_{src}_{tgt}"

        plugin_path = Path("plugins") / "stages" / "translation" / plugin_name
        if (plugin_path / "plugin.json").exists():
            logger.info("Translation plugin already exists: %s", plugin_name)
            return True

        if is_multilingual:
            display = f"{meta.family} (Multilingual)"
            desc = f"Translation plugin for {meta.family} — multilingual model"
        else:
            display = f"{meta.family} {src.upper()}\u2192{tgt.upper()}"
            desc = f"Translation plugin for {meta.family} ({src}\u2192{tgt})"

        generator = PluginGenerator(output_dir="plugins")
        settings: dict = {
            "model_name": {
                "type": "string",
                "default": meta.hf_repo or model_id,
                "description": "HuggingFace model name",
            },
            "max_length": {
                "type": "int",
                "default": 512,
                "description": "Maximum sequence length",
            },
        }
        if not is_multilingual:
            settings["source_language"] = {
                "type": "string",
                "default": src,
                "description": "Source language code",
            }
            settings["target_language"] = {
                "type": "string",
                "default": tgt,
                "description": "Target language code",
            }

        success = generator.create_plugin_programmatically(
            plugin_type="translation",
            name=plugin_name,
            display_name=display,
            description=desc,
            author="OptikR Auto-Generator",
            version="pre-realese-1.0.0",
            dependencies=["transformers", "torch", "sentencepiece"],
            settings=settings,
        )
        if not success:
            logger.error("PluginGenerator failed for %s", plugin_name)
        return success


    def _register_ocr_plugin(self, model_id: str, meta: ModelMetadata) -> bool:
        """Enable pre-built OCR plugin rather than generating one."""
        family_to_dir = {
            "EasyOCR": "easyocr",
            "Tesseract": "tesseract",
            "PaddleOCR": "paddleocr",
            "Mokuro": "mokuro",
        }
        dir_name = family_to_dir.get(meta.family)
        if not dir_name:
            logger.error("Unknown OCR family: %s", meta.family)
            return False

        plugin_dir = Path("plugins") / "stages" / "ocr" / dir_name
        if not plugin_dir.exists():
            logger.warning("Pre-built OCR plugin missing: %s", plugin_dir)
            return False

        logger.info("OCR plugin already present: %s", plugin_dir)
        return True

    # ------------------------------------------------------------------
    # Public API — deletion
    # ------------------------------------------------------------------

    def delete(self, model_id: str) -> bool:
        """Delete model files, deregister plugin, update registry."""
        status = self._status_from_registry(model_id)
        if not status.downloaded:
            logger.warning("Model %s is not downloaded — nothing to delete", model_id)
            return False

        if status.local_path:
            local = Path(status.local_path)
            if local.exists():
                try:
                    shutil.rmtree(local)
                except OSError as exc:
                    logger.error("Failed to remove %s: %s", local, exc)
                    return False

        meta = self._resolve_metadata(model_id)
        if meta and status.plugin_registered:
            self._try_delete_plugin(model_id, meta)

        self._registry.update_entry(
            model_id,
            downloaded=False,
            plugin_registered=False,
            enabled=False,
            local_path=None,
            download_timestamp=None,
        )
        return True

    def _try_delete_plugin(self, model_id: str, meta: ModelMetadata) -> None:
        if meta.category == "translation":
            family_prefix = meta.family.lower().replace("-", "")

            _MULTILINGUAL_FAMILIES = {"nllb200", "m2m100", "mbart"}
            if family_prefix in _MULTILINGUAL_FAMILIES:
                plugin_name = family_prefix
            else:
                langs = meta.languages
                if len(langs) >= 2:
                    src, tgt = langs[0], langs[1]
                else:
                    src = tgt = langs[0] if langs else "xx"
                plugin_name = f"{family_prefix}_{src}_{tgt}"

            plugin_path = Path("plugins") / "stages" / "translation" / plugin_name
            if plugin_path.exists():
                try:
                    shutil.rmtree(plugin_path)
                    logger.info("Deleted plugin %s", plugin_path)
                except OSError as exc:
                    logger.warning("Could not remove plugin %s: %s", plugin_path, exc)


    # ------------------------------------------------------------------
    # Public API — recommendations
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        gpu_available: bool,
        languages_needed: list[str],
    ) -> list[Recommendation]:
        """Return ranked recommendations based on hardware + language needs."""
        all_models = (
            self.list_available("translation")
            + self.list_available("ocr")
            + self.list_available("vision")
        )
        return self._recommender.recommend(all_models, gpu_available, languages_needed)

    # ------------------------------------------------------------------
    # Public API — cache
    # ------------------------------------------------------------------

    def get_cache_info(self) -> CacheInfo:
        """Return total models, disk usage, available space.

        Scans the HuggingFace cache for models that are tracked in the
        registry.
        """
        total_models = 0
        total_size = 0

        for raw in self._registry.models.values():
            if not raw.get("downloaded"):
                continue
            total_models += 1
            local = raw.get("local_path")
            if local:
                p = Path(local)
                if p.is_dir():
                    total_size += sum(
                        f.stat().st_size for f in p.rglob("*") if f.is_file()
                    )

        total_size_mb = total_size / (1024 * 1024)

        try:
            stat = shutil.disk_usage(self._hf_cache)
            available_gb = stat.free / (1024 ** 3)
        except OSError:
            available_gb = 0.0

        return CacheInfo(
            total_models=total_models,
            total_size_mb=round(total_size_mb, 2),
            available_space_gb=round(available_gb, 2),
        )

    def cleanup_cache(self, max_age_days: int = 30, max_size_gb: float = 10.0) -> int:
        """Remove old/oversized models.  Returns count of models removed."""
        now = datetime.now(timezone.utc)
        removed = 0

        entries_by_age = []
        for model_id, raw in list(self._registry.models.items()):
            if not raw.get("downloaded"):
                continue
            ts = raw.get("download_timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    entries_by_age.append((model_id, dt))
                except ValueError:
                    entries_by_age.append((model_id, now))
            else:
                entries_by_age.append((model_id, now))

        entries_by_age.sort(key=lambda x: x[1])

        for model_id, dt in entries_by_age:
            age_days = (now - dt).days
            if age_days > max_age_days:
                if self.delete(model_id):
                    removed += 1

        cache_info = self.get_cache_info()
        if cache_info.total_size_mb / 1024 > max_size_gb:
            for model_id, _ in entries_by_age:
                status = self.get_status(model_id)
                if not status.downloaded:
                    continue
                if self.delete(model_id):
                    removed += 1
                cache_info = self.get_cache_info()
                if cache_info.total_size_mb / 1024 <= max_size_gb:
                    break

        return removed

    # ------------------------------------------------------------------
    # Public API — import
    # ------------------------------------------------------------------

    def import_model(self, source_path: str) -> bool:
        """Import a local model directory into the catalog."""
        src = Path(source_path)
        if not src.is_dir():
            logger.error("Source directory not found: %s", source_path)
            return False

        from app.utils.path_utils import get_models_dir

        family = self._importer.detect_family(src)
        if not family:
            logger.error("Could not detect model family in %s", source_path)
            return False

        ok, err = self._importer.validate(src, family)
        if not ok:
            logger.error("Validation failed for %s: %s", source_path, err)
            return False

        category = self._importer.get_category(family)
        cache_dir = get_models_dir()

        try:
            dest = self._importer.copy_to_cache(src, family, cache_dir)
        except OSError as exc:
            logger.error("Copy failed for %s: %s", source_path, exc)
            return False

        model_id = f"imported-{src.name}"
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)

        self._registry.update_entry(
            model_id,
            family=family,
            category=category,
            downloaded=True,
            plugin_registered=False,
            enabled=False,
            local_path=str(dest),
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            size_mb=round(size_mb, 2),
            custom=True,
        )

        if self.register_plugin(model_id):
            logger.info("Imported and registered plugin for %s (%s)", model_id, family)
        else:
            logger.warning("Imported %s but plugin registration failed", model_id)

        return True

    # ------------------------------------------------------------------
    # Public API — custom models
    # ------------------------------------------------------------------

    def register_custom_model(self, model_entry: ModelEntry) -> None:
        """Register a custom ModelEntry at runtime (extensibility)."""
        self._custom_models[model_entry.model_id] = model_entry
        self._registry.update_entry(
            model_entry.model_id,
            family=model_entry.family,
            category=model_entry.category,
            hf_repo=model_entry.metadata.hf_repo,
            languages=model_entry.metadata.languages,
            size_mb=model_entry.metadata.size_mb,
            downloaded=model_entry.status.downloaded,
            plugin_registered=model_entry.status.plugin_registered,
            enabled=model_entry.status.enabled,
            custom=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metadata(self, model_id: str) -> ModelMetadata | None:
        if model_id in BUILTIN_MODELS:
            return BUILTIN_MODELS[model_id]
        custom = self._custom_models.get(model_id)
        if custom:
            return custom.metadata

        raw = self._registry.get_entry(model_id)
        if raw:
            return ModelMetadata(
                family=raw.get("family", "Unknown"),
                category=raw.get("category", "translation"),
                languages=raw.get("languages", []),
                size_mb=raw.get("size_mb", 0),
                speed="medium",
                quality="good",
                gpu_required=False,
                rationale="Imported model",
                hf_repo=raw.get("hf_repo"),
            )
        return None

    def _status_from_registry(self, model_id: str) -> ModelStatus:
        raw = self._registry.get_entry(model_id)
        if raw is None:
            return ModelStatus()
        return ModelStatus(
            downloaded=raw.get("downloaded", False),
            plugin_registered=raw.get("plugin_registered", False),
            enabled=raw.get("enabled", False),
            download_timestamp=raw.get("download_timestamp"),
            local_path=raw.get("local_path"),
        )

    def _find_hf_snapshot(self, hf_repo: str) -> Path | None:
        """Locate the most recent HF cache snapshot for *hf_repo*."""
        repo_dir_name = "models--" + hf_repo.replace("/", "--")
        snapshots = self._hf_cache / repo_dir_name / "snapshots"
        if not snapshots.is_dir():
            return None
        latest = max(
            (p for p in snapshots.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        return latest
