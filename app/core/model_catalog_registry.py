"""Registry manager for the unified ModelCatalog.

Owns the single ``model_registry.json`` file that tracks download state,
plugin registration, and per-model metadata.  Scans the HuggingFace Hub
cache to detect already-downloaded models.

All writes are protected by a threading.Lock and are atomic (write to a
temporary file, then rename) so the registry is never left in a corrupt
state.
"""

import json
import logging
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WEIGHTS_GLOBS = (
    "pytorch_model.bin",
    "pytorch_model*.bin",
    "model.safetensors",
    "model*.safetensors",
    "model.pth",
    "*.pth",
)


class RegistryManager:
    """Manages the single JSON registry file and HF cache scanning."""

    _REGISTRY_VERSION = 1

    def __init__(
        self,
        registry_path: Path,
        hf_cache_dir: Path | None = None,
    ):
        self._path = registry_path
        self._hf_cache_dir = hf_cache_dir or self._default_hf_cache()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    @staticmethod
    def _default_hf_cache() -> Path:
        import os
        hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if hub_cache:
            return Path(hub_cache)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Load registry from disk, recovering from corruption."""
        if not self._path.exists():
            return self._empty_registry()
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data.get("models"), dict):
                raise ValueError("missing or invalid 'models' key")
            return data
        except Exception as exc:
            logger.warning("Registry corrupted (%s), backing up and resetting", exc)
            try:
                backup = self._path.with_suffix(".json.bak")
                shutil.copy2(self._path, backup)
            except OSError:
                pass
            return self._empty_registry()

    def _empty_registry(self) -> dict[str, Any]:
        return {
            "version": self._REGISTRY_VERSION,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "models": {},
        }

    def save(self) -> None:
        """Persist registry to disk atomically (thread-safe)."""
        with self._lock:
            self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd, tmp = tempfile.mkstemp(
                    dir=str(self._path.parent), suffix=".tmp"
                )
                try:
                    import os
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(self._data, f, indent=2, ensure_ascii=False)
                    tmp_path = Path(tmp)
                    tmp_path.replace(self._path)
                except BaseException:
                    Path(tmp).unlink(missing_ok=True)
                    raise
            except OSError as exc:
                logger.error("Failed to save registry: %s", exc)
                raise

    def load(self) -> dict[str, Any]:
        """Return a shallow copy of the current registry data."""
        return dict(self._data)

    # ------------------------------------------------------------------
    # Entry CRUD
    # ------------------------------------------------------------------

    def get_entry(self, model_id: str) -> dict[str, Any] | None:
        return self._data["models"].get(model_id)

    def update_entry(self, model_id: str, **fields: Any) -> None:
        """Update (or create) fields on a registry entry and persist."""
        entry = self._data["models"].setdefault(model_id, {})
        entry.update(fields)
        self.save()

    def remove_entry(self, model_id: str) -> None:
        self._data["models"].pop(model_id, None)
        self.save()

    @property
    def models(self) -> dict[str, Any]:
        return self._data.get("models", {})

    # ------------------------------------------------------------------
    # Local scan
    # ------------------------------------------------------------------

    def scan_local_models(self) -> None:
        """Scan the HuggingFace cache and update ``downloaded`` status."""
        if self._hf_cache_dir.is_dir():
            self._scan_hf_cache()

        self.save()

    def _scan_hf_cache(self) -> None:
        """Look inside the HF Hub snapshot cache for known repos."""
        try:
            for repo_dir in self._hf_cache_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                snapshots = repo_dir / "snapshots"
                if not snapshots.is_dir():
                    continue
                for snap in snapshots.iterdir():
                    if snap.is_dir() and self.verify_completeness(snap):
                        repo_name = repo_dir.name.replace("models--", "").replace("--", "/")
                        for mid, entry in self._data["models"].items():
                            if entry.get("hf_repo") == repo_name:
                                entry["downloaded"] = True
                                if not entry.get("local_path"):
                                    entry["local_path"] = str(snap)
        except OSError as exc:
            logger.debug("HF cache scan skipped: %s", exc)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_completeness(model_path: Path) -> bool:
        """Return True if *model_path* contains config + at least one weights file.

        Accepts config.json or preprocessor_config.json (vision models often use both).
        """
        has_config = (
            (model_path / "config.json").exists()
            or (model_path / "preprocessor_config.json").exists()
        )
        if not has_config:
            return False
        for pattern in _WEIGHTS_GLOBS:
            if any(model_path.glob(pattern)):
                return True
        return False
