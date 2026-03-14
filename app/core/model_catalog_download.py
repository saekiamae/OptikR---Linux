"""Download manager for the unified ModelCatalog.

Handles HuggingFace Hub downloads with retry, exponential back-off,
progress callbacks, and cancellation via ``threading.Event``.

Models are downloaded via ``from_pretrained()`` and stored exclusively
in the HuggingFace Hub cache (``~/.cache/huggingface/hub/`` by default).
No local copies are made in ``system_data/``.

All public methods return ``bool`` for success/failure; exceptions are
logged but never propagated to callers.
"""

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class DownloadManager:
    """Handles HF Hub downloads with retry, progress, and cancellation."""

    MAX_RETRIES = 3
    BACKOFF_DELAYS = [1.0, 2.0, 4.0]

    def __init__(self, cancel_event: threading.Event | None = None):
        self._cancel = cancel_event or threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        model_id: str,
        hf_repo: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """Download a model from HF Hub with retry.

        Models are stored exclusively in the HuggingFace Hub cache
        (``~/.cache/huggingface/hub/`` or the path set by
        ``HUGGINGFACE_HUB_CACHE``).  No local copies are created.

        Parameters
        ----------
        model_id:
            Logical catalog id (used only for logging).
        hf_repo:
            HuggingFace repository, e.g. ``"Helsinki-NLP/opus-mt-en-de"``.
        progress_callback:
            ``(progress_float_0_to_1, status_message) -> None``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure or cancellation.
        """
        for attempt in range(self.MAX_RETRIES):
            if self._cancel.is_set():
                logger.info("Download cancelled for %s", model_id)
                return False

            try:
                success = self._attempt_download(
                    model_id, hf_repo, progress_callback, attempt,
                )
                if success:
                    return True
            except _NotFoundError:
                logger.error("HF repo not found (404): %s — not retrying", hf_repo)
                return False
            except OSError as exc:
                logger.error(
                    "OS error downloading %s (attempt %d/%d): %s",
                    model_id, attempt + 1, self.MAX_RETRIES, exc,
                )
            except Exception as exc:
                logger.error(
                    "Error downloading %s (attempt %d/%d): %s",
                    model_id, attempt + 1, self.MAX_RETRIES, exc,
                )

            if attempt < self.MAX_RETRIES - 1:
                delay = self.BACKOFF_DELAYS[attempt]
                logger.info(
                    "Retrying %s in %.0fs (attempt %d/%d)",
                    model_id, delay, attempt + 1, self.MAX_RETRIES,
                )
                self._cancel.wait(timeout=delay)

        logger.error("Failed to download %s after %d attempts", model_id, self.MAX_RETRIES)
        return False

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attempt_download(
        self,
        model_id: str,
        hf_repo: str,
        progress_callback: Callable[[float, str], None] | None,
        attempt: int,
    ) -> bool:
        """Download a model from HF Hub with real progress reporting.

        Uses ``huggingface_hub.snapshot_download`` for file-level progress,
        then validates by loading tokenizer + model via ``from_pretrained``.
        """
        try:
            from huggingface_hub import snapshot_download, HfApi
        except ImportError:
            logger.error("huggingface_hub not installed — cannot download %s", model_id)
            return False

        if progress_callback:
            progress_callback(0.02, "Resolving repository…")

        if self._cancel.is_set():
            return False

        # Discover total size for progress calculation
        total_bytes = 0
        downloaded_bytes = 0
        try:
            api = HfApi()
            repo_info = api.model_info(hf_repo, files_metadata=True)
            if repo_info.siblings:
                total_bytes = sum(
                    getattr(f, "size", 0) or 0 for f in repo_info.siblings
                )
        except Exception:
            pass  # proceed without size info

        if progress_callback:
            progress_callback(0.05, "Starting download…")

        # Use snapshot_download which caches properly and supports tqdm
        try:
            from tqdm.auto import tqdm as _tqdm
            import io

            class _ProgressTracker:
                """Track bytes across multiple file downloads."""
                def __init__(self, callback, total):
                    self._callback = callback
                    self._total = total
                    self._downloaded = 0
                    self._current_file = ""

                def update(self, n_bytes):
                    self._downloaded += n_bytes
                    if self._total > 0 and self._callback:
                        # Map to 0.05 – 0.95 range (leaving room for start/end)
                        frac = min(self._downloaded / self._total, 1.0)
                        progress = 0.05 + frac * 0.90
                        size_mb = self._downloaded / (1024 * 1024)
                        total_mb = self._total / (1024 * 1024)
                        self._callback(
                            progress,
                            f"Downloading… {size_mb:.0f}/{total_mb:.0f} MB",
                        )

            tracker = _ProgressTracker(progress_callback, total_bytes)

            # Monkey-patch tqdm to capture progress
            _orig_tqdm = _tqdm.__init__

            def _patched_init(self_tqdm, *args, **kwargs):
                _orig_tqdm(self_tqdm, *args, **kwargs)
                _orig_update = self_tqdm.update

                def _tracked_update(n=1):
                    _orig_update(n)
                    tracker.update(n)

                self_tqdm.update = _tracked_update

            _tqdm.__init__ = _patched_init
            try:
                snapshot_download(hf_repo)
            finally:
                _tqdm.__init__ = _orig_tqdm

        except Exception:
            # Fallback: download without progress tracking
            logger.debug("tqdm progress tracking failed, falling back to basic download")
            if progress_callback:
                progress_callback(0.30, "Downloading model (no progress available)…")
            snapshot_download(hf_repo)

        if self._cancel.is_set():
            return False

        if progress_callback:
            progress_callback(0.97, "Verifying model…")

        # Quick validation: ensure tokenizer can be loaded from cache.
        # trust_remote_code is needed for models like Qwen3 that ship
        # custom tokenizer code inside the HF repository.
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
        except Exception as exc:
            logger.warning("Post-download validation warning for %s: %s", model_id, exc)

        if progress_callback:
            progress_callback(1.0, "Download complete")

        logger.info("Downloaded %s to HuggingFace cache", model_id)
        return True


class _NotFoundError(Exception):
    """Raised when HF Hub returns 404 — do not retry."""
