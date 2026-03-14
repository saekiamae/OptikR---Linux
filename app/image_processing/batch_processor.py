"""Batch image processor.

:class:`QThread`-based worker that processes multiple images through
the :class:`ImagePipeline` and saves translated images to disk.  Emits
progress signals so the UI can update a progress bar and results panel.

Error handling is per-image: a failure on one file does not abort the
entire batch.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from .image_compositor import ImageCompositor
from .image_pipeline import ImagePipeline

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)


class BatchProcessor(QThread):
    """Processes a batch of images through OCR, translation, and compositing.

    Signals
    -------
    progress(current, total, filename, status)
        Emitted before each image is processed.
    image_completed(filepath, success, error_msg)
        Emitted after each image finishes (or fails).
    batch_completed(total, succeeded, failed)
        Emitted once when the entire batch is done or cancelled.
    """

    progress = pyqtSignal(int, int, str, str)
    image_completed = pyqtSignal(str, bool, str)
    batch_completed = pyqtSignal(int, int, int)

    def __init__(
        self,
        pipeline: ImagePipeline,
        compositor: ImageCompositor,
        config_manager: Any = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._compositor = compositor
        self._config_manager = config_manager

        self._files: list[str] = []
        self._output_config: dict[str, Any] = {}
        self._compositor_config: dict[str, Any] = {}
        self._source_lang: str | None = None
        self._target_lang: str | None = None
        self._cancel_requested = False

    # ------------------------------------------------------------------
    # Configuration setters (call before start())
    # ------------------------------------------------------------------

    def set_files(self, files: list[str]) -> None:
        """Set the list of image file paths to process."""
        self._files = list(files)

    def set_output_config(self, config: dict[str, Any]) -> None:
        """Set output options (folder, naming, format, quality)."""
        self._output_config = dict(config)

    def set_compositor_config(self, config: dict[str, Any]) -> None:
        """Set style overrides forwarded to the compositor."""
        self._compositor_config = dict(config)

    def set_languages(self, source: str, target: str) -> None:
        """Override source/target language for this batch."""
        self._source_lang = source
        self._target_lang = target

    def stop(self) -> None:
        """Request cancellation.  The current image will finish but no
        further images will be started."""
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D401 — Qt override
        """Process all queued images (runs on the worker thread)."""
        self._cancel_requested = False
        total = len(self._files)
        succeeded = 0
        failed = 0

        for idx, filepath in enumerate(self._files):
            if self._cancel_requested:
                logger.info(
                    "Batch cancelled at %d/%d", idx, total,
                )
                break

            filename = Path(filepath).name
            self.progress.emit(idx + 1, total, filename, "processing")

            try:
                ok, error = self._process_single(filepath)
                if ok:
                    succeeded += 1
                    self.image_completed.emit(filepath, True, "")
                else:
                    failed += 1
                    self.image_completed.emit(filepath, False, error)
            except Exception as exc:
                failed += 1
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error("Failed to process %s: %s", filepath, error_msg)
                self.image_completed.emit(filepath, False, error_msg)

        self.batch_completed.emit(total, succeeded, failed)

    # ------------------------------------------------------------------
    # Single-image processing
    # ------------------------------------------------------------------

    def _process_single(self, filepath: str) -> tuple[bool, str]:
        """Load, translate, and save a single image.

        Returns ``(success, error_message)``.
        """
        # --- Load ---
        try:
            pil_image = Image.open(filepath)
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")
            elif pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
            image_array = np.array(pil_image)
        except Exception as exc:
            return False, f"Failed to load image: {exc}"

        # PIL loads as RGB; pipeline uses BGR (OpenCV convention)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array

        # --- Pipeline (OCR + translation + compositing) ---
        result = self._pipeline.process_image(
            image_bgr,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            compositor_config=self._compositor_config,
        )

        if not result["success"]:
            return False, result.get("error", "Pipeline processing failed")

        composited = result["image"]

        # --- Save ---
        try:
            output_path = self._resolve_output_path(filepath)
            self._save_image(composited, output_path, self._output_config)
            return True, ""
        except Exception as exc:
            return False, f"Failed to save: {exc}"

    # ------------------------------------------------------------------
    # Output path resolution
    # ------------------------------------------------------------------

    def _resolve_output_path(self, input_path: str) -> str:
        """Determine the output file path based on naming config."""
        src = Path(input_path)
        output_folder = self._output_config.get("output_folder", "")
        if not output_folder:
            output_folder = str(src.parent)

        naming = self._output_config.get("naming_pattern", "suffix")
        suffix_text = self._output_config.get("naming_suffix", "_translated")

        fmt = self._output_config.get("output_format", "same")
        ext = src.suffix if fmt == "same" else f".{fmt}"
        stem = src.stem

        if naming == "prefix":
            prefix_text = suffix_text.lstrip("_") or "translated"
            return str(Path(output_folder) / f"{prefix_text}_{stem}{ext}")

        if naming == "subfolder":
            subfolder = Path(output_folder) / "translated"
            subfolder.mkdir(parents=True, exist_ok=True)
            return str(subfolder / f"{stem}{ext}")

        # Default: suffix
        return str(Path(output_folder) / f"{stem}{suffix_text}{ext}")

    # ------------------------------------------------------------------
    # Image saving
    # ------------------------------------------------------------------

    @staticmethod
    def _save_image(
        image: np.ndarray,
        output_path: str,
        output_config: dict[str, Any],
    ) -> None:
        """Write *image* (BGR) to *output_path* with format-specific quality."""
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(output_path).suffix.lower()
        jpg_quality = output_config.get("jpg_quality", 95)

        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(
                output_path, image,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
            )
        elif ext == ".png":
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif ext == ".webp":
            cv2.imwrite(output_path, image, [cv2.IMWRITE_WEBP_QUALITY, 95])
        else:
            cv2.imwrite(output_path, image)

        logger.info("Saved translated image: %s", output_path)
