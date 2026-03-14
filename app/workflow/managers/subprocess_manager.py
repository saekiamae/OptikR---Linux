"""
SubprocessManager -- orchestrator that owns all subprocess instances.

SubprocessStrategy reads the ``ocr_subprocess`` (and future
``capture_subprocess`` / ``translation_subprocess``) attributes from
this class to route pipeline stages to isolated child processes.

Only OCR is supported initially; the other attributes remain ``None``
so SubprocessStrategy falls back to in-process execution for those.
"""

import logging
import os
from typing import Any

from app.workflow.subprocesses.ocr_subprocess import OCRSubprocess


logger = logging.getLogger(__name__)


class SubprocessManager:
    """Manages lifecycle of all pipeline subprocesses."""

    def __init__(self, config_manager: Any = None) -> None:
        self._config_manager = config_manager
        self._started = False

        self.ocr_subprocess: OCRSubprocess | None = None
        self.capture_subprocess = None
        self.translation_subprocess = None

    # -- public API ----------------------------------------------------

    def start(self, ocr_plugin_name: str, ocr_plugin_path: str) -> bool:
        """Start the OCR subprocess.

        Args:
            ocr_plugin_name: Human-readable plugin name (e.g. ``"easyocr"``).
            ocr_plugin_path: Absolute path to the plugin directory that
                contains ``worker.py``.

        Returns:
            ``True`` if the subprocess started and reported ready.
        """
        worker_script = os.path.join(ocr_plugin_path, "worker.py")
        if not os.path.exists(worker_script):
            logger.warning(
                "No worker.py found for %s at %s", ocr_plugin_name, worker_script,
            )
            return False

        self.ocr_subprocess = OCRSubprocess(ocr_plugin_name, worker_script)
        config = self._build_ocr_config()

        logger.info(
            "Starting OCR subprocess for %s (worker=%s)",
            ocr_plugin_name, worker_script,
        )
        success = self.ocr_subprocess.start(config)
        self._started = success

        if not success:
            logger.error("OCR subprocess failed to start for %s", ocr_plugin_name)
            self.ocr_subprocess = None

        return success

    def stop(self) -> None:
        """Stop all managed subprocesses."""
        if self.ocr_subprocess is not None:
            self.ocr_subprocess.stop()
            self.ocr_subprocess = None
        self._started = False
        logger.info("SubprocessManager stopped")

    def is_healthy(self) -> bool:
        """Return ``True`` if all started subprocesses are alive."""
        if not self._started:
            return False
        if self.ocr_subprocess is not None and not self.ocr_subprocess.is_alive():
            return False
        return True

    # -- internals -----------------------------------------------------

    def _build_ocr_config(self) -> dict:
        """Build the init config dict sent to the OCR worker."""
        config: dict[str, Any] = {}
        if self._config_manager is not None:
            config["gpu"] = self._config_manager.get_setting(
                "performance.enable_gpu_acceleration", True,
            )
            config["language"] = self._config_manager.get_setting(
                "translation.source_language", "en",
            )
        return config
