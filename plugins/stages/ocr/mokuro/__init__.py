"""
Mokuro OCR Plugin

Manga page OCR with text detection and bounding boxes.
Mokuro detects individual text regions using comic_text_detector
and returns their positions with bounding boxes.
"""

import logging
import time
from pathlib import Path
from typing import Any

MangaPageOcr = None
ENGINE_AVAILABLE: bool | None = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import IOCREngine, OCRProcessingOptions, OCREngineType, OCREngineStatus
from app.models import Frame, TextBlock, Rectangle


class OCREngine(IOCREngine):
    """Mokuro OCR engine — Japanese manga page OCR with bounding boxes."""

    def __init__(self, engine_name: str = "mokuro", engine_type=None):
        if engine_type is None:
            engine_type = OCREngineType.MOKURO
        super().__init__(engine_name, engine_type)

        self.mpocr = None
        self.current_language = "ja"
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: dict[str, Any]) -> bool:
        """Initialize Mokuro with GPU->CPU fallback."""
        try:
            global MangaPageOcr, ENGINE_AVAILABLE

            if ENGINE_AVAILABLE is None:
                try:
                    from mokuro.manga_page_ocr import MangaPageOcr as _MangaPageOcr
                    MangaPageOcr = _MangaPageOcr
                    ENGINE_AVAILABLE = True
                except ImportError:
                    ENGINE_AVAILABLE = False

            if not ENGINE_AVAILABLE:
                self.logger.error("mokuro library not available")
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.INITIALIZING

            use_gpu = config.get("gpu", True) or config.get("use_gpu", True)

            import torch
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            try:
                self.logger.info("Initializing Mokuro OCR (device=%s)…", device)
                self.mpocr = MangaPageOcr(force_cpu=(device == "cpu"))
                self.capabilities.has_text_detection = True
                self.status = OCREngineStatus.READY
                self.logger.info("Mokuro OCR initialized successfully (%s)", device)
                return True
            except Exception as e:
                self.logger.warning("Mokuro init failed on %s: %s", device, e)
                if device != "cpu":
                    try:
                        self.logger.info("Falling back to CPU…")
                        self.mpocr = MangaPageOcr(force_cpu=True)
                        self.capabilities.has_text_detection = True
                        self.status = OCREngineStatus.READY
                        self.logger.info("Mokuro OCR initialized successfully (CPU fallback)")
                        return True
                    except Exception as e2:
                        self.logger.error("Mokuro CPU fallback also failed: %s", e2)

            self.logger.error("Mokuro OCR: all init attempts failed")
            self.status = OCREngineStatus.ERROR
            return False

        except Exception as e:
            self.logger.error("Failed to initialize Mokuro: %s", e)
            import traceback
            self.logger.error(traceback.format_exc())
            self.status = OCREngineStatus.ERROR
            return False

    def extract_text(self, frame: Frame, options: OCRProcessingOptions) -> list[TextBlock]:
        """Extract text regions with bounding boxes from a manga page."""
        if not self.is_ready():
            return []

        try:
            import numpy as np
            import tempfile
            import cv2

            if not isinstance(frame.data, np.ndarray):
                self.logger.error("Frame data is not a numpy array")
                return []

            start_ms = self._record_processing_start()

            # Mokuro expects a file path, not a PIL Image.
            # Write the frame to a temporary file and pass the path.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame.data)

            try:
                result = self.mpocr(tmp_path)
            finally:
                import os
                os.unlink(tmp_path)

            text_blocks: list[TextBlock] = []
            img_h, img_w = frame.data.shape[:2]

            blocks = result.get("blocks", []) if isinstance(result, dict) else []
            for block in blocks:
                lines = block.get("lines", [])
                text = "".join(lines).strip()
                if not text:
                    continue

                box = block.get("box", [0, 0, img_w, img_h])
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(img_w, int(box[2]))
                y2 = min(img_h, int(box[3]))

                position = Rectangle(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
                text_blocks.append(TextBlock(
                    text=text,
                    position=position,
                    confidence=0.90,
                    language="ja",
                ))

            self._record_processing_end(start_ms, success=True)
            self.logger.info("Mokuro extracted %d text block(s)", len(text_blocks))
            return text_blocks

        except Exception as e:
            self.logger.error("Mokuro OCR processing failed: %s", e)
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def extract_text_batch(self, frames: list[Frame], options: OCRProcessingOptions) -> list[list[TextBlock]]:
        """Extract text from multiple frames sequentially."""
        return [self.extract_text(frame, options) for frame in frames]

    def set_language(self, language: str) -> bool:
        """Set OCR language (Mokuro only supports Japanese)."""
        if language != "ja":
            self.logger.warning("Mokuro only supports Japanese, ignoring language: %s", language)
        return True

    def get_supported_languages(self) -> list[str]:
        return ["ja"]

    def cleanup(self) -> None:
        """Release resources and GPU memory."""
        self.mpocr = None
        self.status = OCREngineStatus.UNINITIALIZED

        from app.utils.pytorch_manager import release_gpu_memory
        release_gpu_memory()

        self.logger.info("Mokuro OCR engine cleaned up")
