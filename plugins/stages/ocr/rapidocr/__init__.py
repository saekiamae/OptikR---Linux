"""
RapidOCR Plugin - Main Entry Point

Lightweight OCR using RapidOCR (PaddleOCR models via ONNX Runtime).
CPU-optimized with no GPU dependency.
"""

import logging
from pathlib import Path

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import IOCREngine, OCRProcessingOptions, OCREngineType, OCREngineStatus
from app.models import Frame, TextBlock, Rectangle


class OCREngine(IOCREngine):
    """RapidOCR engine implementation."""

    _MAX_DIMENSION = 1280

    def __init__(self, engine_name: str = "rapidocr", engine_type=None):
        if engine_type is None:
            engine_type = OCREngineType.RAPIDOCR
        super().__init__(engine_name, engine_type)

        self.engine = None
        self.current_language = "en"
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: dict) -> bool:
        try:
            if not RAPIDOCR_AVAILABLE:
                self.logger.error("rapidocr-onnxruntime library not available")
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.INITIALIZING

            self.current_language = config.get("language", "en")

            self.logger.info(f"Initializing RapidOCR (language={self.current_language})")

            self.engine = _RapidOCR()

            self.status = OCREngineStatus.READY
            self.logger.info("RapidOCR initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize RapidOCR: {e}")
            self.status = OCREngineStatus.ERROR
            return False

    def extract_text(self, frame: Frame, options: OCRProcessingOptions) -> list[TextBlock]:
        if not self.is_ready():
            return []

        try:
            import cv2
            import numpy as np

            image = frame.data
            h, w = image.shape[:2]
            scale = 1.0

            if max(h, w) > self._MAX_DIMENSION:
                scale = self._MAX_DIMENSION / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            result, elapse = self.engine(image)

            text_blocks = []
            min_confidence = options.confidence_threshold if options else 0.5

            if result is not None:
                for line in result:
                    # Each line: [bbox_points, text, confidence]
                    bbox_points, text, confidence = line

                    if confidence < min_confidence:
                        continue

                    if not text or not text.strip():
                        continue

                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]

                    inv_scale = 1.0 / scale
                    x = int(min(xs) * inv_scale)
                    y = int(min(ys) * inv_scale)
                    width = int((max(xs) - min(xs)) * inv_scale)
                    height = int((max(ys) - min(ys)) * inv_scale)

                    text_block = TextBlock(
                        text=text,
                        position=Rectangle(x, y, width, height),
                        confidence=float(confidence),
                        language=options.language if options else self.current_language,
                    )
                    text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return []

    def extract_text_batch(self, frames: list[Frame], options: OCRProcessingOptions) -> list[list[TextBlock]]:
        results = []
        for frame in frames:
            results.append(self.extract_text(frame, options))
        return results

    def set_language(self, language: str) -> bool:
        try:
            if language != self.current_language:
                self.logger.info(f"Changing language from {self.current_language} to {language}")
                self.current_language = language
            return True
        except Exception as e:
            self.logger.error(f"Failed to set language: {e}")
            return False

    def get_supported_languages(self) -> list[str]:
        return ["en", "ja", "ko", "zh_sim", "zh_tra", "de", "fr", "es", "ru", "ar", "it", "pt", "nl", "tr", "pl", "vi"]

    def cleanup(self) -> None:
        self.engine = None
        self.status = OCREngineStatus.UNINITIALIZED
        self.logger.info("RapidOCR engine cleaned up")
