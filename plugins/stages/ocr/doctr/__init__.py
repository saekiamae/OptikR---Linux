"""
DocTR Plugin - Main Entry Point

Transformer-based OCR using DocTR (Mindee).
Supports detection + recognition with confidence scores and GPU acceleration.
"""

import logging
from pathlib import Path

try:
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import IOCREngine, OCRProcessingOptions, OCREngineType, OCREngineStatus
from app.models import Frame, TextBlock, Rectangle


class OCREngine(IOCREngine):
    """DocTR engine implementation."""

    _MAX_DIMENSION = 1280

    def __init__(self, engine_name: str = "doctr", engine_type=None):
        if engine_type is None:
            engine_type = OCREngineType.DOCTR
        super().__init__(engine_name, engine_type)

        self.predictor = None
        self.current_language = "en"
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: dict) -> bool:
        try:
            if not DOCTR_AVAILABLE:
                self.logger.error("python-doctr library not available")
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.INITIALIZING

            self.current_language = config.get("language", "en")
            use_gpu = config.get("gpu", True)
            det_arch = config.get("det_arch", "db_resnet50")
            reco_arch = config.get("reco_arch", "crnn_vgg16_bn")

            self.logger.info(
                f"Initializing DocTR (language={self.current_language}, "
                f"det={det_arch}, reco={reco_arch}, gpu={use_gpu})"
            )

            self.predictor = ocr_predictor(
                det_arch=det_arch,
                reco_arch=reco_arch,
                pretrained=True,
            )

            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.predictor = self.predictor.cuda()
                        self.logger.info("DocTR using CUDA GPU")
                    else:
                        self.logger.info("CUDA not available, DocTR falling back to CPU")
                except ImportError:
                    self.logger.info("torch not available for GPU check, using CPU")

            self.status = OCREngineStatus.READY
            self.logger.info("DocTR initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize DocTR: {e}")
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

            # DocTR expects RGB; convert from BGR if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            scaled_h, scaled_w = image.shape[:2]

            result = self.predictor([image])

            text_blocks = []
            min_confidence = options.confidence_threshold if options else 0.5
            inv_scale = 1.0 / scale

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if word.confidence < min_confidence:
                                continue

                            if not word.value or not word.value.strip():
                                continue

                            # DocTR geometry is relative (0-1), convert to pixel coords
                            (x_min, y_min), (x_max, y_max) = word.geometry
                            px_x = int(x_min * scaled_w * inv_scale)
                            px_y = int(y_min * scaled_h * inv_scale)
                            px_w = int((x_max - x_min) * scaled_w * inv_scale)
                            px_h = int((y_max - y_min) * scaled_h * inv_scale)

                            text_block = TextBlock(
                                text=word.value,
                                position=Rectangle(px_x, px_y, px_w, px_h),
                                confidence=float(word.confidence),
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
        return [
            "en", "fr", "de", "es", "pt", "it", "nl", "pl", "ru", "ar",
            "zh", "ja", "ko", "vi", "hi", "cs", "ro", "hu", "sv", "da",
        ]

    def cleanup(self) -> None:
        self.predictor = None
        self.status = OCREngineStatus.UNINITIALIZED

        try:
            from app.utils.pytorch_manager import release_gpu_memory
            release_gpu_memory()
        except ImportError:
            pass

        self.logger.info("DocTR engine cleaned up")
