"""
Surya OCR Worker - Subprocess for text recognition.

Runs in a separate process using surya-ocr (vision-transformer OCR).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False
    print("[Surya OCR WORKER] Warning: surya-ocr not available", file=sys.stderr)


class SuryaOCRWorker(BaseWorker):
    """Worker for OCR using Surya."""

    def initialize(self, config: dict) -> bool:
        try:
            if not SURYA_AVAILABLE:
                self.log("surya-ocr not available")
                return False

            language = config.get("language", "en")
            use_gpu = config.get("gpu", True)

            self.log(f"Initializing Surya OCR (language={language}, gpu={use_gpu})")

            if not use_gpu:
                import os
                os.environ["TORCH_DEVICE"] = "cpu"

            from surya.foundation import FoundationPredictor
            foundation = FoundationPredictor()
            self.recognition_predictor = RecognitionPredictor(foundation)
            self.detection_predictor = DetectionPredictor()

            self.log("Surya OCR initialized successfully")
            return True

        except Exception as e:
            self.log(f"Failed to initialize Surya OCR: {e}")
            return False

    def process(self, data: dict) -> dict:
        """
        Perform OCR on frame.

        Args:
            data: {'frame': base64_encoded_frame, 'shape': list, 'dtype': str, 'language': str}

        Returns:
            {'text_blocks': [{'text': str, 'bbox': [x,y,w,h], 'confidence': float}], 'count': int}
        """
        try:
            from PIL import Image as PILImage
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb

            frame, error = decode_frame(data)
            if error:
                return {"error": error}

            self.log(f"Frame decoded: {frame.shape}, min={frame.min()}, max={frame.max()}")

            frame = bgr_to_rgb(frame)

            pil_image = PILImage.fromarray(frame)

            self.log("Starting OCR...")
            try:
                predictions = self.recognition_predictor(
                    [pil_image], det_predictor=self.detection_predictor
                )
                pred = predictions[0]
                line_count = len(pred.text_lines)
                self.log(f"OCR complete: {line_count} lines detected")
            except Exception as ocr_error:
                self.log(f"Surya OCR error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                return {"text_blocks": [], "count": 0}

            text_blocks = []
            for line in pred.text_lines:
                x1, y1, x2, y2 = line.bbox
                px_x = int(x1)
                px_y = int(y1)
                px_w = int(x2 - x1)
                px_h = int(y2 - y1)

                text_blocks.append({
                    "text": line.text,
                    "bbox": [px_x, px_y, px_w, px_h],
                    "confidence": float(line.confidence),
                })

            return {"text_blocks": text_blocks, "count": len(text_blocks)}

        except Exception as e:
            return {"error": f"OCR failed: {e}"}

    def cleanup(self):
        super().cleanup()
        if hasattr(self, "recognition_predictor"):
            del self.recognition_predictor
        if hasattr(self, "detection_predictor"):
            del self.detection_predictor
        self.log("Surya OCR worker shutdown")


if __name__ == "__main__":
    worker = SuryaOCRWorker(name="SuryaOCRWorker")
    worker.run()
