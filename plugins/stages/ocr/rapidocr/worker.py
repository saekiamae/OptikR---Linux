"""
RapidOCR Worker - Subprocess for text recognition.

Runs in a separate process using RapidOCR (ONNX Runtime backend).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    print("[RapidOCR WORKER] Warning: rapidocr-onnxruntime not available", file=sys.stderr)


class RapidOCRWorker(BaseWorker):
    """Worker for OCR using RapidOCR."""

    def initialize(self, config: dict) -> bool:
        try:
            if not RAPIDOCR_AVAILABLE:
                self.log("rapidocr-onnxruntime not available")
                return False

            language = config.get("language", "en")
            self.log(f"Initializing RapidOCR for language: {language}")

            self.engine = RapidOCR()

            self.log("RapidOCR initialized successfully")
            return True

        except Exception as e:
            self.log(f"Failed to initialize RapidOCR: {e}")
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
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb

            frame, error = decode_frame(data)
            if error:
                return {"error": error}

            self.log(f"Frame decoded: {frame.shape}, min={frame.min()}, max={frame.max()}")

            frame = bgr_to_rgb(frame)

            self.log("Starting OCR...")
            try:
                result, elapse = self.engine(frame)
                self.log(f"OCR complete: {len(result) if result else 0} results (elapse={elapse})")
            except Exception as ocr_error:
                self.log(f"RapidOCR error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                return {"text_blocks": [], "count": 0}

            text_blocks = []
            if result is not None:
                for bbox_points, text, confidence in result:
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)

                    text_blocks.append({
                        "text": text,
                        "bbox": [x, y, w, h],
                        "confidence": float(confidence),
                    })

            return {"text_blocks": text_blocks, "count": len(text_blocks)}

        except Exception as e:
            return {"error": f"OCR failed: {e}"}

    def cleanup(self):
        super().cleanup()
        if hasattr(self, "engine"):
            del self.engine
        self.log("RapidOCR worker shutdown")


if __name__ == "__main__":
    worker = RapidOCRWorker(name="RapidOCRWorker")
    worker.run()
