"""
DocTR Worker - Subprocess for text recognition.

Runs in a separate process using DocTR (Mindee transformer-based OCR).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    print("[DocTR WORKER] Warning: python-doctr not available", file=sys.stderr)


class DocTRWorker(BaseWorker):
    """Worker for OCR using DocTR."""

    def initialize(self, config: dict) -> bool:
        try:
            if not DOCTR_AVAILABLE:
                self.log("python-doctr not available")
                return False

            language = config.get("language", "en")
            use_gpu = config.get("gpu", True)
            det_arch = config.get("det_arch", "db_resnet50")
            reco_arch = config.get("reco_arch", "crnn_vgg16_bn")

            self.log(f"Initializing DocTR (language={language}, det={det_arch}, reco={reco_arch})")

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
                        self.log("DocTR using CUDA GPU")
                except ImportError:
                    pass

            self.log("DocTR initialized successfully")
            return True

        except Exception as e:
            self.log(f"Failed to initialize DocTR: {e}")
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
            import cv2
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb

            frame, error = decode_frame(data)
            if error:
                return {"error": error}

            self.log(f"Frame decoded: {frame.shape}, min={frame.min()}, max={frame.max()}")

            frame = bgr_to_rgb(frame)
            h, w = frame.shape[:2]

            self.log("Starting OCR...")
            try:
                result = self.predictor([frame])
                page = result.pages[0]
                word_count = sum(
                    len(line.words)
                    for block in page.blocks
                    for line in block.lines
                )
                self.log(f"OCR complete: {word_count} words detected")
            except Exception as ocr_error:
                self.log(f"DocTR error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                return {"text_blocks": [], "count": 0}

            text_blocks = []
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (x_min, y_min), (x_max, y_max) = word.geometry
                        px_x = int(x_min * w)
                        px_y = int(y_min * h)
                        px_w = int((x_max - x_min) * w)
                        px_h = int((y_max - y_min) * h)

                        text_blocks.append({
                            "text": word.value,
                            "bbox": [px_x, px_y, px_w, px_h],
                            "confidence": float(word.confidence),
                        })

            return {"text_blocks": text_blocks, "count": len(text_blocks)}

        except Exception as e:
            return {"error": f"OCR failed: {e}"}

    def cleanup(self):
        super().cleanup()
        if hasattr(self, "predictor"):
            del self.predictor
        self.log("DocTR worker shutdown")


if __name__ == "__main__":
    worker = DocTRWorker(name="DocTRWorker")
    worker.run()
