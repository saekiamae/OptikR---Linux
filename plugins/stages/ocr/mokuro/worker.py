"""
Mokuro - Manga page OCR with text detection and bounding boxes

OCR plugin worker script.
"""

import sys
from pathlib import Path
import warnings

# Silence deprecation warning from comic_text_detector/pkg_resources so the log
# isn't spammed on startup. This does not affect functionality.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.",
    category=UserWarning,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker


class OCRWorker(BaseWorker):
    """Worker for Mokuro OCR."""

    def initialize(self, config: dict) -> bool:
        """
        Initialize Mokuro OCR engine.

        Args:
            config: Configuration dictionary with 'language' key

        Returns:
            True if successful
        """
        try:
            from mokuro.manga_page_ocr import MangaPageOcr

            self.log("Initializing Mokuro OCR (Japanese only)...")
            self.mpocr = MangaPageOcr(force_cpu=True)
            self.log("Mokuro OCR initialized successfully")
            return True

        except Exception as e:
            import traceback
            self.log(f"Failed to initialize Mokuro: {e}\n{traceback.format_exc()}")
            return False

    def process(self, data: dict) -> dict:
        """
        Perform OCR on frame using Mokuro.

        Mokuro detects individual text regions and returns bounding boxes.

        Args:
            data: {
                'frame': base64_string,
                'shape': [h, w, c],
                'dtype': 'uint8',
                'language': str
            }

        Returns:
            {
                'text_blocks': [
                    {'text': str, 'bbox': [x, y, w, h], 'confidence': float}
                ],
                'count': int
            }
        """
        try:
            import cv2
            import tempfile
            import os
            from app.workflow.base.ocr_frame_utils import decode_frame

            frame, error = decode_frame(data)
            if error:
                return {"error": error}

            # Mokuro expects a file path, not a PIL Image.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)

            try:
                result = self.mpocr(tmp_path)
            finally:
                os.unlink(tmp_path)

            text_blocks = []
            img_h, img_w = frame.shape[:2]
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

                text_blocks.append({
                    "text": text,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": 0.90,
                })

            return {
                "text_blocks": text_blocks,
                "count": len(text_blocks),
            }

        except Exception as e:
            import traceback
            error_msg = f"OCR failed: {e}\n{traceback.format_exc()}"
            self.log(error_msg)
            return {"error": error_msg}

    def cleanup(self):
        """Clean up resources."""
        self.mpocr = None
        self.log("Mokuro OCR cleanup complete")


if __name__ == "__main__":
    worker = OCRWorker(name="mokuro")
    worker.run()
