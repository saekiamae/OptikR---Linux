"""
Windows OCR Worker - Subprocess for text recognition.

Runs in a separate process, performs OCR via winocr and sends results back.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    from winocr import recognize_cv2_sync
    WINOCR_AVAILABLE = True
except ImportError:
    WINOCR_AVAILABLE = False
    print("[WinOCR WORKER] Warning: winocr not available", file=sys.stderr)

_ISO_TO_BCP47 = {
    'zh': 'zh-Hans',
    'zh_sim': 'zh-Hans',
    'zh_tra': 'zh-Hant',
    'sr': 'sr-Latn',
}


class OCRWorker(BaseWorker):
    """Worker for OCR using Windows.Media.Ocr (via winocr)."""

    def initialize(self, config: dict) -> bool:
        try:
            if not WINOCR_AVAILABLE:
                self.log("winocr not available")
                return False

            language = config.get('language', 'en')
            self._bcp47_lang = _ISO_TO_BCP47.get(language, language)

            self.log(
                f"Initializing Windows OCR for language: {language} "
                f"(bcp47: {self._bcp47_lang})"
            )

            from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
            from winrt.windows.globalization import Language
            if not WinOcrEngine.is_language_supported(Language(self._bcp47_lang)):
                self.log(
                    f"Language '{self._bcp47_lang}' not supported on this Windows build"
                )
                return False

            self.log("Windows OCR initialized successfully")
            return True

        except Exception as e:
            self.log(f"Failed to initialize Windows OCR: {e}")
            return False

    def process(self, data: dict) -> dict:
        """
        Perform OCR on a frame.

        Args:
            data: {'frame': base64_encoded_frame, 'shape': [...], 'dtype': str,
                   'language': str (optional)}

        Returns:
            {'text_blocks': [{'text': str, 'bbox': [x,y,w,h], 'confidence': float}],
             'count': int}
        """
        try:
            from app.workflow.base.ocr_frame_utils import decode_frame

            frame, error = decode_frame(data)
            if error:
                return {'error': error}

            self.log(
                f"Frame decoded: {frame.shape}, "
                f"min={frame.min()}, max={frame.max()}"
            )

            lang = data.get('language')
            if lang:
                bcp47 = _ISO_TO_BCP47.get(lang, lang)
            else:
                bcp47 = self._bcp47_lang

            self.log("Starting Windows OCR...")
            try:
                result = recognize_cv2_sync(frame, bcp47)
                lines = result.get('lines', [])
                self.log(f"OCR complete: {len(lines)} lines")
            except Exception as ocr_error:
                self.log(f"winocr error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                return {'text_blocks': [], 'count': 0}

            text_blocks = []
            for line in lines:
                line_text = line.get('text', '').strip()
                if not line_text:
                    continue

                words = line.get('words', [])
                if not words:
                    continue

                x_min = float('inf')
                y_min = float('inf')
                x_max = float('-inf')
                y_max = float('-inf')
                for word in words:
                    br = word.get('bounding_rect', {})
                    wx = br.get('x', 0)
                    wy = br.get('y', 0)
                    ww = br.get('width', 0)
                    wh = br.get('height', 0)
                    x_min = min(x_min, wx)
                    y_min = min(y_min, wy)
                    x_max = max(x_max, wx + ww)
                    y_max = max(y_max, wy + wh)

                x = int(x_min)
                y = int(y_min)
                w = int(x_max - x_min)
                h = int(y_max - y_min)

                text_blocks.append({
                    'text': line_text,
                    'bbox': [x, y, w, h],
                    'confidence': 1.0,
                })

            return {
                'text_blocks': text_blocks,
                'count': len(text_blocks),
            }

        except Exception as e:
            return {'error': f'Windows OCR failed: {e}'}

    def cleanup(self):
        super().cleanup()
        self.log("Windows OCR worker shutdown")


if __name__ == '__main__':
    worker = OCRWorker(name="WinOCRWorker")
    worker.run()
