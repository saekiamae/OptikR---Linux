"""
Windows OCR Plugin - Main Entry Point

Uses the winocr package (Windows.Media.Ocr / DirectML) for fast,
GPU-accelerated OCR on Windows.  Typically ~10-30 ms per frame.
"""

import logging
from pathlib import Path

try:
    from winocr import recognize_cv2_sync
    WINOCR_AVAILABLE = True
except ImportError:
    WINOCR_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import IOCREngine, OCRProcessingOptions
from app.models import Frame, TextBlock, Rectangle

# ISO 639-1 -> BCP-47 tag expected by winocr's Language() constructor.
# Most ISO codes work as-is; Chinese variants need explicit script subtags.
_ISO_TO_BCP47 = {
    'zh': 'zh-Hans',
    'zh_sim': 'zh-Hans',
    'zh_tra': 'zh-Hant',
    'sr': 'sr-Latn',
}


class OCREngine(IOCREngine):
    """Windows OCR engine implementation via winocr."""

    _MAX_DIMENSION = 1280

    def __init__(self, engine_name: str = "windows_ocr", engine_type=None):
        from app.ocr.ocr_engine_interface import OCREngineType, OCREngineStatus

        if engine_type is None:
            engine_type = OCREngineType.WINDOWS_OCR
        super().__init__(engine_name, engine_type)

        self.current_language = 'en'
        self._bcp47_lang = 'en'
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: dict) -> bool:
        try:
            from app.ocr.ocr_engine_interface import OCREngineStatus

            if not WINOCR_AVAILABLE:
                self.logger.error("winocr library not available")
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.INITIALIZING

            lang = config.get('language', 'en')
            self._set_bcp47(lang)

            self.logger.info(
                f"Initializing Windows OCR (language={self.current_language}, "
                f"bcp47={self._bcp47_lang})"
            )

            # Verify the language is actually installed on this Windows build
            from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
            from winrt.windows.globalization import Language
            if not WinOcrEngine.is_language_supported(Language(self._bcp47_lang)):
                self.logger.error(
                    f"Windows does not have OCR support for '{self._bcp47_lang}'. "
                    f"Install the language pack via Settings > Language."
                )
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.READY
            self.logger.info("Windows OCR initialized successfully")
            return True

        except Exception as e:
            from app.ocr.ocr_engine_interface import OCREngineStatus
            self.logger.error(f"Failed to initialize Windows OCR: {e}")
            self.status = OCREngineStatus.ERROR
            return False

    def extract_text(self, frame: Frame, options: OCRProcessingOptions) -> list[TextBlock]:
        if not self.is_ready():
            return []

        try:
            import cv2

            image = frame.data
            h, w = image.shape[:2]
            scale = 1.0

            if max(h, w) > self._MAX_DIMENSION:
                scale = self._MAX_DIMENSION / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            result = recognize_cv2_sync(image, self._bcp47_lang)

            text_blocks = []
            inv_scale = 1.0 / scale

            for line in result.get('lines', []):
                line_text = line.get('text', '').strip()
                if not line_text:
                    continue

                words = line.get('words', [])
                if not words:
                    continue

                # Compute bounding box spanning all words in the line
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

                x = int(x_min * inv_scale)
                y = int(y_min * inv_scale)
                width = int((x_max - x_min) * inv_scale)
                height = int((y_max - y_min) * inv_scale)

                # Windows OCR does not expose per-word confidence scores
                text_block = TextBlock(
                    text=line_text,
                    position=Rectangle(x, y, width, height),
                    confidence=1.0,
                    language=options.language if options else self.current_language,
                )
                text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            self.logger.error(f"Windows OCR processing failed: {e}")
            return []

    def extract_text_batch(
        self, frames: list[Frame], options: OCRProcessingOptions
    ) -> list[list[TextBlock]]:
        return [self.extract_text(frame, options) for frame in frames]

    def set_language(self, language: str) -> bool:
        try:
            if language != self.current_language:
                self.logger.info(
                    f"Changing language from {self.current_language} to {language}"
                )
                self._set_bcp47(language)

                from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
                from winrt.windows.globalization import Language
                if not WinOcrEngine.is_language_supported(Language(self._bcp47_lang)):
                    self.logger.error(
                        f"Language '{self._bcp47_lang}' not supported on this system"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to set language: {e}")
            return False

    def get_supported_languages(self) -> list[str]:
        return [
            'en', 'ar', 'bg', 'cs', 'da', 'de', 'el', 'es', 'fi', 'fr',
            'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'nl', 'no',
            'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'th', 'tr',
            'uk', 'vi', 'zh',
        ]

    def cleanup(self) -> None:
        from app.ocr.ocr_engine_interface import OCREngineStatus
        self.status = OCREngineStatus.UNINITIALIZED
        self.logger.info("Windows OCR engine cleaned up")

    # ------------------------------------------------------------------

    def _set_bcp47(self, iso_code: str) -> None:
        """Convert an ISO 639-1 code to a BCP-47 tag and store both."""
        self.current_language = iso_code
        self._bcp47_lang = _ISO_TO_BCP47.get(iso_code, iso_code)
