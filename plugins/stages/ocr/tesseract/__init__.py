"""
TESSERACT Plugin

OCR engine using tesserocr (Cython C++ bindings) as primary backend
with pytesseract (subprocess) as fallback.
"""

import logging
from pathlib import Path

# Try tesserocr first (fast C++ bindings), fall back to pytesseract (subprocess)
_BACKEND = None

try:
    import tesserocr
    from tesserocr import PyTessBaseAPI, RIL, OEM, PSM
    _BACKEND = "tesserocr"
except ImportError:
    tesserocr = None

if _BACKEND is None:
    try:
        import pytesseract
        _BACKEND = "pytesseract"
    except ImportError:
        pytesseract = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import IOCREngine, OCRProcessingOptions, OCREngineType, OCREngineStatus
from app.models import Frame, TextBlock, Rectangle


class OCREngine(IOCREngine):
    """Tesseract engine implementation with tesserocr/pytesseract dual backend."""

    def __init__(self, engine_name: str = "tesseract", engine_type=None):
        if engine_type is None:
            engine_type = OCREngineType.TESSERACT
        super().__init__(engine_name, engine_type)

        self._api = None  # tesserocr.PyTessBaseAPI instance (persistent)
        self._tessdata_path: str | None = None
        self._oem_mode: int = 1  # LSTM only by default
        self.current_language = 'en'
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, config: dict) -> bool:
        try:
            if _BACKEND is None:
                self.logger.error(
                    "Neither tesserocr nor pytesseract is installed"
                )
                self.status = OCREngineStatus.ERROR
                return False

            self.status = OCREngineStatus.INITIALIZING
            self.current_language = config.get('language', 'en')
            self._tessdata_path = config.get('tessdata_path', None)
            self._oem_mode = int(config.get('oem_mode', 1))

            if _BACKEND == "tesserocr":
                self._init_tesserocr()
            
            self.logger.info(
                "Tesseract initialized (backend=%s, oem=%d, lang=%s)",
                _BACKEND, self._oem_mode, self.current_language,
            )
            self.status = OCREngineStatus.READY
            return True

        except Exception as e:
            self.logger.error("Failed to initialize tesseract: %s", e)
            self.status = OCREngineStatus.ERROR
            return False

    def _init_tesserocr(self) -> None:
        """Create a persistent PyTessBaseAPI instance."""
        from app.utils.language_mapper import LanguageCodeMapper
        tess_lang = LanguageCodeMapper.to_tesseract(self.current_language)

        oem_map = {
            0: OEM.TESSERACT_ONLY,
            1: OEM.LSTM_ONLY,
            2: OEM.TESSERACT_LSTM_COMBINED,
            3: OEM.DEFAULT,
        }
        oem = oem_map.get(self._oem_mode, OEM.LSTM_ONLY)

        kwargs: dict = {"lang": tess_lang, "oem": oem}
        if self._tessdata_path:
            kwargs["path"] = self._tessdata_path

        if self._api is not None:
            self._api.End()
        self._api = PyTessBaseAPI(**kwargs)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def extract_text(self, frame: Frame, options: OCRProcessingOptions) -> list[TextBlock]:
        if not self.is_ready():
            return []

        try:
            from PIL import Image
            import numpy as np

            image = self._frame_to_pil(frame)
            if image is None:
                return []

            if _BACKEND == "tesserocr":
                return self._extract_tesserocr(image)
            else:
                return self._extract_pytesseract(image)

        except Exception as e:
            self.logger.error("OCR processing failed: %s", e)
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    # -- tesserocr path --------------------------------------------------

    def _extract_tesserocr(self, image) -> list[TextBlock]:
        api = self._api
        api.SetImage(image)

        text_blocks: list[TextBlock] = []
        iterator = api.GetIterator()
        if iterator is None:
            return text_blocks

        while True:
            try:
                if iterator.Empty(RIL.TEXTLINE):
                    if not iterator.Next(RIL.TEXTLINE):
                        break
                    continue

                line_text = iterator.GetUTF8Text(RIL.TEXTLINE)
                if not line_text or not line_text.strip():
                    if not iterator.Next(RIL.TEXTLINE):
                        break
                    continue

                conf = iterator.Confidence(RIL.TEXTLINE)
                bbox = iterator.BoundingBox(RIL.TEXTLINE)  # (left, top, right, bottom)

                if bbox:
                    left, top, right, bottom = bbox
                    position = Rectangle(
                        x=left, y=top,
                        width=right - left, height=bottom - top,
                    )
                    text_blocks.append(TextBlock(
                        text=line_text.strip(),
                        position=position,
                        confidence=conf / 100.0,
                        language=self.current_language,
                    ))
            except StopIteration:
                break

            if not iterator.Next(RIL.TEXTLINE):
                break

        self.logger.info("Tesseract (tesserocr) extracted %d text blocks", len(text_blocks))
        return text_blocks

    # -- pytesseract fallback path ----------------------------------------

    def _extract_pytesseract(self, image) -> list[TextBlock]:
        from app.utils.language_mapper import LanguageCodeMapper
        tess_lang = LanguageCodeMapper.to_tesseract(self.current_language)

        custom_config = f'--psm 6 --oem {self._oem_mode}'

        ocr_data = pytesseract.image_to_data(
            image,
            lang=tess_lang,
            config=custom_config,
            output_type=pytesseract.Output.DICT,
        )

        lines: dict[int, list[dict]] = {}
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if not text or conf < 0:
                continue
            line_num = ocr_data['line_num'][i]
            lines.setdefault(line_num, []).append({
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': conf,
            })

        text_blocks: list[TextBlock] = []
        for words in lines.values():
            if not words:
                continue
            combined_text = ' '.join(w['text'] for w in words)
            min_x = min(w['left'] for w in words)
            min_y = min(w['top'] for w in words)
            max_x = max(w['left'] + w['width'] for w in words)
            max_y = max(w['top'] + w['height'] for w in words)
            avg_conf = sum(w['conf'] for w in words) / len(words)

            position = Rectangle(
                x=min_x, y=min_y,
                width=max_x - min_x, height=max_y - min_y,
            )
            text_blocks.append(TextBlock(
                text=combined_text,
                position=position,
                confidence=avg_conf / 100.0,
                language=self.current_language,
            ))

        self.logger.info("Tesseract (pytesseract) extracted %d text blocks", len(text_blocks))
        return text_blocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_pil(frame: Frame):
        from PIL import Image
        import numpy as np

        if not isinstance(frame.data, np.ndarray):
            return None
        data = frame.data
        if len(data.shape) == 3 and data.shape[2] == 3:
            data = data[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(data)

    # ------------------------------------------------------------------
    # Batch / language / cleanup
    # ------------------------------------------------------------------

    def extract_text_batch(self, frames: list[Frame], options: OCRProcessingOptions) -> list[list[TextBlock]]:
        return [self.extract_text(f, options) for f in frames]

    def set_language(self, language: str) -> bool:
        try:
            self.current_language = language
            if _BACKEND == "tesserocr" and self._api is not None:
                self._init_tesserocr()
            return True
        except Exception as e:
            self.logger.error("Failed to set language: %s", e)
            return False

    def get_supported_languages(self) -> list[str]:
        return ['en', 'ja', 'ko', 'zh', 'de', 'fr', 'es']

    def cleanup(self) -> None:
        if self._api is not None:
            try:
                self._api.End()
            except Exception:
                pass
            self._api = None
        self.status = OCREngineStatus.UNINITIALIZED
        self.logger.info("Tesseract engine cleaned up")
