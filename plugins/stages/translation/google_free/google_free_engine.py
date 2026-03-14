"""
Google Translate Free Plugin

Free translation using the deep-translator library.
No API key required. Uses the Google Translate web API.
"""

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None

from plugins.stages.translation._base import CloudTranslationEngine


_LANG_MAP = {
    'zh-CN': 'zh-CN', 'zh-TW': 'zh-TW', 'zh': 'zh-CN',
}


class TranslationEngine(CloudTranslationEngine):
    """Google Translate Free engine plugin."""

    _default_confidence = 0.85

    def __init__(self):
        super().__init__("google_free")
        if not DEEP_TRANSLATOR_AVAILABLE:
            self._logger.warning("deep-translator not available (pip install deep-translator)")

    def initialize(self, config: dict) -> bool:
        if not DEEP_TRANSLATOR_AVAILABLE:
            self._logger.error("deep-translator library not available")
            return False
        try:
            GoogleTranslator(source='en', target='de').translate("test")
            self._is_initialized = True
            self._logger.info("Google Translate Free initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            return False

    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = _LANG_MAP.get(src_lang, src_lang.lower())
        tgt = _LANG_MAP.get(tgt_lang, tgt_lang.lower())
        return GoogleTranslator(source=src, target=tgt).translate(text)

    def get_supported_languages(self) -> list[str]:
        if not DEEP_TRANSLATOR_AVAILABLE:
            return []
        try:
            return list(GoogleTranslator().get_supported_languages(as_dict=True).values())
        except Exception:
            return [
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh-CN', 'zh-TW',
                'ar', 'hi', 'th', 'vi', 'nl', 'pl', 'tr', 'sv', 'no', 'da', 'fi',
                'cs', 'el', 'he', 'id', 'ms', 'ro', 'uk', 'bg', 'hr', 'sr', 'sk',
                'sl', 'et', 'lv', 'lt', 'fa', 'ur', 'bn', 'ta', 'te', 'mr', 'gu',
                'kn', 'ml', 'pa', 'si', 'ne', 'my', 'km', 'lo', 'ka', 'hy', 'az',
                'eu', 'be', 'ca', 'gl', 'is', 'mk', 'mn', 'sq', 'sw', 'tl', 'cy',
            ]

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        supported = self.get_supported_languages()
        src = _LANG_MAP.get(src_lang, src_lang.lower())
        tgt = _LANG_MAP.get(tgt_lang, tgt_lang.lower())
        return src in supported and tgt in supported

    def cleanup(self) -> None:
        self._is_initialized = False
        self._logger.info("Google Translate Free cleaned up")
