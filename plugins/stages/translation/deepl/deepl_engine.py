"""
DeepL Translation Plugin

Premium translation using the DeepL API via the official Python SDK.
Requires API key from DeepL.
"""


try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False
    deepl = None

from plugins.stages.translation._base import CloudTranslationEngine


class TranslationEngine(CloudTranslationEngine):
    """DeepL translation engine plugin."""

    _default_confidence = 0.98

    def __init__(self):
        super().__init__("deepl")
        self._api_key: str | None = None
        self._translator = None
        if not DEEPL_AVAILABLE:
            self._logger.warning("deepl library not available (pip install deepl)")

    def initialize(self, config: dict) -> bool:
        if not DEEPL_AVAILABLE:
            self._logger.error("deepl library not available")
            return False
        try:
            self._api_key = config.get('api_key', self._api_key)
            if not self._api_key:
                self._logger.error("No API key provided")
                return False
            self._translator = deepl.Translator(self._api_key)
            self._is_initialized = True
            self._logger.info("DeepL engine initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            return False

    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        result = self._translator.translate_text(
            text,
            source_lang=src_lang.upper(),
            target_lang=tgt_lang.upper(),
        )
        return result.text

    def get_supported_languages(self) -> list[str]:
        return [
            'en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ru',
            'ja', 'zh', 'bg', 'cs', 'da', 'el', 'et', 'fi', 'hu',
            'id', 'lv', 'lt', 'ro', 'sk', 'sl', 'sv', 'tr', 'uk',
        ]

    def cleanup(self) -> None:
        self._translator = None
        self._is_initialized = False
        self._logger.info("DeepL engine cleaned up")
