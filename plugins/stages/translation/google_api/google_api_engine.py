"""
Google Cloud Translation Plugin

Premium translation using Google Cloud Translation API v2.
Requires API key from Google Cloud Console.
"""


try:
    from google.cloud import translate_v2 as translate
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    translate = None

from plugins.stages.translation._base import CloudTranslationEngine


class TranslationEngine(CloudTranslationEngine):
    """Google Cloud Translation engine plugin."""

    _default_confidence = 0.95

    def __init__(self):
        super().__init__("google_api")
        self._api_key: str | None = None
        self._client = None
        if not GOOGLE_API_AVAILABLE:
            self._logger.warning("google-cloud-translate not available (pip install google-cloud-translate)")

    def initialize(self, config: dict) -> bool:
        if not GOOGLE_API_AVAILABLE:
            self._logger.error("google-cloud-translate library not available")
            return False
        try:
            self._api_key = config.get('api_key', self._api_key)
            if not self._api_key:
                self._logger.error("No API key provided")
                return False
            self._client = translate.Client(api_key=self._api_key)
            self._is_initialized = True
            self._logger.info("Google Cloud Translation initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            return False

    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        result = self._client.translate(
            text,
            source_language=src_lang,
            target_language=tgt_lang,
        )
        return result['translatedText']

    def get_supported_languages(self) -> list[str]:
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'nl', 'pl', 'tr', 'sv', 'no', 'da', 'fi', 'cs', 'el',
            'he', 'hi', 'id', 'ms', 'th', 'vi', 'uk', 'ro', 'hu', 'bg',
        ]

    def cleanup(self) -> None:
        self._client = None
        self._is_initialized = False
        self._logger.info("Google Cloud Translation cleaned up")
