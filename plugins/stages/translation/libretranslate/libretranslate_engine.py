"""
LibreTranslate Plugin

Free, open-source translation using the LibreTranslate API.
Supports public instance or self-hosted.
"""

import requests

from plugins.stages.translation._base import CloudTranslationEngine


# Language code normalization for LibreTranslate (ISO 639-1)
_LANG_MAP = {
    'zh-CN': 'zh', 'zh-TW': 'zh', 'zh-cn': 'zh', 'zh-tw': 'zh',
}


class TranslationEngine(CloudTranslationEngine):
    """LibreTranslate engine plugin."""

    _default_confidence = 0.80

    def __init__(self):
        super().__init__("libretranslate")
        self._api_url = "https://libretranslate.com"
        self._api_key: str | None = None
        self._timeout = 10
        self._fetched_languages: list[str] = []

    def initialize(self, config: dict) -> bool:
        try:
            self._api_url = config.get('api_url', self._api_url).rstrip('/')
            # Strip /translate suffix if present (plugin.json default has it)
            if self._api_url.endswith('/translate'):
                self._api_url = self._api_url[:-len('/translate')]
            self._api_key = config.get('api_key') or None
            self._timeout = int(config.get('timeout', 10))
            # Fetch supported languages (also validates connectivity)
            self._fetch_languages()
            self._is_initialized = True
            self._logger.info(f"LibreTranslate initialized (URL: {self._api_url})")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            return False

    def _fetch_languages(self) -> None:
        """Fetch supported languages from the API."""
        try:
            resp = requests.get(f"{self._api_url}/languages", timeout=self._timeout)
            resp.raise_for_status()
            self._fetched_languages = [lang['code'] for lang in resp.json()]
            self._logger.info(f"LibreTranslate supports {len(self._fetched_languages)} languages")
        except Exception as e:
            self._logger.warning(f"Could not fetch languages, using fallback list: {e}")
            self._fetched_languages = [
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar',
            ]

    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = _LANG_MAP.get(src_lang, src_lang.lower())
        tgt = _LANG_MAP.get(tgt_lang, tgt_lang.lower())
        payload = {'q': text, 'source': src, 'target': tgt, 'format': 'text'}
        if self._api_key:
            payload['api_key'] = self._api_key
        resp = requests.post(
            f"{self._api_url}/translate",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get('translatedText', text)

    def get_supported_languages(self) -> list[str]:
        return list(self._fetched_languages)

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        langs = self._fetched_languages
        src = _LANG_MAP.get(src_lang, src_lang.lower())
        tgt = _LANG_MAP.get(tgt_lang, tgt_lang.lower())
        return src in langs and tgt in langs

    def cleanup(self) -> None:
        self._is_initialized = False
        self._logger.info("LibreTranslate cleaned up")
