"""
Azure Translator Plugin

Premium translation using Microsoft Azure Cognitive Services Translator API v3.0.
Requires API key and region from Azure Portal.
"""

import requests

from plugins.stages.translation._base import CloudTranslationEngine


class TranslationEngine(CloudTranslationEngine):
    """Azure Translator engine plugin."""

    _default_confidence = 0.95

    def __init__(self):
        super().__init__("azure")
        self._api_key: str | None = None
        self._region = "global"
        self._endpoint = "https://api.cognitive.microsofttranslator.com"

    def initialize(self, config: dict) -> bool:
        try:
            self._api_key = config.get('api_key', self._api_key)
            self._region = config.get('region', self._region)
            self._endpoint = config.get('endpoint', self._endpoint)
            if not self._api_key:
                self._logger.error("No API key provided")
                return False
            self._is_initialized = True
            self._logger.info(f"Azure Translator initialized (region: {self._region})")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            return False

    def _do_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        response = requests.post(
            f"{self._endpoint}/translate",
            params={'api-version': '3.0', 'from': src_lang, 'to': tgt_lang},
            headers={
                'Ocp-Apim-Subscription-Key': self._api_key,
                'Ocp-Apim-Subscription-Region': self._region,
                'Content-type': 'application/json',
            },
            json=[{'text': text}],
            timeout=10,
        )
        response.raise_for_status()
        return response.json()[0]['translations'][0]['text']

    def get_supported_languages(self) -> list[str]:
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'nl', 'pl', 'tr', 'sv', 'no', 'da', 'fi', 'cs', 'el',
            'he', 'hi', 'id', 'ms', 'th', 'vi', 'uk', 'ro', 'hu', 'bg',
        ]

    def cleanup(self) -> None:
        self._is_initialized = False
        self._logger.info("Azure Translator cleaned up")
