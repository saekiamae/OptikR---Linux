"""
JSON-based Translation System for OptikR

This module provides a flexible, user-friendly translation system that:
- Loads translations from JSON files
- Supports user-provided custom languages
- Falls back to English for missing translations
- Allows hot-reloading of language packs
"""

import logging

logger = logging.getLogger(__name__)

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import threading


@dataclass
class CompletenessReport:
    """Report on how complete a language pack is relative to the English locale."""
    lang_code: str
    total_keys: int
    translated_keys: int
    untranslated_keys: int
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Result of importing a language pack."""
    success: bool
    lang_code: str = ""
    lang_name: str = ""
    translated_keys: int = 0
    total_keys: int = 0
    missing_keys: list[str] = field(default_factory=list)


class JSONTranslator:
    """
    JSON-based translator with support for:
    - Multiple language packs
    - User-provided custom languages
    - Nested key access (e.g., "ui.buttons.save")
    - Parameter substitution
    - Fallback to English
    - Thread-safe operations
    """
    
    def __init__(self, locales_dir: str | None = None):
        if locales_dir is None:
            # Default to app/localization/locales
            locales_dir = Path(__file__).parent / "locales"
        
        self.locales_dir = Path(locales_dir)
        self.current_language = "en"
        self.translations: dict[str, dict[str, Any]] = {}
        self.available_languages: dict[str, str] = {}
        self._lock = threading.RLock()
        
        # Ensure locales directory exists
        self.locales_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all available language packs
        self._discover_languages()
        self._load_language("en")  # Always load English as fallback
    
    def _discover_languages(self):
        """Discover all available language packs."""
        if not self.locales_dir.exists():
            return
        
        with self._lock:
            # Check main locales directory
            for json_file in self.locales_dir.glob("*.json"):
                self._register_language_file(json_file)
            
            # Check custom directory for user-provided languages
            custom_dir = self.locales_dir / "custom"
            if custom_dir.exists():
                for json_file in custom_dir.glob("*.json"):
                    self._register_language_file(json_file, is_custom=True)
    
    def _register_language_file(self, json_file: Path, is_custom: bool = False):
        """Register a language file."""
        lang_code = json_file.stem
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('_metadata', {})
                lang_name = metadata.get('language_name', lang_code.upper())
                
                if is_custom:
                    lang_name += " (Custom)"
                
                self.available_languages[lang_code] = lang_name
        except Exception as e:
            logger.warning("Failed to register %s: %s", json_file, e)
    
    def _load_language(self, lang_code: str) -> bool:
        """Load a language pack into memory."""
        # Try main locales directory first
        json_file = self.locales_dir / f"{lang_code}.json"
        
        # Try custom directory if not found
        if not json_file.exists():
            json_file = self.locales_dir / "custom" / f"{lang_code}.json"
        
        if not json_file.exists():
            logger.warning("Language pack not found: %s", lang_code)
            return False
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self._lock:
                self.translations[lang_code] = data.get('translations', {})
            
            return True
        except Exception as e:
            logger.error("Failed to load %s: %s", lang_code, e)
            return False
    
    def set_language(self, lang_code: str):
        """Set the current language."""
        with self._lock:
            if lang_code not in self.translations:
                if not self._load_language(lang_code):
                    logger.warning("Failed to set language to %s, using English", lang_code)
                    return
            
            self.current_language = lang_code
            logger.info("Language changed to: %s", lang_code)
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        with self._lock:
            return self.current_language
    
    def tr(self, key: str, **kwargs) -> str:
        """
        Translate a key.
        
        Args:
            key: Translation key (can be nested with dots or flat)
            **kwargs: Parameters for string formatting
        
        Returns:
            Translated string with parameters substituted
        
        Examples:
            tr("save")  # Returns "Save"
            tr("buttons.save")  # Also works if organized
            tr("error_message", error="File not found")
        """
        with self._lock:
            # Try current language
            translation = self._get_translation(self.current_language, key)
            
            # Fallback to English
            if translation is None and self.current_language != "en":
                translation = self._get_translation("en", key)
            
            # Fallback to key itself
            if translation is None:
                translation = key
            
            # Substitute parameters
            if kwargs:
                try:
                    translation = translation.format(**kwargs)
                except (KeyError, ValueError) as e:
                    logger.warning("Failed to format translation '%s': %s", key, e)
            
            return translation
    
    def _get_translation(self, lang_code: str, key: str) -> str | None:
        """Get translation for a specific language and key."""
        if lang_code not in self.translations:
            return None
        
        translations = self.translations[lang_code]
        
        # Try direct key first (flat structure)
        if key in translations:
            return translations[key]
        
        # Try nested key (e.g., "buttons.save")
        if '.' in key:
            parts = key.split('.')
            value = translations
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            if isinstance(value, str):
                return value
        
        return None
    
    def get_available_languages(self) -> dict[str, str]:
        """Get dictionary of available languages {code: name}."""
        with self._lock:
            return self.available_languages.copy()
    
    def reload_languages(self):
        """Reload all language packs (useful for hot-reload).

        Loads new data into temporary dicts first, then swaps under the lock
        so that ``tr()`` calls never see an empty translations dict.
        """
        # Build new state outside the lock (I/O heavy)
        new_translations: dict[str, dict[str, Any]] = {}
        new_available: dict[str, str] = {}

        # Discover languages into new_available
        if self.locales_dir.exists():
            for json_file in self.locales_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    metadata = data.get('_metadata', {})
                    new_available[json_file.stem] = metadata.get('language_name', json_file.stem.upper())
                except Exception as e:
                    logger.warning("Failed to register %s: %s", json_file, e)

            custom_dir = self.locales_dir / "custom"
            if custom_dir.exists():
                for json_file in custom_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        metadata = data.get('_metadata', {})
                        new_available[json_file.stem] = metadata.get('language_name', json_file.stem.upper()) + " (Custom)"
                    except Exception as e:
                        logger.warning("Failed to register %s: %s", json_file, e)

        # Load English
        en_file = self.locales_dir / "en.json"
        if en_file.exists():
            try:
                with open(en_file, 'r', encoding='utf-8') as f:
                    new_translations["en"] = json.load(f).get('translations', {})
            except Exception as e:
                logger.error("Failed to reload en.json: %s", e)

        # Load current language if not English
        with self._lock:
            current = self.current_language

        if current != "en":
            lang_file = self.locales_dir / f"{current}.json"
            if not lang_file.exists():
                lang_file = self.locales_dir / "custom" / f"{current}.json"
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        new_translations[current] = json.load(f).get('translations', {})
                except Exception as e:
                    logger.error("Failed to reload %s: %s", current, e)

        # Atomic swap under lock
        with self._lock:
            self.translations = new_translations
            self.available_languages = new_available
    
    def export_template(self, output_file: str, lang_code: str = "en") -> bool:
        """Export a language template for users to translate.

        The exported file includes a ``_metadata`` section with key counts
        and a ``translations`` section containing all keys from *lang_code*.
        """
        # --- Snapshot data under lock (no I/O here) ---
        with self._lock:
            if lang_code not in self.translations:
                logger.error("Language %s not loaded", lang_code)
                return False

            import copy
            source_translations = copy.deepcopy(self.translations[lang_code])
            en_translations = copy.deepcopy(self.translations.get("en", {}))
            lang_name = self.available_languages.get(lang_code, lang_code.upper())

        # --- Compute and write outside lock so tr() is not blocked ---
        try:
            # Count leaf string values (flat keys)
            def _count_leaves(d):
                count = 0
                for v in d.values():
                    if isinstance(v, dict):
                        count += _count_leaves(v)
                    else:
                        count += 1
                return count

            total_keys = _count_leaves(source_translations)

            # Compute _untranslated: keys whose value equals the English
            # default (only meaningful when exporting a non-English lang).
            def _count_untranslated(src, en):
                count = 0
                for k, v in src.items():
                    if isinstance(v, dict):
                        count += _count_untranslated(v, en.get(k, {}) if isinstance(en.get(k), dict) else {})
                    else:
                        en_val = en.get(k)
                        if en_val is not None and v == en_val and lang_code != "en":
                            count += 1
                return count

            untranslated = _count_untranslated(source_translations, en_translations)
            translated_keys = total_keys - untranslated

            template = {
                "_metadata": {
                    "language_code": lang_code,
                    "language_name": lang_name,
                    "version": "pre-realese-1.0.0",
                    "total_keys": total_keys,
                    "translated_keys": translated_keys,
                    "_untranslated": untranslated,
                },
                "translations": source_translations,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            logger.info("Template exported to: %s", output_file)
            return True
        except Exception as e:
            logger.error("Failed to export template: %s", e)
            return False

    
    def import_language_pack(self, json_file: str, custom: bool = True) -> "ImportResult":
        """Import a user-provided language pack.

        Validates JSON structure (requires ``_metadata`` with ``language_code``
        and ``language_name``, and a ``translations`` section).  Logs warnings
        for keys present in English but missing from the imported pack.  Saves
        to ``locales/custom/`` when *custom* is True and makes the language
        immediately available via ``get_available_languages()``.

        Returns an :class:`ImportResult` with completeness info.
        """
        fail = ImportResult(success=False)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to import language pack: %s", e)
            return fail

        # --- Validate structure ---
        if "_metadata" not in data:
            logger.warning("Language pack missing metadata")
            return fail

        metadata = data["_metadata"]
        lang_code = metadata.get("language_code")
        lang_name = metadata.get("language_name")

        if not lang_code or not lang_name:
            logger.error("Invalid metadata: missing language_code or language_name")
            return fail

        if "translations" not in data:
            logger.error("Language pack missing translations")
            return fail

        imported_translations = data["translations"]

        # --- Compute completeness against English locale ---
        en_keys = set(self._flatten_keys(self.translations.get("en", {})))
        imported_keys = set(self._flatten_keys(imported_translations))
        missing = sorted(en_keys - imported_keys)

        if missing:
            logger.warning(
                "Imported pack '%s' is missing %d key(s): %s",
                lang_code,
                len(missing),
                ", ".join(missing[:20]) + ("..." if len(missing) > 20 else ""),
            )

        # --- Save to disk ---
        try:
            if custom:
                save_dir = self.locales_dir / "custom"
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.locales_dir

            save_path = save_dir / f"{lang_code}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save language pack: %s", e)
            return fail

        # --- Make immediately available ---
        with self._lock:
            self.translations[lang_code] = imported_translations
            self.available_languages[lang_code] = lang_name + (" (Custom)" if custom else "")

        logger.info("Imported language pack: %s (%s)", lang_name, lang_code)

        return ImportResult(
            success=True,
            lang_code=lang_code,
            lang_name=lang_name,
            translated_keys=len(imported_keys & en_keys),
            total_keys=len(en_keys),
            missing_keys=missing,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_keys(d: dict, prefix: str = "") -> list:
        """Return a flat list of dot-separated keys from a (possibly nested) dict."""
        keys: list = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(JSONTranslator._flatten_keys(v, full_key))
            else:
                keys.append(full_key)
        return keys

    def get_completeness_report(self, lang_code: str) -> "CompletenessReport":
        """Compare *lang_code* keys against English keys."""
        with self._lock:
            en_keys = set(self._flatten_keys(self.translations.get("en", {})))
            lang_keys = set(self._flatten_keys(self.translations.get(lang_code, {})))

        translated = len(lang_keys & en_keys)
        return CompletenessReport(
            lang_code=lang_code,
            total_keys=len(en_keys),
            translated_keys=translated,
            untranslated_keys=len(en_keys) - translated,
            missing_keys=sorted(en_keys - lang_keys),
            extra_keys=sorted(lang_keys - en_keys),
        )


# Global translator instance
_translator: JSONTranslator | None = None
_init_lock = threading.Lock()


def init_translator(locales_dir: str | None = None):
    """Initialize the global translator."""
    global _translator
    with _init_lock:
        if _translator is None:
            _translator = JSONTranslator(locales_dir)


def get_translator() -> JSONTranslator:
    """Get the global translator instance."""
    if _translator is None:
        init_translator()
    return _translator


def set_language(lang_code: str):
    """Set the current language."""
    get_translator().set_language(lang_code)


def get_current_language() -> str:
    """Get the current language code."""
    return get_translator().get_current_language()


def tr(key: str, **kwargs) -> str:
    """Translate a key."""
    return get_translator().tr(key, **kwargs)


def get_available_languages() -> dict[str, str]:
    """Get available languages."""
    return get_translator().get_available_languages()


def reload_languages():
    """Reload all language packs."""
    get_translator().reload_languages()


def export_template(output_file: str, lang_code: str = "en") -> bool:
    """Export a language template."""
    return get_translator().export_template(output_file, lang_code)


def import_language_pack(json_file: str, custom: bool = True) -> "ImportResult":
    """Import a language pack."""
    return get_translator().import_language_pack(json_file, custom)
