"""
Dictionary Operations for Translation Layer.

Handles dictionary export, reload, stats, and language pair management.

Requirements: 3.1
"""
import logging
from typing import Any

from app.text_translation.translation_engine_interface import TranslationEngineRegistry


class DictionaryOps:
    """Dictionary export, reload, stats, and language pair operations."""

    def __init__(
        self,
        engine_registry: TranslationEngineRegistry,
        config_manager: Any | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._engine_registry = engine_registry
        self._config_manager = config_manager

    def _get_dict_engine_internals(self) -> Any | None:
        """Get the dictionary engine's internal _dictionary object, or None."""
        dict_engine = self._engine_registry.get_engine("dictionary")
        if dict_engine and hasattr(dict_engine, "_dictionary"):
            return dict_engine._dictionary
        return None

    # -- language pair helpers ------------------------------------------------

    def get_current_language_pair(self) -> tuple[str, str]:
        """Get the current language pair from config."""
        try:
            if self._config_manager:
                source = self._config_manager.get_setting(
                    "translation.source_language", "en"
                )
                target = self._config_manager.get_setting(
                    "translation.target_language", "de"
                )
            else:
                source = "en"
                target = "de"
            return (source, target)
        except Exception as e:
            self._logger.error(f"Failed to get current language pair: {e}")
            return ("en", "de")

    def set_language_pair(self, source_lang: str, target_lang: str) -> None:
        """Set the current language pair."""
        try:
            if self._config_manager:
                self._config_manager.set_setting(
                    "translation.source_language", source_lang
                )
                self._config_manager.set_setting(
                    "translation.target_language", target_lang
                )
            self._logger.info(f"Language pair set to {source_lang} → {target_lang}")
        except Exception as e:
            self._logger.error(f"Failed to set language pair: {e}")

    # -- dictionary stats / management -----------------------------------------

    def get_dictionary_stats(self) -> dict[str, Any]:
        """Get statistics for the current dictionary."""
        try:
            dict_internal = self._get_dict_engine_internals()
            if dict_internal is None:
                return {"total_entries": 0, "total_usage": 0}

            source, target = self.get_current_language_pair()
            lang_pair = (source, target)

            if hasattr(dict_internal, "_dictionaries"):
                if lang_pair in dict_internal._dictionaries:
                    dictionary = dict_internal._dictionaries[lang_pair]
                    total_entries = len(dictionary)
                    total_usage = sum(
                        entry.get("usage_count", 0)
                        if isinstance(entry, dict)
                        else 0
                        for entry in dictionary.values()
                    )
                    return {
                        "total_entries": total_entries,
                        "total_usage": total_usage,
                    }

            return {"total_entries": 0, "total_usage": 0}
        except Exception as e:
            self._logger.error(f"Failed to get dictionary stats: {e}")
            return {"total_entries": 0, "total_usage": 0}

    def clear_dictionary(self) -> None:
        """Clear the current dictionary."""
        try:
            dict_internal = self._get_dict_engine_internals()
            if dict_internal is None:
                self._logger.warning("Dictionary engine not available")
                return

            source, target = self.get_current_language_pair()
            lang_pair = (source, target)

            if hasattr(dict_internal, "_dictionaries"):
                if lang_pair in dict_internal._dictionaries:
                    dict_internal._dictionaries[lang_pair] = {}

                    from app.utils.path_utils import get_dictionary_file

                    dict_path = get_dictionary_file(source, target)
                    dict_path.parent.mkdir(parents=True, exist_ok=True)
                    dict_internal._save_dictionary(lang_pair, dict_path)

                    self._logger.info(f"Cleared dictionary for {source} → {target}")
        except Exception as e:
            self._logger.error(f"Failed to clear dictionary: {e}")

    def reload_dictionary_from_file(
        self,
        file_path: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> None:
        """Reload dictionary from a specific file."""
        try:
            if source_lang is None or target_lang is None:
                source_lang, target_lang = self.get_current_language_pair()

            dict_internal = self._get_dict_engine_internals()
            if dict_internal is None:
                self._logger.warning("Dictionary engine not available")
                return

            dict_internal.reload_specific_dictionary(
                file_path, source_lang, target_lang
            )
            self._logger.info(f"Reloaded dictionary from {file_path}")
        except Exception as e:
            self._logger.error(f"Failed to reload dictionary from file: {e}")

    def get_loaded_dictionary_path(
        self,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str | None:
        """Get the file path of the currently loaded dictionary."""
        try:
            if source_lang is None or target_lang is None:
                source_lang, target_lang = self.get_current_language_pair()

            dict_internal = self._get_dict_engine_internals()
            if dict_internal is None:
                return None

            result: str | None = dict_internal.get_loaded_dictionary_path(
                source_lang, target_lang
            )
            return result
        except Exception as e:
            self._logger.error(f"Failed to get loaded dictionary path: {e}")
            return None

    def get_available_language_pairs(self) -> list[tuple[str, str, str, int]]:
        """Get list of all available language-pair dictionaries."""
        try:
            dict_internal = self._get_dict_engine_internals()

            if dict_internal is not None:
                pairs = dict_internal.get_available_language_pairs()

                unique_pairs: dict[tuple[str, str], tuple[str, str, str, int]] = {}
                for source, target, path, count in pairs:
                    key = (source, target)
                    if key not in unique_pairs or count > unique_pairs[key][3]:
                        unique_pairs[key] = (source, target, path, count)

                return list(unique_pairs.values())
            else:
                self._logger.warning("SmartDictionary not available")
                return []
        except Exception as e:
            self._logger.error(f"Failed to get available language pairs: {e}")
            return []

    def export_dictionary_wordbook(
        self,
        output_path: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str | None:
        """Export dictionary as a human-readable wordbook."""
        try:
            from pathlib import Path

            if source_lang is None:
                source_lang, _ = self.get_current_language_pair()
            if target_lang is None:
                _, target_lang = self.get_current_language_pair()

            dict_engine = self._engine_registry.get_engine("dictionary")
            if not dict_engine:
                self._logger.warning("Dictionary engine not available for export")
                return None

            dict_internal = self._get_dict_engine_internals()
            lang_pair = (source_lang, target_lang)
            if dict_internal is None or not hasattr(dict_internal, "_dictionaries"):
                self._logger.warning("No dictionaries loaded")
                return None

            if lang_pair not in dict_internal._dictionaries:
                self._logger.warning(
                    f"No dictionary for {source_lang} → {target_lang}"
                )
                return None

            dictionary = dict_internal._dictionaries[lang_pair]

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# Dictionary Wordbook: {source_lang} → {target_lang}\n"
                )
                f.write(f"# Total entries: {len(dictionary)}\n")
                f.write(
                    f"# Exported: {__import__('datetime').datetime.now().isoformat()}\n"
                )
                f.write("\n" + "=" * 80 + "\n\n")

                sorted_entries = []
                for key, entry_data in dictionary.items():
                    if ":" in key:
                        parts = key.split(":", 2)
                        source_text = parts[2] if len(parts) > 2 else key
                    else:
                        source_text = key

                    if isinstance(entry_data, dict):
                        translation = entry_data.get("translation", source_text)
                        usage_count = entry_data.get("usage_count", 0)
                        confidence = entry_data.get("confidence", 0.0)
                    else:
                        translation = str(entry_data)
                        usage_count = 0
                        confidence = 0.0

                    sorted_entries.append(
                        (source_text, translation, usage_count, confidence)
                    )

                sorted_entries.sort(key=lambda x: x[2], reverse=True)

                for source_text, translation, usage_count, confidence in sorted_entries:
                    f.write(f"{source_text}\n")
                    f.write(f"  → {translation}\n")
                    f.write(
                        f"  Usage: {usage_count} | Confidence: {confidence:.2f}\n"
                    )
                    f.write("\n")

            self._logger.info(f"Exported dictionary wordbook to {output_file}")
            return str(output_file)

        except Exception as e:
            self._logger.error(f"Failed to export dictionary wordbook: {e}")
            import traceback
            traceback.print_exc()
            return None
