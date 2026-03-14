"""
Context Manager Plugin Implementation

A domain-aware context intelligence layer that provides pre- and post-translation
processing through structured JSON profiles. Profiles contain locked terms,
translation memory, regex rules, and formatting rules organized by translation stage.

The plugin is compiled once when the translation session starts and provides
deterministic, fast lookups during live translation.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .context_profile import (
    ContextProfile,
    StageContext,
    LockedTerm,
    TranslationMemoryEntry,
    RegexRule,
    FormattingRules,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


class ContextManagerPlugin:
    """
    Context Manager optimizer plugin.

    Manages context profiles and provides pre/post translation processing.
    Profiles are JSON files stored in user_data/context_profiles/.

    Lifecycle:
        1. Plugin is initialized at app startup (warm pipeline)
        2. User selects a profile in the UI
        3. User presses Start → compile() is called → profile is frozen
        4. During translation: pre_process() before Marian, post_process() after
        5. User presses Stop → session ends
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the Context Manager plugin."""
        self.config = config
        self.config_manager = None
        self.error_notification_callback = None

        # Profile management
        self._profiles_dir: Path | None = None
        self._available_profiles: dict[str, str] = {}  # name -> file path
        self._active_profile: ContextProfile | None = None
        self._active_profile_name: str = ""

        # Session state
        self._session_active: bool = False
        self._restore_map: dict[str, str] = {}  # per-text placeholder map

        self._last_error: str = ""

        # Stats
        self._stats = {
            "terms_masked": 0,
            "terms_restored": 0,
            "regex_rules_applied": 0,
            "texts_processed": 0,
        }

        logger.info("Context Manager plugin created")

    # ------------------------------------------------------------------
    # Setup (called by plugin system)
    # ------------------------------------------------------------------

    def set_config_manager(self, config_manager):
        """Set the configuration manager reference."""
        self.config_manager = config_manager

    def set_error_notification_callback(self, callback):
        """Set callback for error notifications to UI."""
        self.error_notification_callback = callback

    def initialize(self) -> bool:
        """
        Initialize plugin: discover profiles directory and scan for available profiles.
        Called by plugin system after instantiation.
        """
        try:
            # Determine profiles directory
            self._profiles_dir = self._resolve_profiles_dir()
            self._profiles_dir.mkdir(parents=True, exist_ok=True)

            # Create a sample profile if directory is empty
            self._ensure_sample_profile()

            # Scan for available profiles
            self.refresh_profiles()

            # Load last active profile (but don't compile — that happens at Start)
            if self.config_manager:
                saved_name = self.config_manager.get_setting(
                    "plugins.context_manager.active_profile", ""
                )
                if saved_name and saved_name in self._available_profiles:
                    self.load_profile(saved_name)

            logger.info(
                f"Context Manager initialized. "
                f"{len(self._available_profiles)} profiles found in {self._profiles_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Context Manager initialization failed: {e}")
            self._notify_error("Initialization Error", str(e))
            return False

    # ------------------------------------------------------------------
    # Profile discovery & management
    # ------------------------------------------------------------------

    def refresh_profiles(self) -> dict[str, str]:
        """Scan profiles directory and return {name: filepath} map."""
        self._available_profiles = {}
        if not self._profiles_dir or not self._profiles_dir.exists():
            return self._available_profiles

        for f in sorted(self._profiles_dir.glob("*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                name = raw.get("meta", {}).get("name", f.stem)
                self._available_profiles[name] = str(f)
            except Exception as e:
                logger.warning(f"Skipping invalid profile {f.name}: {e}")

        return self._available_profiles

    def get_available_profiles(self) -> dict[str, str]:
        """Return dict of {profile_name: file_path}."""
        return dict(self._available_profiles)

    def get_profiles_by_category(self) -> dict[str, list[str]]:
        """Return profiles grouped by category."""
        categories: dict[str, list[str]] = {}
        for name, path in self._available_profiles.items():
            try:
                raw = json.loads(Path(path).read_text(encoding="utf-8"))
                cat = raw.get("meta", {}).get("category", "General")
            except Exception:
                cat = "General"
            categories.setdefault(cat, []).append(name)
        return categories

    def load_profile(self, name: str) -> bool:
        """Load a profile by name (does NOT compile it)."""
        if name not in self._available_profiles:
            logger.error(f"Profile '{name}' not found")
            return False

        path = self._available_profiles[name]
        profile = ContextProfile.from_file(path)
        if profile is None:
            return False

        self._active_profile = profile
        self._active_profile_name = name

        # Persist selection
        if self.config_manager:
            self.config_manager.set_setting(
                "plugins.context_manager.active_profile", name
            )
            self.config_manager.save_config()

        logger.info(f"Profile '{name}' loaded (not yet compiled)")
        return True

    def get_active_profile(self) -> ContextProfile | None:
        """Return the currently loaded profile (may or may not be compiled)."""
        return self._active_profile

    def get_active_profile_name(self) -> str:
        return self._active_profile_name

    # ------------------------------------------------------------------
    # Session lifecycle (Start / Stop)
    # ------------------------------------------------------------------

    def compile(self) -> bool:
        """
        Compile the active profile for the translation session.
        Call this when the user presses Start.
        """
        if self._active_profile is None:
            logger.info("No context profile active — nothing to compile")
            return True  # Not an error, just no context

        success = self._active_profile.compile()
        if success:
            self._session_active = True
            self._reset_stats()
            logger.info(f"Context session started with profile '{self._active_profile_name}'")
        return success

    def end_session(self):
        """End the current translation session."""
        self._session_active = False
        self._restore_map = {}
        logger.info("Context session ended")

    @property
    def session_active(self) -> bool:
        return self._session_active

    # ------------------------------------------------------------------
    # Pipeline processing (called during live translation)
    # ------------------------------------------------------------------

    def pre_process(self, text: str, source_lang: str = "", target_lang: str = "") -> tuple[str, dict[str, str]]:
        """
        Pre-translation processing: mask locked terms, apply pre-regex rules.

        Returns:
            (processed_text, restore_map) — pass restore_map to post_process().
        """
        if not self._session_active or self._active_profile is None:
            return text, {}

        if not self._active_profile.is_compiled:
            return text, {}

        processed, restore_map = self._active_profile.pre_process(
            text, source_lang, target_lang
        )

        if processed != text:
            self._stats["terms_masked"] += len(restore_map)

        self._stats["texts_processed"] += 1
        return processed, restore_map

    def post_process(
        self,
        text: str,
        restore_map: dict[str, str],
        source_lang: str = "",
        target_lang: str = "",
    ) -> str:
        """
        Post-translation processing: restore placeholders, apply formatting rules.
        """
        if not self._session_active or self._active_profile is None:
            return text

        if not self._active_profile.is_compiled:
            return text

        result = self._active_profile.post_process(
            text, restore_map, source_lang, target_lang
        )

        if result != text:
            self._stats["terms_restored"] += len(restore_map)

        return result

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pre-translation hook: mask locked terms in each text block.

        For each text block in ``text_blocks``, this calls
        ``pre_process()`` to replace locked terms with placeholders
        before the text reaches the translator.  The per-block restore
        maps are stored in ``data["_context_restore_maps"]`` so
        ``post_process_pipeline()`` can undo the masking after
        translation.
        """
        if self._active_profile:
            data["context_profile"] = self._active_profile_name
            data["context_category"] = self._active_profile.category
            data["context_compiled"] = self._active_profile.is_compiled

        if not self._session_active or self._active_profile is None:
            return data
        if not self._active_profile.is_compiled:
            return data

        text_blocks = data.get("text_blocks", [])
        if not text_blocks:
            return data

        source_lang = data.get("source_lang", data.get("source_language", ""))
        target_lang = data.get("target_lang", data.get("target_language", ""))
        restore_maps: list[dict[str, str]] = []

        for block in text_blocks:
            if isinstance(block, dict):
                original = block.get("text", "")
            else:
                original = getattr(block, "text", str(block))

            processed, restore_map = self.pre_process(
                original, source_lang, target_lang
            )
            restore_maps.append(restore_map)

            if processed != original:
                if isinstance(block, dict):
                    block["text"] = processed
                    block["_original_text"] = original
                else:
                    block.text = processed
                    block._original_text = original

        data["_context_restore_maps"] = restore_maps
        return data

    def post_process_pipeline(self, data: dict[str, Any]) -> dict[str, Any]:
        """Post-translation hook: restore masked placeholders or apply locked terms (vision path).

        Text pipeline: uses restore maps from pre_process to restore placeholders.
        Vision pipeline: no pre_process, so restore_maps is empty; applies locked-term
        lookup using the original (source) text from each block so e.g. 覇気 → "Haki".
        """
        if not self._session_active or self._active_profile is None:
            return data
        if not self._active_profile.is_compiled:
            return data

        translations = data.get("translations", [])
        restore_maps = data.get("_context_restore_maps", [])
        text_blocks = data.get("text_blocks", [])
        source_lang = data.get("source_lang", data.get("source_language", ""))
        target_lang = data.get("target_lang", data.get("target_language", ""))

        if not translations:
            return data

        # Vision path: no restore maps (we didn't mask before vision). Apply locked terms
        # using original text so e.g. character names / synonyms (Haki, etc.) are used.
        if not restore_maps and text_blocks:
            new_translations = []
            for i, trans in enumerate(translations):
                trans_str = str(trans).strip()
                original = ""
                if i < len(text_blocks):
                    block = text_blocks[i]
                    original = (
                        block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    )
                    if isinstance(original, str):
                        original = original.strip()
                locked = self._active_profile.lookup_locked_term(original, source_lang, target_lang)
                if locked is not None and locked.strip():
                    trans_str = locked.strip()
                trans_str = self.post_process(trans_str, {}, source_lang, target_lang)
                new_translations.append(trans_str)
            data["translations"] = new_translations
            return data

        if not restore_maps:
            return data

        new_translations = []
        for i, trans in enumerate(translations):
            rmap = restore_maps[i] if i < len(restore_maps) else {}
            if rmap:
                trans = self.post_process(
                    str(trans), rmap, source_lang, target_lang
                )

            # Restore original source text on the block (undo masking)
            if i < len(text_blocks):
                block = text_blocks[i]
                if isinstance(block, dict):
                    orig = block.pop("_original_text", None)
                    if orig is not None:
                        block["text"] = orig
                elif hasattr(block, "_original_text"):
                    block.text = block._original_text
                    del block._original_text

            new_translations.append(trans)

        data["translations"] = new_translations
        return data

    def lookup_term(self, source_text: str, source_lang: str = "", target_lang: str = "") -> str | None:
        """
        Look up a locked term. Used by Smart Dictionary integration.
        Context locked terms have highest priority.
        """
        if self._active_profile is None or not self._active_profile.is_compiled:
            return None
        return self._active_profile.lookup_locked_term(source_text, source_lang, target_lang)

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_profile(self, name: str, export_path: str) -> bool:
        """Export a profile to an external path."""
        if name not in self._available_profiles:
            self._last_error = f"Profile '{name}' not found"
            logger.error(f"Cannot export: {self._last_error}")
            return False

        try:
            src = Path(self._available_profiles[name])
            dst = Path(export_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            logger.info(f"Profile '{name}' exported to {export_path}")
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Export failed: {e}")
            self._notify_error("Export Error", str(e))
            return False

    def import_profile(self, import_path: str) -> str | None:
        """
        Import a profile from an external JSON file.
        Returns the profile name on success, None on failure.

        If a profile with the same name already exists the imported
        copy is renamed (e.g. "My Profile (2)") so that the original
        entry in the profile list is not silently overwritten.
        """
        try:
            src = Path(import_path)
            if not src.exists():
                logger.error(f"Import file not found: {import_path}")
                return None

            raw = json.loads(src.read_text(encoding="utf-8"))
            profile = ContextProfile.from_dict(raw)
            if not profile.name:
                logger.error("Imported profile has no name")
                return None

            original_name = profile.name
            final_name = original_name

            # Deduplicate meta name so refresh_profiles() won't overwrite
            existing_names = set(self._available_profiles.keys())
            if final_name in existing_names:
                counter = 2
                while f"{original_name} ({counter})" in existing_names:
                    counter += 1
                final_name = f"{original_name} ({counter})"
                profile.name = final_name
                raw.setdefault("meta", {})["name"] = final_name

            dest_name = self._safe_filename(final_name) + ".json"
            dest = self._profiles_dir / dest_name

            counter = 1
            while dest.exists():
                dest_name = f"{self._safe_filename(final_name)}_{counter}.json"
                dest = self._profiles_dir / dest_name
                counter += 1

            dest.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.refresh_profiles()
            logger.info(f"Profile '{final_name}' imported from {import_path}")
            return final_name

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Import failed: {e}")
            self._notify_error("Import Error", str(e))
            return None

    # ------------------------------------------------------------------
    # Profile CRUD
    # ------------------------------------------------------------------

    def create_profile(self, name: str, category: str = "General") -> ContextProfile | None:
        """Create a new empty profile and save it."""
        profile = ContextProfile.create_empty(name, category)
        filename = self._safe_filename(name) + ".json"
        path = self._profiles_dir / filename

        if path.exists():
            logger.warning(f"Profile file already exists: {filename}")
            return None

        if profile.save(str(path)):
            self.refresh_profiles()
            return profile
        return None

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""
        if name not in self._available_profiles:
            return False

        try:
            Path(self._available_profiles[name]).unlink()
            if self._active_profile_name == name:
                self._active_profile = None
                self._active_profile_name = ""
            self.refresh_profiles()
            logger.info(f"Profile '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete profile '{name}': {e}")
            return False

    def save_active_profile(self) -> bool:
        """Save the currently active profile back to disk."""
        if self._active_profile is None or not self._active_profile_name:
            return False

        if self._active_profile_name not in self._available_profiles:
            return False

        path = self._available_profiles[self._active_profile_name]
        return self._active_profile.save(path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get plugin statistics."""
        return {
            "active_profile": self._active_profile_name or "None",
            "session_active": self._session_active,
            "profiles_available": len(self._available_profiles),
            "compiled": self._active_profile.is_compiled if self._active_profile else False,
            "locked_terms_count": len(self._active_profile._sorted_locked_terms) if self._active_profile and self._active_profile.is_compiled else 0,
            **self._stats,
        }

    def _reset_stats(self):
        for key in self._stats:
            self._stats[key] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_profiles_dir(self) -> Path:
        """Determine the profiles directory path."""
        if self.config_manager:
            custom = self.config_manager.get_setting(
                "plugins.context_manager.profiles_directory", ""
            )
            if custom:
                return Path(custom)

        from app.utils.path_utils import get_context_profiles_dir
        return get_context_profiles_dir()

    def _ensure_sample_profile(self):
        """Create a sample profile if the profiles directory is empty."""
        if not self._profiles_dir:
            return
        existing = list(self._profiles_dir.glob("*.json"))
        if existing:
            return

        sample = ContextProfile.create_empty("Sample - Manga", "Media")
        sample.description = "Example manga context profile. Edit or replace with your own."
        sample.source_language = "ja"
        sample.target_language = "en"

        # Add some example locked terms
        sample.global_context.locked_terms = [
            LockedTerm(source="ナルト", target="Naruto", type="character", priority=100,
                       notes="Main protagonist - keep original name"),
            LockedTerm(source="木ノ葉", target="Konoha", type="location", priority=90,
                       notes="Hidden Leaf Village"),
            LockedTerm(source="暁", target="Akatsuki", type="organization", priority=90),
        ]

        sample.global_context.translation_memory = [
            TranslationMemoryEntry(source="火影", target="Hokage", priority=80,
                                   notes="Title - Shadow of Fire"),
        ]

        sample.global_context.regex_rules = [
            RegexRule(
                pattern=r"「(.*?)」",
                action="preserve",
                stage="pre",
                description="Preserve text inside Japanese quotation brackets",
            ),
        ]

        sample.global_context.formatting_rules = FormattingRules(
            preserve_honorifics=True,
            attack_uppercase=True,
            translate_sound_effects=False,
            language_style="casual",
        )

        filename = "sample_manga.json"
        sample.save(str(self._profiles_dir / filename))
        logger.info("Created sample manga context profile")

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert a profile name to a safe filename."""
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
        return safe.strip().replace(" ", "_").lower()

    def _notify_error(self, title: str, message: str):
        """Send error notification to UI if callback is set."""
        if self.error_notification_callback:
            try:
                self.error_notification_callback(title, message)
            except Exception:
                pass
        logger.error(f"{title}: {message}")
