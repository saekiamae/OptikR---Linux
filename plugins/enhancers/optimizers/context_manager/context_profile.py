"""
Context Profile - Data model and compilation logic for context profiles.

A context profile is a structured JSON file that defines domain-specific
translation behavior: locked terms, translation memory, regex rules,
and formatting rules organized by translation stage (e.g. JP-EN, EN-DE).
"""
from __future__ import annotations

import json
import re
import logging
import copy
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Schema version – bump when the format changes
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LockedTerm:
    """A term that must always be translated a specific way."""
    source: str
    target: str
    type: str = "general"          # character, location, organization, attack, general
    case_sensitive: bool = True
    priority: int = 100            # higher = applied first
    notes: str = ""

    # Runtime-only (set during compilation)
    _compiled_pattern: re.Pattern | None = field(default=None, repr=False)


@dataclass
class TranslationMemoryEntry:
    """A softer translation suggestion (can be overridden by Marian if better)."""
    source: str
    target: str
    priority: int = 50
    notes: str = ""


@dataclass
class RegexRule:
    """A regex-based text transformation rule."""
    pattern: str
    action: str = "preserve"       # preserve, replace, mask
    replacement: str = ""          # used when action == "replace"
    stage: str = "pre"             # pre, post, both
    enabled: bool = True
    description: str = ""

    # Runtime-only
    _compiled: re.Pattern | None = field(default=None, repr=False)


@dataclass
class FormattingRules:
    """Post-processing formatting behaviour."""
    preserve_honorifics: bool = False
    attack_uppercase: bool = False
    translate_sound_effects: bool = True
    preserve_brackets: bool = True
    language_style: str = "neutral"   # neutral, formal, casual, technical, literary


@dataclass
class StageContext:
    """Context data for a single translation stage (e.g. JP-EN)."""
    locked_terms: list[LockedTerm] = field(default_factory=list)
    translation_memory: list[TranslationMemoryEntry] = field(default_factory=list)
    regex_rules: list[RegexRule] = field(default_factory=list)
    formatting_rules: FormattingRules = field(default_factory=FormattingRules)


# ---------------------------------------------------------------------------
# Placeholder strategy
# ---------------------------------------------------------------------------
# We use rare Unicode markers that Marian won't split or modify.
PLACEHOLDER_PREFIX = "\u27E6"   # ⟦
PLACEHOLDER_SUFFIX = "\u27E7"   # ⟧


def _make_placeholder(index: int) -> str:
    return f"{PLACEHOLDER_PREFIX}T{index:03d}{PLACEHOLDER_SUFFIX}"


# ---------------------------------------------------------------------------
# Context Profile
# ---------------------------------------------------------------------------

class ContextProfile:
    """
    Represents a loaded (and optionally compiled) context profile.

    Lifecycle:
        1. Load from JSON  →  ContextProfile.from_file() / from_dict()
        2. Compile          →  profile.compile()
        3. Use at runtime   →  profile.pre_process() / profile.post_process()
    """

    def __init__(self):
        # Meta
        self.name: str = ""
        self.version: str = "1.0"
        self.category: str = "General"
        self.description: str = ""
        self.source_language: str = ""
        self.target_language: str = ""
        self.schema_version: str = SCHEMA_VERSION

        # Stage-specific context  (key = "JP-EN", "EN-DE", or "*" for global)
        self.stages: dict[str, StageContext] = {}

        # Global fallback (applies when no stage-specific match)
        self.global_context: StageContext = StageContext()

        # Runtime state (populated by compile())
        self._compiled: bool = False
        self._placeholder_map: dict[str, str] = {}   # placeholder -> target
        self._reverse_map: dict[str, str] = {}        # placeholder -> target (for restore)
        self._sorted_locked_terms: list[LockedTerm] = []

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a JSON-compatible dict."""
        stages_dict = {}
        for stage_key, ctx in self.stages.items():
            stages_dict[stage_key] = self._stage_to_dict(ctx)

        return {
            "schema_version": self.schema_version,
            "meta": {
                "name": self.name,
                "version": self.version,
                "category": self.category,
                "description": self.description,
                "source_language": self.source_language,
                "target_language": self.target_language,
            },
            "global": self._stage_to_dict(self.global_context),
            "stages": stages_dict,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> bool:
        """Save profile to a JSON file."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.to_json(), encoding="utf-8")
            logger.info(f"Context profile saved: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save context profile: {e}")
            return False

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextProfile":
        """Create a ContextProfile from a dict (parsed JSON)."""
        profile = cls()
        meta = data.get("meta", {})
        profile.name = meta.get("name", "Unnamed")
        profile.version = meta.get("version", "1.0")
        profile.category = meta.get("category", "General")
        profile.description = meta.get("description", "")
        profile.source_language = meta.get("source_language", "")
        profile.target_language = meta.get("target_language", "")
        profile.schema_version = data.get("schema_version", "1.0")

        # Global context
        if "global" in data:
            profile.global_context = cls._parse_stage(data["global"])

        # Per-stage contexts
        for stage_key, stage_data in data.get("stages", {}).items():
            profile.stages[stage_key] = cls._parse_stage(stage_data)

        return profile

    @classmethod
    def from_file(cls, path: str) -> "ContextProfile" | None:
        """Load a profile from a JSON file."""
        try:
            p = Path(path)
            if not p.exists():
                logger.error(f"Profile file not found: {path}")
                return None
            raw = json.loads(p.read_text(encoding="utf-8"))
            profile = cls.from_dict(raw)
            logger.info(f"Loaded context profile '{profile.name}' from {path}")
            return profile
        except Exception as e:
            logger.error(f"Failed to load context profile from {path}: {e}")
            return None

    @classmethod
    def create_empty(cls, name: str = "New Profile", category: str = "General") -> "ContextProfile":
        """Create a blank profile ready for editing."""
        profile = cls()
        profile.name = name
        profile.category = category
        return profile

    # ------------------------------------------------------------------
    # Compilation  (call once when session starts)
    # ------------------------------------------------------------------

    def compile(self) -> bool:
        """
        Compile the profile for fast runtime use.

        - Sorts locked terms longest-first (avoids substring conflicts)
        - Pre-compiles regex patterns
        - Builds placeholder maps

        Returns True on success.
        """
        try:
            all_terms: list[LockedTerm] = []

            # Gather locked terms from global + all stages
            all_terms.extend(self.global_context.locked_terms)
            for ctx in self.stages.values():
                all_terms.extend(ctx.locked_terms)

            # Deduplicate by source text (highest priority wins)
            seen: dict[str, LockedTerm] = {}
            for term in all_terms:
                key = term.source if term.case_sensitive else term.source.lower()
                if key not in seen or term.priority > seen[key].priority:
                    seen[key] = term

            # Sort longest first, then by priority descending
            sorted_terms = sorted(
                seen.values(),
                key=lambda t: (-len(t.source), -t.priority),
            )

            # Build compiled patterns and placeholder maps
            self._placeholder_map = {}
            self._reverse_map = {}
            for i, term in enumerate(sorted_terms):
                flags = 0 if term.case_sensitive else re.IGNORECASE
                try:
                    term._compiled_pattern = re.compile(re.escape(term.source), flags)
                except re.error as e:
                    logger.warning(f"Bad regex for locked term '{term.source}': {e}")
                    continue
                placeholder = _make_placeholder(i)
                self._placeholder_map[term.source] = placeholder
                self._reverse_map[placeholder] = term.target

            self._sorted_locked_terms = sorted_terms

            # Compile regex rules
            for ctx in [self.global_context] + list(self.stages.values()):
                for rule in ctx.regex_rules:
                    if rule.enabled:
                        try:
                            rule._compiled = re.compile(rule.pattern)
                        except re.error as e:
                            logger.warning(f"Bad regex rule '{rule.pattern}': {e}")
                            rule.enabled = False

            self._compiled = True
            logger.info(
                f"Context profile '{self.name}' compiled: "
                f"{len(self._sorted_locked_terms)} locked terms, "
                f"{sum(len(c.regex_rules) for c in [self.global_context] + list(self.stages.values()))} regex rules"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to compile context profile: {e}")
            self._compiled = False
            return False

    @property
    def is_compiled(self) -> bool:
        return self._compiled


    # ------------------------------------------------------------------
    # Runtime processing
    # ------------------------------------------------------------------

    def get_stage_context(self, source_lang: str, target_lang: str) -> StageContext:
        """Get the context for a specific translation stage, falling back to global."""
        key = f"{source_lang}-{target_lang}".upper()
        return self.stages.get(key, self.global_context)

    def pre_process(self, text: str, source_lang: str = "", target_lang: str = "") -> tuple[str, dict[str, str]]:
        """
        Pre-translation processing.

        1. Mask locked terms with placeholders
        2. Apply pre-stage regex rules

        Returns:
            (processed_text, placeholder_restore_map)
        """
        if not self._compiled:
            logger.warning("Profile not compiled, skipping pre-processing")
            return text, {}

        result = text
        restore_map: dict[str, str] = {}

        # 1. Mask locked terms (longest first)
        for term in self._sorted_locked_terms:
            if term._compiled_pattern is None:
                continue
            placeholder = self._placeholder_map.get(term.source)
            if placeholder and term._compiled_pattern.search(result):
                result = term._compiled_pattern.sub(placeholder, result)
                restore_map[placeholder] = term.target

        # 2. Apply pre-stage regex rules
        stage_ctx = self.get_stage_context(source_lang, target_lang)
        for rule in self.global_context.regex_rules + stage_ctx.regex_rules:
            if not rule.enabled or not rule._compiled:
                continue
            if rule.stage not in ("pre", "both"):
                continue
            if rule.action == "preserve":
                # Mask matched content so Marian doesn't touch it
                for i, match in enumerate(rule._compiled.finditer(result)):
                    ph = f"{PLACEHOLDER_PREFIX}R{id(rule):08x}_{i}{PLACEHOLDER_SUFFIX}"
                    restore_map[ph] = match.group(0)
                    result = result.replace(match.group(0), ph, 1)
            elif rule.action == "replace":
                result = rule._compiled.sub(rule.replacement, result)

        return result, restore_map

    def post_process(
        self,
        text: str,
        restore_map: dict[str, str],
        source_lang: str = "",
        target_lang: str = "",
    ) -> str:
        """
        Post-translation processing.

        1. Restore placeholders (locked terms + regex preserves)
        2. Apply post-stage regex rules
        3. Apply formatting rules

        Returns:
            Final processed text.
        """
        if not self._compiled:
            return text

        result = text

        # 1. Restore all placeholders
        for placeholder, target_text in restore_map.items():
            result = result.replace(placeholder, target_text)

        # Also restore any placeholders that survived from the reverse map
        for placeholder, target_text in self._reverse_map.items():
            if placeholder in result:
                result = result.replace(placeholder, target_text)

        # 2. Apply post-stage regex rules
        stage_ctx = self.get_stage_context(source_lang, target_lang)
        for rule in self.global_context.regex_rules + stage_ctx.regex_rules:
            if not rule.enabled or not rule._compiled:
                continue
            if rule.stage not in ("post", "both"):
                continue
            if rule.action == "replace":
                result = rule._compiled.sub(rule.replacement, result)

        # 3. Apply formatting rules
        fmt = stage_ctx.formatting_rules
        if not fmt:
            fmt = self.global_context.formatting_rules

        if fmt.attack_uppercase:
            # Uppercase text inside 「」or similar attack markers
            def _upper_match(m):
                return m.group(0).upper()
            result = re.sub(r'(?<=「).*?(?=」)', _upper_match, result)

        return result

    def lookup_locked_term(self, source_text: str, source_lang: str = "", target_lang: str = "") -> str | None:
        """
        Look up a locked term by source text.
        Returns the target translation if found, None otherwise.
        Useful for Smart Dictionary integration.
        """
        if not self._compiled:
            return None

        stage_ctx = self.get_stage_context(source_lang, target_lang)
        all_terms = self.global_context.locked_terms + stage_ctx.locked_terms

        for term in all_terms:
            compare_source = source_text if term.case_sensitive else source_text.lower()
            compare_term = term.source if term.case_sensitive else term.source.lower()
            if compare_source == compare_term:
                return term.target
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_to_dict(ctx: StageContext) -> dict[str, Any]:
        return {
            "locked_terms": [
                {
                    "source": t.source,
                    "target": t.target,
                    "type": t.type,
                    "case_sensitive": t.case_sensitive,
                    "priority": t.priority,
                    "notes": t.notes,
                }
                for t in ctx.locked_terms
            ],
            "translation_memory": [
                {
                    "source": t.source,
                    "target": t.target,
                    "priority": t.priority,
                    "notes": t.notes,
                }
                for t in ctx.translation_memory
            ],
            "regex_rules": [
                {
                    "pattern": r.pattern,
                    "action": r.action,
                    "replacement": r.replacement,
                    "stage": r.stage,
                    "enabled": r.enabled,
                    "description": r.description,
                }
                for r in ctx.regex_rules
            ],
            "formatting_rules": {
                "preserve_honorifics": ctx.formatting_rules.preserve_honorifics,
                "attack_uppercase": ctx.formatting_rules.attack_uppercase,
                "translate_sound_effects": ctx.formatting_rules.translate_sound_effects,
                "preserve_brackets": ctx.formatting_rules.preserve_brackets,
                "language_style": ctx.formatting_rules.language_style,
            },
        }

    @classmethod
    def _parse_stage(cls, data: dict[str, Any]) -> StageContext:
        ctx = StageContext()

        for item in data.get("locked_terms", []):
            ctx.locked_terms.append(LockedTerm(
                source=item.get("source", ""),
                target=item.get("target", ""),
                type=item.get("type", "general"),
                case_sensitive=item.get("case_sensitive", True),
                priority=item.get("priority", 100),
                notes=item.get("notes", ""),
            ))

        for item in data.get("translation_memory", []):
            ctx.translation_memory.append(TranslationMemoryEntry(
                source=item.get("source", ""),
                target=item.get("target", ""),
                priority=item.get("priority", 50),
                notes=item.get("notes", ""),
            ))

        for item in data.get("regex_rules", []):
            ctx.regex_rules.append(RegexRule(
                pattern=item.get("pattern", ""),
                action=item.get("action", "preserve"),
                replacement=item.get("replacement", ""),
                stage=item.get("stage", "pre"),
                enabled=item.get("enabled", True),
                description=item.get("description", ""),
            ))

        fmt_data = data.get("formatting_rules", {})
        ctx.formatting_rules = FormattingRules(
            preserve_honorifics=fmt_data.get("preserve_honorifics", False),
            attack_uppercase=fmt_data.get("attack_uppercase", False),
            translate_sound_effects=fmt_data.get("translate_sound_effects", True),
            preserve_brackets=fmt_data.get("preserve_brackets", True),
            language_style=fmt_data.get("language_style", "neutral"),
        )

        return ctx
