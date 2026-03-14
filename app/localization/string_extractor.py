"""String extractor tool for scanning Python source files and extracting user-facing strings."""

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a string extraction scan."""
    new_strings: dict[str, str] = field(default_factory=dict)  # key -> english value
    existing_keys_preserved: int = 0
    total_keys: int = 0


@dataclass
class ValidationReport:
    """Result of validating tr() calls against the English locale file."""
    missing_keys: list[tuple[str, str, int]] = field(default_factory=list)  # (key, file_path, line_number)
    orphaned_keys: list[str] = field(default_factory=list)


    @dataclass
    class ValidationReport:
        """Result of validating tr() calls against the English locale file."""
        missing_keys: list[tuple[str, str, int]] = field(default_factory=list)  # (key, file_path, line_number)
        orphaned_keys: list[str] = field(default_factory=list)



# Method/function names whose first string argument should be extracted
_SETTER_PATTERNS = frozenset({
    "setText",
    "setTitle",
    "setWindowTitle",
    "setPlaceholderText",
    "setToolTip",
    "addItem",
})

# Widget constructors whose first string argument should be extracted
_WIDGET_PATTERNS = frozenset({
    "QLabel",
    "QPushButton",
    "QGroupBox",
})

# QMessageBox static methods (information, warning, critical, question, about)
_QMESSAGEBOX_METHODS = frozenset({
    "information",
    "warning",
    "critical",
    "question",
    "about",
})


class StringExtractor:
    """Scans Python source files for user-facing strings and extracts them."""

    def __init__(self, source_dirs: list[str], locale_file: str):
        self.source_dirs = source_dirs
        self.locale_file = locale_file
        self._existing_keys: dict[str, str] = {}
        self._used_keys: set[str] = set()
        self._load_existing_keys()

    def _load_existing_keys(self) -> None:
        """Load existing keys from the locale file."""
        if not os.path.exists(self.locale_file):
            return
        try:
            with open(self.locale_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            translations = data.get("translations", {})
            self._existing_keys = dict(self._flatten_translations(translations))
            self._used_keys = set(self._existing_keys.keys())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load existing locale file %s: %s", self.locale_file, e)

    @staticmethod
    def _flatten_translations(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        """Flatten nested translation dict into (key, value) pairs."""
        items = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(StringExtractor._flatten_translations(v, full_key))
            else:
                items.append((full_key, v))
        return items

    def scan(self) -> ExtractionResult:
        """Scan source files for hardcoded strings."""
        discovered: dict[str, str] = {}  # text -> generated key (deduped by text)

        for source_dir in self.source_dirs:
            for root, _dirs, files in os.walk(source_dir):
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    filepath = os.path.join(root, filename)
                    strings = self._extract_from_file(filepath)
                    for s in strings:
                        if s not in discovered:
                            discovered[s] = None  # placeholder

        # Generate keys for discovered strings
        new_strings: dict[str, str] = {}
        for text in discovered:
            # Check if this text already exists as a value in existing keys
            existing_key = self._find_existing_key_for_value(text)
            if existing_key is not None:
                continue  # Already in locale file
            key = self.generate_key(text)
            new_strings[key] = text

        existing_preserved = len(self._existing_keys)
        total = existing_preserved + len(new_strings)

        return ExtractionResult(
            new_strings=new_strings,
            existing_keys_preserved=existing_preserved,
            total_keys=total,
        )

    def _find_existing_key_for_value(self, text: str) -> str | None:
        """Find an existing key that maps to the given text value."""
        for key, value in self._existing_keys.items():
            if value == text:
                return key
        return None

    def _extract_from_file(self, filepath: str) -> list[str]:
        """Extract user-facing strings from a single Python file using AST."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read file %s: %s", filepath, e)
            return []

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            logger.warning("Syntax error in %s: %s", filepath, e)
            return []

        strings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            extracted = self._extract_from_call(node)
            if extracted:
                strings.extend(extracted)
        return strings

    def _extract_from_call(self, node: ast.Call) -> list[str]:
        """Extract string arguments from a Call node if it matches target patterns."""
        results = []

        func = node.func

        # Pattern: obj.method(...) where method is a setter pattern
        if isinstance(func, ast.Attribute):
            method_name = func.attr

            # Check for QMessageBox.information(...), QMessageBox.warning(...), etc.
            if (
                method_name in _QMESSAGEBOX_METHODS
                and isinstance(func.value, ast.Name)
                and func.value.id == "QMessageBox"
            ):
                # QMessageBox static methods: (parent, title, text, ...)
                # Extract title (arg index 1) and text (arg index 2)
                for idx in (1, 2):
                    if len(node.args) > idx:
                        s = self._get_string_value(node.args[idx])
                        if s:
                            results.append(s)
                return results

            # Check for setter patterns: obj.setText("..."), obj.setTitle("..."), etc.
            if method_name in _SETTER_PATTERNS:
                if method_name == "addItem":
                    # addItem first string arg
                    if node.args:
                        s = self._get_string_value(node.args[0])
                        if s:
                            results.append(s)
                else:
                    # First argument is the string
                    if node.args:
                        s = self._get_string_value(node.args[0])
                        if s:
                            results.append(s)
                return results

            # Check for addTab(widget, "label")
            if method_name == "addTab" and len(node.args) >= 2:
                s = self._get_string_value(node.args[1])
                if s:
                    results.append(s)
                return results

        # Pattern: QLabel("..."), QPushButton("..."), QGroupBox("...")
        if isinstance(func, ast.Name) and func.id in _WIDGET_PATTERNS:
            if node.args:
                s = self._get_string_value(node.args[0])
                if s:
                    results.append(s)
            return results

        return results

    @staticmethod
    def _get_string_value(node: ast.expr) -> str | None:
        """Extract a string literal value from an AST node, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value:
                return value
        return None

    def generate_key(self, text: str) -> str:
        """Generate a stable, deterministic key from string content.

        Algorithm:
        1. Lowercase the string.
        2. Replace non-alphanumeric characters with underscores.
        3. Collapse consecutive underscores.
        4. Strip leading/trailing underscores.
        5. Truncate to 60 characters.
        6. If the result collides with an existing key, append a numeric suffix (_2, _3, etc.).
        """
        key = text.lower()
        key = re.sub(r"[^a-z0-9]", "_", key)
        key = re.sub(r"_+", "_", key)
        key = key.strip("_")
        key = key[:60]

        # Handle empty key edge case (e.g., string was all special chars)
        if not key:
            key = "_"

        # Resolve collisions
        original_key = key
        suffix = 2
        while key in self._used_keys:
            key = f"{original_key}_{suffix}"
            suffix += 1

        self._used_keys.add(key)
        return key

    def validate_tr_calls(self) -> ValidationReport:
        """Cross-reference tr() calls in source files against English locale file keys.

        Scans all Python files in source_dirs for tr("key") calls using AST,
        then compares found keys against the English locale file to identify:
        - missing_keys: keys used in tr() calls but not in the locale file
        - orphaned_keys: keys in the locale file but not referenced by any tr() call
        """
        referenced_keys: dict[str, list[tuple[str, int]]] = {}  # key -> [(file_path, line_number)]

        for source_dir in self.source_dirs:
            for root, _dirs, files in os.walk(source_dir):
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    filepath = os.path.join(root, filename)
                    calls = self._extract_tr_calls(filepath)
                    for key, line_number in calls:
                        if key not in referenced_keys:
                            referenced_keys[key] = []
                        referenced_keys[key].append((filepath, line_number))

        locale_keys = set(self._existing_keys.keys())
        referenced_key_set = set(referenced_keys.keys())

        # Missing keys: referenced in tr() but not in locale file
        missing_keys: list[tuple[str, str, int]] = []
        for key in sorted(referenced_key_set - locale_keys):
            for file_path, line_number in referenced_keys[key]:
                missing_keys.append((key, file_path, line_number))

        # Orphaned keys: in locale file but not referenced by any tr() call
        orphaned_keys = sorted(locale_keys - referenced_key_set)

        return ValidationReport(
            missing_keys=missing_keys,
            orphaned_keys=orphaned_keys,
        )

    def _extract_tr_calls(self, filepath: str) -> list[tuple[str, int]]:
        """Extract (key, line_number) pairs from tr("key") calls in a Python file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read file %s: %s", filepath, e)
            return []

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            logger.warning("Syntax error in %s: %s", filepath, e)
            return []

        results: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match tr("key") — either as a bare function or as obj.tr("key")
            func = node.func
            is_tr = False
            if isinstance(func, ast.Name) and func.id == "tr":
                is_tr = True
            elif isinstance(func, ast.Attribute) and func.attr == "tr":
                is_tr = True

            if is_tr and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    key = arg.value.strip()
                    if key:
                        results.append((key, node.lineno))

        return results

    def update_locale_file(self, result: ExtractionResult) -> None:
        """Merge extracted strings into the English locale file, preserving existing keys."""
        # Load existing data
        data = {"_metadata": {}, "translations": {}}
        if os.path.exists(self.locale_file):
            try:
                with open(self.locale_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read locale file %s: %s", self.locale_file, e)

        translations = data.get("translations", {})

        # Merge new strings (existing keys are preserved, only new ones added)
        for key, value in result.new_strings.items():
            # Only add if key doesn't already exist (flat key check)
            if key not in translations:
                translations[key] = value

        data["translations"] = translations

        # Update metadata key count
        flat_keys = self._flatten_translations(translations)
        if "_metadata" in data:
            data["_metadata"]["total_keys"] = len(flat_keys)

        # Write with 2-space indent, UTF-8, ensure_ascii=False
        os.makedirs(os.path.dirname(self.locale_file), exist_ok=True)
        with open(self.locale_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
