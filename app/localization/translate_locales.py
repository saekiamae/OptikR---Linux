#!/usr/bin/env python3
"""
translate_locales.py — Batch-translate missing locale keys using Google Translate (free).

Uses en.json as the reference and fills in missing/untranslated keys for all other locales.

Usage:
    python app/localization/translate_locales.py                     # translate all locales
    python app/localization/translate_locales.py --lang fr it        # only French and Italian
    python app/localization/translate_locales.py --dry-run           # preview without writing
    python app/localization/translate_locales.py --retranslate       # also retranslate keys whose value == English
    python app/localization/translate_locales.py --batch-size 30     # adjust batch size (default 25)

Requires: pip install deep-translator
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("ERROR: deep-translator not installed. Run:  pip install deep-translator")
    sys.exit(1)


LOCALES_DIR = Path(__file__).parent / "locales"

# Google Translate language codes (map our locale filenames to GT codes)
LANG_MAP = {
    "de": "de",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
}

# Regex to match emoji / special prefix patterns we want to preserve
EMOJI_PREFIX_RE = re.compile(r"^([\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u25A0-\u25FF\u2700-\u27BF\u2300-\u23FF\u2190-\u21FF\u2022\u25CF\u25CB\u2713\u2717\u2716\u2714\u2611\u2610\u2B06\u2B07\u23F8\u23F9\u23FA\u25B6\u25C0\u23ED\u23EE\u23EF\u270F\u2702\u2728\u2764\u2705\u274C\u26A0\u2699\u2696\u269B\u267B\u2615\u231A\u231B\u23F0\u23F1\u23F2\u23F3\u2934\u2935\u25AA\u25AB\u25FE\u25FD\u25FC\u25FB\u2B1B\u2B1C\U0001F170-\U0001F19A\U0001F1E0-\U0001F1FF]+[\uFE0F]?\s*)")

# Keys whose values should never be translated (numbers, technical strings, etc.)
SKIP_VALUE_RE = re.compile(
    r"^[\d\.\,\s%×]+$"           # pure numbers / percentages
    r"|^--.+$"                    # placeholder like "-- FPS"
    r"|^━.+$"                     # separator lines
    r"|^●.+$"                     # bullet markers
    r"|^■.+$"                     # block markers
    r"|^♦.+$"                     # diamond markers
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {path.name}")


def should_skip_value(value: str) -> bool:
    """Return True if the value is technical / numeric and shouldn't be translated."""
    stripped = value.strip()
    if SKIP_VALUE_RE.match(stripped):
        return True
    # Very short values that are just symbols
    if len(stripped) <= 2 and not stripped.isalpha():
        return True
    return False


def split_emoji_prefix(text: str) -> tuple[str, str]:
    """Split leading emoji/symbol prefix from the translatable body."""
    m = EMOJI_PREFIX_RE.match(text)
    if m:
        return m.group(0), text[m.end():]
    return "", text


def translate_batch(texts: list[str], src: str, tgt: str, translator: GoogleTranslator) -> list[str]:
    """Translate a list of texts, preserving emoji prefixes."""
    if not texts:
        return []

    prefixes = []
    bodies = []
    for t in texts:
        prefix, body = split_emoji_prefix(t)
        prefixes.append(prefix)
        bodies.append(body)

    # Google Translate can handle batches via newline-joined text
    # but deep_translator's translate_batch is more reliable
    try:
        translated = translator.translate_batch(bodies)
    except Exception as e:
        print(f"    ⚠ Batch translation failed ({e}), falling back to one-by-one...")
        translated = []
        for body in bodies:
            try:
                translated.append(translator.translate(body))
            except Exception:
                translated.append(body)  # keep original on failure

    # Reassemble with emoji prefixes
    results = []
    for prefix, orig_body, trans_body in zip(prefixes, bodies, translated):
        if trans_body is None:
            trans_body = orig_body
        results.append(f"{prefix}{trans_body}")

    return results


def find_missing_keys(en_translations: dict, lang_translations: dict, retranslate: bool = False) -> dict[str, str]:
    """
    Find keys that need translation.
    Returns dict of {key: english_value} for keys that are missing or untranslated.
    """
    missing = {}
    for key, en_value in en_translations.items():
        if key not in lang_translations:
            missing[key] = en_value
        elif retranslate and lang_translations[key] == en_value:
            # Value is identical to English — likely untranslated
            if not should_skip_value(en_value):
                missing[key] = en_value
    return missing


def translate_locale(
    lang_code: str,
    en_data: dict,
    dry_run: bool = False,
    retranslate: bool = False,
    batch_size: int = 25,
) -> int | None:
    """Translate missing keys for a single locale. Returns count of translated keys."""
    gt_code = LANG_MAP.get(lang_code)
    if not gt_code:
        print(f"  ✗ No Google Translate mapping for '{lang_code}', skipping.")
        return None

    locale_path = LOCALES_DIR / f"{lang_code}.json"
    if not locale_path.exists():
        print(f"  ✗ Locale file not found: {locale_path}")
        return None

    lang_data = load_json(locale_path)
    en_translations = en_data["translations"]
    lang_translations = lang_data.get("translations", {})

    missing = find_missing_keys(en_translations, lang_translations, retranslate)

    if not missing:
        print(f"  ✓ {lang_code}: Already complete — nothing to translate.")
        return 0

    print(f"  → {lang_code}: {len(missing)} keys to translate")

    if dry_run:
        # Show a sample of what would be translated
        sample = list(missing.items())[:10]
        for key, val in sample:
            print(f"    {key}: \"{val}\"")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        return len(missing)

    translator = GoogleTranslator(source="en", target=gt_code)

    keys = list(missing.keys())
    values = list(missing.values())
    translated_count = 0

    for i in range(0, len(values), batch_size):
        batch_keys = keys[i : i + batch_size]
        batch_values = values[i : i + batch_size]

        # Separate skippable values
        to_translate_idx = []
        to_translate_vals = []
        skip_results = {}

        for j, (k, v) in enumerate(zip(batch_keys, batch_values)):
            if should_skip_value(v):
                skip_results[j] = v  # keep as-is
            else:
                to_translate_idx.append(j)
                to_translate_vals.append(v)

        # Translate the non-skippable ones
        if to_translate_vals:
            translated_vals = translate_batch(to_translate_vals, "en", gt_code, translator)
        else:
            translated_vals = []

        # Merge results
        trans_iter = iter(translated_vals)
        for j, key in enumerate(batch_keys):
            if j in skip_results:
                lang_translations[key] = skip_results[j]
            else:
                lang_translations[key] = next(trans_iter)
            translated_count += 1

        progress = min(i + batch_size, len(values))
        print(f"    [{progress}/{len(values)}] translated...")

        # Rate limiting — be nice to the free API
        if i + batch_size < len(values):
            time.sleep(1.5)

    # Update the locale data
    lang_data["translations"] = dict(sorted(lang_translations.items()))
    lang_data["_metadata"]["total_keys"] = len(lang_data["translations"])
    lang_data["_metadata"]["translated_keys"] = len(lang_data["translations"])
    lang_data["_metadata"]["last_updated"] = time.strftime("%Y-%m-%d")

    save_json(locale_path, lang_data)
    return translated_count


def main():
    parser = argparse.ArgumentParser(
        description="Translate missing locale keys using Google Translate (free)."
    )
    parser.add_argument(
        "--lang",
        nargs="*",
        default=None,
        help="Language codes to translate (e.g. fr it ja). Default: all non-English locales.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be translated without writing files.",
    )
    parser.add_argument(
        "--retranslate",
        action="store_true",
        help="Also retranslate keys whose current value is identical to English.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of strings to translate per API call (default: 25).",
    )
    args = parser.parse_args()

    # Load English reference
    en_path = LOCALES_DIR / "en.json"
    if not en_path.exists():
        print(f"ERROR: English locale not found at {en_path}")
        sys.exit(1)

    en_data = load_json(en_path)
    en_count = len(en_data["translations"])
    print(f"Reference: en.json ({en_count} keys)\n")

    # Determine which languages to process
    if args.lang:
        lang_codes = args.lang
    else:
        lang_codes = [p.stem for p in sorted(LOCALES_DIR.glob("*.json")) if p.stem != "en"]

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")

    total_translated = 0
    for lang_code in lang_codes:
        print(f"[{lang_code.upper()}]")
        result = translate_locale(
            lang_code,
            en_data,
            dry_run=args.dry_run,
            retranslate=args.retranslate,
            batch_size=args.batch_size,
        )
        if result is not None:
            total_translated += result
        print()

    print(f"Done. {total_translated} keys {'would be' if args.dry_run else ''} translated.")


if __name__ == "__main__":
    main()
