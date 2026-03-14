# Context Manager Plugin (v2)

## Overview

The Context Manager is a domain-aware translation intelligence layer. It provides
pre- and post-translation processing through structured JSON profiles that contain
locked terms, translation memory, regex rules, and formatting controls.

Instead of simple presets that tweak config values, context profiles actively
participate in the translation pipeline — masking terms before Marian processes
them and restoring/formatting results afterward.

## How It Works

```
OCR Text
  ↓
Context Pre-Process (mask locked terms, apply regex rules)
  ↓
Smart Dictionary lookup
  ↓
Marian Translation
  ↓
Context Post-Process (restore placeholders, apply formatting)
  ↓
Spell Checker
  ↓
Overlay
```

Context profiles are **compiled once** when the user presses Start and remain
frozen for the session. Changing profiles requires Stop → Start.

## Profile JSON Structure

```json
{
  "schema_version": "2.0",
  "meta": {
    "name": "One Piece",
    "version": "1.0",
    "category": "Media",
    "description": "Context pack for One Piece manga/anime",
    "source_language": "ja",
    "target_language": "en"
  },
  "global": {
    "locked_terms": [
      {
        "source": "ルフィ",
        "target": "Luffy",
        "type": "character",
        "case_sensitive": true,
        "priority": 100,
        "notes": "Main protagonist"
      }
    ],
    "translation_memory": [
      {
        "source": "海賊王",
        "target": "Pirate King",
        "priority": 80,
        "notes": ""
      }
    ],
    "regex_rules": [
      {
        "pattern": "「(.*?)」",
        "action": "preserve",
        "replacement": "",
        "stage": "pre",
        "enabled": true,
        "description": "Preserve Japanese bracket content"
      }
    ],
    "formatting_rules": {
      "preserve_honorifics": true,
      "attack_uppercase": true,
      "translate_sound_effects": false,
      "preserve_brackets": true,
      "language_style": "casual"
    }
  },
  "stages": {
    "JA-EN": {
      "locked_terms": [],
      "translation_memory": [],
      "regex_rules": [],
      "formatting_rules": {}
    }
  }
}
```

## Key Concepts

- **Locked Terms**: Hard overrides that are masked before translation and restored
  after. Marian never sees them. Use for character names, locations, organizations.
- **Translation Memory**: Softer suggestions per stage. Can be used by Smart
  Dictionary integration.
- **Regex Rules**: Pattern-based text transformations (preserve, replace, mask)
  applied pre or post translation.
- **Formatting Rules**: Style controls applied after translation (honorifics,
  uppercase attacks, SFX handling).
- **Stages**: Per-language-pair overrides (e.g. `JA-EN`, `EN-DE`) for multi-hop
  translation chains. Falls back to `global` when no stage match.

## Import / Export

Profiles are plain JSON files stored in `user_data/context_profiles/`.

- **Export**: Copy a profile to any location for sharing
- **Import**: Load a JSON file into the profiles directory
- **Edit externally**: Open in any text editor, re-import or just edit in place
- **AI-expandable**: Hand the JSON to an AI to add more terms/rules

## Usage

```python
plugin = context_manager_plugin

# Load and compile
plugin.load_profile("One Piece")
plugin.compile()

# During translation
processed_text, restore_map = plugin.pre_process(ocr_text, "ja", "en")
translated = marian.translate(processed_text)
final = plugin.post_process(translated, restore_map, "ja", "en")

# End session
plugin.end_session()
```

## Version History

- **2.0.0**: Complete redesign
  - JSON-based context profiles replace simple presets
  - Pre/post translation processing with placeholder masking
  - Locked terms, translation memory, regex rules, formatting rules
  - Per-stage context for multi-hop translation chains
  - Import/export support
  - Profile compilation for fast runtime performance
- **pre-realese-1.0.0**: Initial release (preset-based config switching)
