"""
Translation system for OptikR UI

New JSON-based system with support for:
- User-provided language packs
- AI-friendly translation workflow
- Hot-reloading of languages
- Automatic fallback to English
"""

# Import from new JSON translator
from .json_translator import (
    tr,
    set_language,
    get_current_language,
    get_available_languages,
    reload_languages,
    export_template,
    import_language_pack,
    init_translator,
    get_translator,
    ImportResult,
    CompletenessReport,
)

# Import language manager and translatable mixin
from .language_manager import (
    LanguageManager,
    get_language_manager
)

from .translatable_mixin import (
    TranslatableMixin
)

__all__ = [
    # Main translation system
    'tr',
    'set_language',
    'get_current_language',
    'get_available_languages',
    'reload_languages',
    'export_template',
    'import_language_pack',
    'init_translator',
    'get_translator',
    'ImportResult',
    'CompletenessReport',
    # Language manager
    'LanguageManager',
    'get_language_manager',
    'TranslatableMixin'
]
