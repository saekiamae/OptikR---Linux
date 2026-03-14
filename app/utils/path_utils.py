"""
Path utilities for handling both Python script and EXE execution.

This module is the SINGLE SOURCE OF TRUTH for all application paths.
Every module that needs a filesystem path must use these functions —
no hardcoded path strings anywhere else in the codebase.

Directory structure (Phase 2):
    <app_root>/
    ├── user_data/                  # User-generated content
    │   ├── config/                 # Configuration files
    │   ├── learned/translations/   # Learned dictionary files
    │   ├── exports/                # Exported data (translations, screenshots, logs)
    │   ├── custom_plugins/         # User-installed plugins
    │   ├── context_profiles/       # Context manager profiles
    │   ├── image_processing_presets/  # Image processing presets
    │   └── backups/                # Config backups
    ├── system_data/                # System-managed content
    │   ├── ai_models/              # model_registry.json only; model files live in HF cache
    │   ├── cache/                  # Translation/OCR cache files
    │   ├── logs/                   # Application logs
    │   └── temp/                   # Temporary processing files
    └── plugins/                    # Built-in plugins
        ├── stages/
        │   ├── capture/            # Capture plugins (bettercam, screenshot)
        │   ├── ocr/                # OCR plugins (easyocr, tesseract, paddleocr, etc.)
        │   └── translation/        # Translation plugins (marianmt, libretranslate, etc.)
        └── enhancers/
            ├── optimizers/         # Pipeline optimizer plugins
            └── text_processors/    # Text processing plugins
"""

import sys
from pathlib import Path

# Cached app root — resolved once, reused forever
_app_root: Path | None = None


def get_app_root() -> Path:
    """
    Get the application root directory (cached).

    Works correctly for both:
    - Python script execution: Returns the directory containing run.py
    - EXE execution: Returns the directory containing the .exe file
    """
    global _app_root
    if _app_root is None:
        if getattr(sys, 'frozen', False):
            # Running as compiled EXE (PyInstaller, cx_Freeze, etc.)
            _app_root = Path(sys.executable).parent.resolve()
        else:
            # Running as Python script
            # Go up from app/utils/ to project root (where run.py is)
            _app_root = Path(__file__).parent.parent.parent.resolve()
    return _app_root


def get_app_path(*parts: str) -> Path:
    """
    Get a path relative to the application root.

    This is the low-level building block. Prefer the specific helpers below
    for standard directories (config, dictionary, logs, etc.).

    Args:
        *parts: Path components to join

    Returns:
        Path: Absolute path relative to application root
    """
    return get_app_root().joinpath(*parts)


def ensure_app_directory(*parts: str) -> Path:
    """
    Ensure a directory exists relative to the application root.

    Args:
        *parts: Path components to join

    Returns:
        Path: Absolute path to the (created) directory
    """
    directory = get_app_path(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ============================================================================
# CANONICAL PATH REGISTRY
# Single source of truth for all application directories.
# Keys are logical names, values are relative paths from app root.
# ============================================================================

PATHS: dict[str, str] = {
    # User data (user-generated, portable)
    "config":               "user_data/config",
    "dictionary":           "user_data/learned/translations",
    "exports":              "user_data/exports",
    "exports_translations": "user_data/exports/translations",
    "exports_screenshots":  "user_data/exports/screenshots",
    "exports_logs":         "user_data/exports/logs",
    "custom_plugins":       "user_data/custom_plugins",
    "context_profiles":     "user_data/context_profiles",
    "image_processing_presets": "user_data/image_processing_presets",
    "backups":              "user_data/backups",
    "benchmarks":           "user_data/benchmarks",

    # System data (managed, regenerable)
    "models":               "system_data/ai_models",
    "cache":                "system_data/cache",
    "logs":                 "system_data/logs",
    "temp":                 "system_data/temp",
    "temp_processing":      "system_data/temp/processing",
    "temp_downloads":       "system_data/temp/downloads",

    # Plugins
    "plugins":                          "plugins",
    "plugins_stages":                   "plugins/stages",
    "plugins_stages_capture":           "plugins/stages/capture",
    "plugins_stages_ocr":               "plugins/stages/ocr",
    "plugins_stages_translation":       "plugins/stages/translation",
    "plugins_enhancers":                "plugins/enhancers",
    "plugins_enhancers_optimizers":     "plugins/enhancers/optimizers",
    "plugins_enhancers_text_processors":"plugins/enhancers/text_processors",
}


# ============================================================================
# DIRECTORY ACCESSORS
# ============================================================================

def get_dir(name: str) -> Path:
    """
    Get the absolute path for a named directory from the PATHS registry.

    Args:
        name: Logical directory name (key in PATHS dict)

    Returns:
        Path: Absolute path to the directory

    Raises:
        KeyError: If name is not in the PATHS registry

    Examples:
        get_dir("dictionary")  # -> <app_root>/user_data/learned/translations
        get_dir("logs")        # -> <app_root>/system_data/logs
        get_dir("config")      # -> <app_root>/user_data/config
    """
    if name not in PATHS:
        raise KeyError(f"Unknown path name: {name!r}. Valid names: {sorted(PATHS.keys())}")
    return get_app_path(PATHS[name])


def ensure_dir(name: str) -> Path:
    """
    Like get_dir(), but creates the directory if it doesn't exist.
    Always creates at the new (Phase 2) location.

    Args:
        name: Logical directory name (key in PATHS dict)

    Returns:
        Path: Absolute path to the (created) directory
    """
    if name not in PATHS:
        raise KeyError(f"Unknown path name: {name!r}. Valid names: {sorted(PATHS.keys())}")
    directory = get_app_path(PATHS[name])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ============================================================================
# CONVENIENCE ACCESSORS (typed, discoverable, IDE-friendly)
# ============================================================================

def get_config_dir() -> Path:
    """Get config directory: user_data/config/"""
    return get_dir("config")


def get_dictionary_dir() -> Path:
    """Get learned translations directory: user_data/learned/translations/"""
    return get_dir("dictionary")


def get_logs_dir() -> Path:
    """Get logs directory: system_data/logs/"""
    return get_dir("logs")


def get_cache_dir() -> Path:
    """Get cache directory: system_data/cache/"""
    return get_dir("cache")


def get_models_dir() -> Path:
    """
    Get AI models registry directory (``system_data/ai_models/``).

    This directory holds only ``model_registry.json``.  Actual model files
    are managed by HuggingFace in its default cache (see
    :func:`get_hf_cache_dir`).
    """
    return get_dir("models")


def get_hf_cache_dir() -> Path:
    """
    Return the HuggingFace Hub cache directory where model files are stored.

    Respects the ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` env-vars that
    HuggingFace itself honours; falls back to ``~/.cache/huggingface/hub``.
    """
    import os
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def get_exports_dir(export_type: str | None = None) -> Path:
    """
    Get exports directory.

    Args:
        export_type: Optional — 'translations', 'screenshots', or 'logs'
    """
    if export_type == "translations":
        return get_dir("exports_translations")
    elif export_type == "screenshots":
        return get_dir("exports_screenshots")
    elif export_type == "logs":
        return get_dir("exports_logs")
    return get_dir("exports")


def get_backups_dir() -> Path:
    """Get backups directory: user_data/backups/"""
    return get_dir("backups")


def get_benchmarks_dir() -> Path:
    """Get benchmarks directory: user_data/benchmarks/"""
    return get_dir("benchmarks")

def get_temp_dir(temp_type: str | None = None) -> Path:
    """
    Get temp directory.

    Args:
        temp_type: Optional — 'processing' or 'downloads'
    """
    if temp_type == "processing":
        return get_dir("temp_processing")
    elif temp_type == "downloads":
        return get_dir("temp_downloads")
    return get_dir("temp")


def get_custom_plugins_dir() -> Path:
    """Get custom plugins directory: user_data/custom_plugins/"""
    return get_dir("custom_plugins")


def get_context_profiles_dir() -> Path:
    """Get context profiles directory: user_data/context_profiles/"""
    return get_dir("context_profiles")


def get_image_processing_presets_dir() -> Path:
    """Get image processing presets directory: user_data/image_processing_presets/"""
    return get_dir("image_processing_presets")


def get_plugins_dir() -> Path:
    """Get built-in plugins directory: plugins/"""
    return get_dir("plugins")


def get_plugin_stages_dir(stage_type: str | None = None) -> Path:
    """
    Get plugin stages directory.

    Args:
        stage_type: Optional — 'capture', 'ocr', or 'translation'

    Returns:
        Path: plugins/stages/ or plugins/stages/<type>/
    """
    if stage_type == "capture":
        return get_dir("plugins_stages_capture")
    elif stage_type == "ocr":
        return get_dir("plugins_stages_ocr")
    elif stage_type == "translation":
        return get_dir("plugins_stages_translation")
    return get_dir("plugins_stages")


def get_plugin_enhancers_dir(enhancer_type: str | None = None) -> Path:
    """
    Get plugin enhancers directory.

    Args:
        enhancer_type: Optional — 'optimizers' or 'text_processors'

    Returns:
        Path: plugins/enhancers/ or plugins/enhancers/<type>/
    """
    if enhancer_type == "optimizers":
        return get_dir("plugins_enhancers_optimizers")
    elif enhancer_type == "text_processors":
        return get_dir("plugins_enhancers_text_processors")
    return get_dir("plugins_enhancers")


# ============================================================================
# FILE PATH HELPERS
# For specific files within the standard directories.
# ============================================================================

def get_config_file(filename: str = "user_config.json") -> Path:
    """
    Get a config file path: user_data/config/<filename>

    Args:
        filename: Config filename (default: 'user_config.json')
    """
    return get_app_path(PATHS["config"], filename)


def get_dictionary_file(source_lang: str, target_lang: str) -> Path:
    """
    Get learned dictionary file path for a language pair.

    Args:
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'de')

    Returns:
        Path: user_data/learned/translations/<src>_<tgt>.json.gz
    """
    filename = f"{source_lang}_{target_lang}.json.gz"
    return get_app_path(PATHS["dictionary"], filename)


# ============================================================================
# STARTUP HELPERS
# ============================================================================

def ensure_all_directories() -> None:
    """
    Create all standard application directories.
    Called once at startup from run.py.
    """
    for name in PATHS:
        ensure_dir(name)


def get_all_paths() -> dict[str, Path]:
    """
    Get all registered paths as a dict of name -> absolute Path.
    Useful for diagnostics display.
    """
    return {name: get_app_path(rel) for name, rel in PATHS.items()}


def get_relative_path(name: str) -> str:
    """
    Get the relative path string for a named directory.
    Useful for display in the UI (storage tab, diagnostics).

    Args:
        name: Logical directory name (key in PATHS dict)

    Returns:
        Relative path string (e.g., 'user_data/config')
    """
    if name not in PATHS:
        raise KeyError(f"Unknown path name: {name!r}")
    return PATHS[name]
