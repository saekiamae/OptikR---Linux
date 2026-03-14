"""
Voice Manager for Audio Translation Plugin

Manages TTS voice selection, custom voice imports, and voice packs.
Supports:
- System voices (pyttsx3 SAPI5/espeak voices)
- Coqui TTS models (natural-sounding neural voices)
- Custom voice imports (.wav/.mp3 reference files for voice cloning)
- Voice packs from the internet (.zip bundles with model + config)
"""

import os
import json
import shutil
import zipfile
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Where user voice data lives
VOICE_DATA_DIR = os.path.join("user_data", "voices")
CUSTOM_VOICES_DIR = os.path.join(VOICE_DATA_DIR, "custom")
VOICE_PACKS_DIR = os.path.join(VOICE_DATA_DIR, "packs")
VOICE_REGISTRY_FILE = os.path.join(VOICE_DATA_DIR, "registry.json")


def _ensure_dirs():
    """Create voice directories if they don't exist."""
    for d in (VOICE_DATA_DIR, CUSTOM_VOICES_DIR, VOICE_PACKS_DIR):
        os.makedirs(d, exist_ok=True)


def _load_registry() -> dict[str, Any]:
    """Load the voice registry (tracks installed voices)."""
    _ensure_dirs()
    if os.path.exists(VOICE_REGISTRY_FILE):
        try:
            with open(VOICE_REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"custom_voices": [], "voice_packs": []}


def _save_registry(registry: dict[str, Any]):
    """Persist the voice registry."""
    _ensure_dirs()
    with open(VOICE_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# =========================================================================
# System voice discovery
# =========================================================================

def get_system_voices() -> list[dict[str, str]]:
    """
    List voices available through pyttsx3 (SAPI5 on Windows, espeak on Linux).
    Returns list of {id, name, language, engine} dicts.
    """
    voices = []
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            voices.append({
                "id": v.id,
                "name": v.name,
                "language": getattr(v, "languages", [""])[0] if getattr(v, "languages", None) else "",
                "engine": "pyttsx3",
                "type": "system",
            })
        engine.stop()
    except Exception as e:
        logger.debug(f"[VOICE_MANAGER] pyttsx3 voices unavailable: {e}")
    return voices


def get_coqui_models() -> list[dict[str, str]]:
    """
    List available Coqui TTS models (high-quality neural voices).
    Only returns models already downloaded locally to avoid long waits.
    """
    models = []
    try:
        from TTS.api import TTS
        tts = TTS()
        for model_name in tts.list_models():
            models.append({
                "id": model_name,
                "name": model_name.split("/")[-1].replace("_", " ").title(),
                "language": _extract_lang_from_model(model_name),
                "engine": "coqui",
                "type": "neural",
            })
    except Exception as e:
        logger.debug(f"[VOICE_MANAGER] Coqui models unavailable: {e}")
    return models


def _extract_lang_from_model(model_name: str) -> str:
    """Extract language hint from a Coqui model name."""
    parts = model_name.split("/")
    for p in parts:
        if p in ("en", "de", "fr", "es", "ja", "zh", "ko", "ru", "it", "pt",
                  "multilingual", "multi-dataset"):
            return p
    return "multilingual"


# =========================================================================
# Custom voice import (reference audio for voice cloning)
# =========================================================================

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}


def import_custom_voice(file_path: str, voice_name: str) -> dict[str, str] | None:
    """
    Import a custom voice reference file for Coqui voice cloning.
    The file should be a clean recording of the desired voice (10-30s ideal).

    Returns the voice entry dict on success, None on failure.
    """
    _ensure_dirs()
    src = Path(file_path)

    if not src.exists():
        logger.error(f"[VOICE_MANAGER] File not found: {file_path}")
        return None

    if src.suffix.lower() not in ALLOWED_AUDIO_EXTS:
        logger.error(f"[VOICE_MANAGER] Unsupported format: {src.suffix}. Use {ALLOWED_AUDIO_EXTS}")
        return None

    # Sanitize name for filesystem
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in voice_name).strip()
    if not safe_name:
        safe_name = src.stem

    dest_dir = os.path.join(CUSTOM_VOICES_DIR, safe_name)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, f"reference{src.suffix.lower()}")
    shutil.copy2(str(src), dest_file)

    entry = {
        "id": f"custom:{safe_name}",
        "name": voice_name,
        "reference_file": dest_file,
        "engine": "coqui_clone",
        "type": "custom",
    }

    registry = _load_registry()
    # Replace if same id exists
    registry["custom_voices"] = [
        v for v in registry["custom_voices"] if v["id"] != entry["id"]
    ]
    registry["custom_voices"].append(entry)
    _save_registry(registry)

    logger.info(f"[VOICE_MANAGER] Imported custom voice '{voice_name}' from {file_path}")
    return entry


def remove_custom_voice(voice_id: str) -> bool:
    """Remove a previously imported custom voice."""
    registry = _load_registry()
    voice = next((v for v in registry["custom_voices"] if v["id"] == voice_id), None)
    if not voice:
        return False

    # Remove files
    ref = voice.get("reference_file", "")
    if ref:
        ref_dir = os.path.dirname(ref)
        if os.path.isdir(ref_dir) and ref_dir.startswith(CUSTOM_VOICES_DIR):
            shutil.rmtree(ref_dir, ignore_errors=True)

    registry["custom_voices"] = [v for v in registry["custom_voices"] if v["id"] != voice_id]
    _save_registry(registry)
    logger.info(f"[VOICE_MANAGER] Removed custom voice: {voice_id}")
    return True


def get_custom_voices() -> list[dict[str, str]]:
    """Return all imported custom voices."""
    return _load_registry().get("custom_voices", [])


# =========================================================================
# Voice pack import (zip bundles from the internet)
# =========================================================================

def import_voice_pack(zip_path: str) -> dict[str, Any] | None:
    """
    Import a voice pack (.zip) containing a TTS model + config.

    Expected zip structure:
        voice_pack.zip/
            manifest.json   — {name, description, engine, language, model_file, config_file}
            model.pth       — model weights
            config.json     — model config
            (optional) reference.wav — sample audio

    Returns the pack entry dict on success, None on failure.
    """
    _ensure_dirs()
    src = Path(zip_path)

    if not src.exists() or src.suffix.lower() != ".zip":
        logger.error(f"[VOICE_MANAGER] Not a valid zip: {zip_path}")
        return None

    try:
        with zipfile.ZipFile(str(src), "r") as zf:
            names = zf.namelist()
            # Look for manifest
            manifest_name = next((n for n in names if n.endswith("manifest.json")), None)
            if not manifest_name:
                logger.error("[VOICE_MANAGER] Voice pack missing manifest.json")
                return None

            manifest = json.loads(zf.read(manifest_name))
            pack_name = manifest.get("name", src.stem)
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in pack_name).strip()

            dest_dir = os.path.join(VOICE_PACKS_DIR, safe_name)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)

            zf.extractall(dest_dir)

        entry = {
            "id": f"pack:{safe_name}",
            "name": pack_name,
            "description": manifest.get("description", ""),
            "language": manifest.get("language", "multilingual"),
            "engine": manifest.get("engine", "coqui"),
            "type": "voice_pack",
            "path": dest_dir,
            "manifest": manifest,
        }

        registry = _load_registry()
        registry["voice_packs"] = [
            p for p in registry["voice_packs"] if p["id"] != entry["id"]
        ]
        registry["voice_packs"].append(entry)
        _save_registry(registry)

        logger.info(f"[VOICE_MANAGER] Imported voice pack '{pack_name}' from {zip_path}")
        return entry

    except Exception as e:
        logger.error(f"[VOICE_MANAGER] Failed to import voice pack: {e}")
        return None


def remove_voice_pack(pack_id: str) -> bool:
    """Remove an installed voice pack."""
    registry = _load_registry()
    pack = next((p for p in registry["voice_packs"] if p["id"] == pack_id), None)
    if not pack:
        return False

    pack_dir = pack.get("path", "")
    if pack_dir and os.path.isdir(pack_dir) and pack_dir.startswith(VOICE_PACKS_DIR):
        shutil.rmtree(pack_dir, ignore_errors=True)

    registry["voice_packs"] = [p for p in registry["voice_packs"] if p["id"] != pack_id]
    _save_registry(registry)
    logger.info(f"[VOICE_MANAGER] Removed voice pack: {pack_id}")
    return True


def get_voice_packs() -> list[dict[str, Any]]:
    """Return all installed voice packs."""
    return _load_registry().get("voice_packs", [])


# =========================================================================
# Unified voice listing
# =========================================================================

def get_all_voices() -> list[dict[str, Any]]:
    """
    Return every available voice across all engines, grouped by type.
    Order: neural (Coqui) first, then custom clones, voice packs, system voices last.
    """
    voices = []
    voices.extend(get_coqui_models())
    voices.extend(get_custom_voices())
    voices.extend(get_voice_packs())
    voices.extend(get_system_voices())
    return voices
