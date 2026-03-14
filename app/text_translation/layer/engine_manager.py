"""
Engine Manager for Translation Layer.

Handles engine registration, loading via plugin manager, selection,
and fallback logic.

Requirements: 3.1
"""
import logging
import threading
from typing import Any

from app.text_translation.translation_engine_interface import (
    AbstractTranslationEngine,
    TranslationEngineRegistry,
)
from app.text_translation.translation_plugin_manager import TranslationPluginManager


class EngineManager:
    """Manages translation engine registration, loading, and selection."""

    def __init__(self, config_manager: Any | None = None) -> None:
        self._logger = logging.getLogger(__name__)
        self._engine_registry = TranslationEngineRegistry()  # type: ignore[no-untyped-call]
        self._config_manager = config_manager
        self._default_engine: str | None = None
        self._fallback_engines: list[str] = []
        self._lock = threading.RLock()

        # Plugin manager for translation engines
        self.plugin_manager = TranslationPluginManager(config_manager=config_manager)

        # Discover available translation plugins
        discovered_plugins = self.plugin_manager.discover_plugins()
        self._logger.info(f"Discovered {len(discovered_plugins)} translation plugin(s)")
        if discovered_plugins:
            plugin_names = [p.name for p in discovered_plugins]
            self._logger.info(f"Available plugins: {plugin_names}")

    # -- public properties for facade access ----------------------------------

    @property
    def engine_registry(self) -> TranslationEngineRegistry:
        return self._engine_registry

    @property
    def default_engine(self) -> str | None:
        return self._default_engine

    @property
    def fallback_engines(self) -> list[str]:
        return list(self._fallback_engines)

    def set_default_engine_unchecked(self, engine_name: str | None) -> None:
        """Set default engine without availability validation (for backward compat)."""
        with self._lock:
            self._default_engine = engine_name

    # -- registration / loading ------------------------------------------------

    def register_engine(
        self,
        engine: AbstractTranslationEngine,
        is_default: bool = False,
        is_fallback: bool = True,
    ) -> bool:
        """Register a translation engine."""
        try:
            success = self._engine_registry.register_engine(engine)
            if not success:
                return False

            with self._lock:
                if is_default or self._default_engine is None:
                    self._default_engine = engine.engine_name
                    self._logger.info(f"Set default translation engine: {engine.engine_name}")

                if is_fallback and engine.engine_name not in self._fallback_engines:
                    self._fallback_engines.append(engine.engine_name)
                    self._logger.info(f"Added fallback translation engine: {engine.engine_name}")

            return True
        except Exception as e:
            self._logger.error(f"Failed to register translation engine: {e}")
            return False

    def load_engine(self, engine_name: str, config: dict[str, Any] | None = None) -> bool:
        """Load a translation engine plugin."""
        try:
            success = self.plugin_manager.load_plugin(engine_name, config)
            if success:
                engine = self.plugin_manager.get_engine(engine_name)
                if engine:
                    return self.register_engine(engine, is_default=False, is_fallback=True)
            return False
        except Exception as e:
            self._logger.error(f"Failed to load engine plugin {engine_name}: {e}")
            return False

    # -- selection / query ----------------------------------------------------

    def set_default_engine(self, engine_name: str) -> bool:
        """Set the default translation engine."""
        with self._lock:
            engine = self._engine_registry.get_engine(engine_name)
            if not engine:
                engine = self.plugin_manager.get_engine(engine_name)

            if engine and engine.is_available():
                self._default_engine = engine_name
                self._logger.info(f"Set default engine: {engine_name}")
                return True

            self._logger.error(f"Cannot set default engine: {engine_name} not available")
            return False

    def get_available_engines(self) -> list[str]:
        """Get list of available translation engine names."""
        registry_engines = list(self._engine_registry.get_available_engines())
        plugin_engines = self.plugin_manager.get_available_engines()
        return list(set(registry_engines + plugin_engines))

    def get_engines_for_language_pair(self, src_lang: str, tgt_lang: str) -> list[str]:
        """Get engines that support specific language pair."""
        return self._engine_registry.get_engines_for_language_pair(src_lang, tgt_lang)

    def get_engine(self, engine_name: str) -> AbstractTranslationEngine | None:
        """Get engine by name from registry."""
        return self._engine_registry.get_engine(engine_name)

    def get_fallback_engine(
        self, src_lang: str, tgt_lang: str
    ) -> AbstractTranslationEngine | None:
        """Get fallback engine that supports the language pair."""
        if self._config_manager:
            fallback_enabled = self._config_manager.get_setting(
                "translation.fallback_enabled", True
            )
            if not fallback_enabled:
                self._logger.debug("Fallback engines disabled in config")
                return None

        for engine_name in self._fallback_engines:
            engine = self._engine_registry.get_engine(engine_name)
            if (
                engine
                and engine.is_available()
                and engine.supports_language_pair(src_lang, tgt_lang)
            ):
                self._logger.info(f"Using fallback engine: {engine_name}")
                return engine
        return None

    def preload_models(self, src_lang: str, tgt_lang: str) -> bool:
        """Pre-load translation models in the main thread."""
        try:
            self._logger.info(f"Pre-loading translation models for {src_lang}->{tgt_lang}...")

            if not self._default_engine:
                self._logger.warning("No default engine set for pre-loading")
                return False

            engine = self._engine_registry.get_engine(self._default_engine)
            if not engine:
                self._logger.warning(
                    f"Default engine '{self._default_engine}' not found in registry"
                )
                return False

            if hasattr(engine, "preload_model"):
                success: bool = engine.preload_model(src_lang, tgt_lang)
                if success:
                    self._logger.info("Models pre-loaded successfully")
                else:
                    self._logger.warning("Failed to pre-load models")
                return success
            else:
                self._logger.info(
                    f"Engine {engine.engine_name} doesn't require pre-loading"
                )
                return True

        except Exception as e:
            self._logger.error(f"Error pre-loading models: {e}", exc_info=True)
            return False

    def unload_engine(self, engine_name: str) -> bool:
        """Unload a single translation engine and free its resources.

        Unregisters from both the engine registry (calls ``cleanup()``)
        and the plugin registry, then releases GPU memory.

        Args:
            engine_name: Name of engine to unload

        Returns:
            True if the engine was found and unloaded
        """
        unloaded = False

        with self._lock:
            if self._engine_registry.get_engine(engine_name):
                self._engine_registry.unregister_engine(engine_name)
                unloaded = True

            if engine_name in self._fallback_engines:
                self._fallback_engines.remove(engine_name)

            if self._default_engine == engine_name:
                self._default_engine = None

        self.plugin_manager.unload_plugin(engine_name)

        if unloaded:
            self._logger.info("Unloaded translation engine: %s", engine_name)
        return unloaded

    def cleanup(self) -> None:
        """Clean up all registered engines."""
        self._engine_registry.cleanup_all()
