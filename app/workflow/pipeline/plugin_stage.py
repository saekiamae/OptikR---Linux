"""
Plugin-aware pipeline stage wrapper.

``PluginAwareStage`` wraps any ``PipelineStageProtocol``-conforming stage
with pre-execution and post-execution plugin hooks.  If a pre-plugin sets
``skip_processing`` in the data dict the inner stage is bypassed entirely
(short-circuit), which is how frame-skip optimisation works.

The wrapper itself conforms to ``PipelineStageProtocol`` so it is
transparent to strategies and the ``BasePipeline`` frame loop.

Requirements: 2.3
"""
import logging
import time
from typing import Any

from .types import PipelineStageProtocol, StageResult

logger = logging.getLogger('optikr.pipeline.plugin_stage')


class PluginAwareStage:
    """Decorator that adds pre/post plugin hooks around a pipeline stage.

    Parameters
    ----------
    stage:
        The inner stage whose ``execute`` / ``cleanup`` are delegated to.
    pre_plugins:
        Plugins whose ``.process(data)`` runs *before* the stage.  If any
        plugin sets ``data["skip_processing"] = True`` the inner stage is
        skipped and a successful result is returned immediately.
    post_plugins:
        Plugins whose ``.process(data)`` runs *after* a successful stage
        execution.
    name:
        Optional override for the stage name used in logging and strategy
        resolution.  Falls back to the inner stage's ``name`` attribute or
        its class name.
    """

    def __init__(
        self,
        stage: PipelineStageProtocol,
        pre_plugins: list[Any] | None = None,
        post_plugins: list[Any] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._stage = stage
        self._pre_plugins: list[Any] = list(pre_plugins) if pre_plugins else []
        self._post_plugins: list[Any] = list(post_plugins) if post_plugins else []
        self.name: str = name or getattr(stage, "name", type(stage).__name__)

    # -- PipelineStageProtocol -----------------------------------------------

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Run pre-plugins, the inner stage, then post-plugins."""
        data = input_data
        total_plugin_ms = 0.0

        # --- Pre-plugins ---
        for plugin in self._pre_plugins:
            plugin_name = type(plugin).__name__
            try:
                t0 = time.perf_counter()
                data = plugin.process(data)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_plugin_ms += elapsed_ms
            except Exception as exc:
                logger.warning(
                    "Pre-plugin %s failed on stage %s: %s",
                    plugin_name,
                    self.name,
                    exc,
                )
                continue

            skip = data.get("skip_processing", False)
            logger.debug(
                "[%s] pre-plugin %s  %.1fms  skip=%s",
                self.name, plugin_name, elapsed_ms, skip,
            )

            if skip:
                return StageResult(
                    success=True,
                    data=data,
                    duration_ms=total_plugin_ms,
                )

        # --- Inner stage ---
        result = self._stage.execute(data)
        logger.debug(
            "[%s] inner stage  %.1fms  success=%s",
            self.name, result.duration_ms, result.success,
        )
        if not result.success:
            return result

        # --- Post-plugins ---
        post_data = result.data
        for plugin in self._post_plugins:
            plugin_name = type(plugin).__name__
            try:
                t0 = time.perf_counter()
                post_data = plugin.process(post_data)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_plugin_ms += elapsed_ms
            except Exception as exc:
                logger.warning(
                    "Post-plugin %s failed on stage %s: %s",
                    plugin_name,
                    self.name,
                    exc,
                )
                continue

            logger.debug(
                "[%s] post-plugin %s  %.1fms",
                self.name, plugin_name, elapsed_ms,
            )

        return StageResult(
            success=True,
            data=post_data,
            duration_ms=result.duration_ms + total_plugin_ms,
        )

    def cleanup(self) -> None:
        """Clean up plugins (reverse order) then the inner stage."""
        for plugin in reversed(self._post_plugins):
            _safe_cleanup(plugin, self.name)
        for plugin in reversed(self._pre_plugins):
            _safe_cleanup(plugin, self.name)
        self._stage.cleanup()

    # -- Introspection -------------------------------------------------------

    @property
    def inner_stage(self) -> PipelineStageProtocol:
        """Return the unwrapped inner stage."""
        return self._stage

    @property
    def pre_plugins(self) -> list[Any]:
        return list(self._pre_plugins)

    @property
    def post_plugins(self) -> list[Any]:
        return list(self._post_plugins)

    def __repr__(self) -> str:
        return (
            f"PluginAwareStage(name={self.name!r}, "
            f"pre={len(self._pre_plugins)}, post={len(self._post_plugins)})"
        )


def _safe_cleanup(plugin: Any, stage_name: str) -> None:
    """Call cleanup/reset on a plugin, swallowing exceptions."""
    try:
        if hasattr(plugin, "cleanup"):
            plugin.cleanup()
        elif hasattr(plugin, "reset"):
            plugin.reset()
    except Exception as exc:
        logger.warning(
            "Plugin %s cleanup failed (stage %s): %s",
            type(plugin).__name__,
            stage_name,
            exc,
        )
