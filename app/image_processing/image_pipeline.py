"""Single-shot image processing pipeline.

Reuses existing :class:`PreprocessingStage`, :class:`OCRStage`, and
:class:`TranslationStage` but runs them in a single sequential pass
without the :class:`BasePipeline` frame loop.  An optional
:class:`ImageCompositor` renders translated text onto the image as a
final step.

The pipeline calls ``stage.execute(data_dict)`` directly, accumulating
``result.data`` the same way :class:`SequentialStrategy` does.
"""

import logging
import time
from typing import Any

import numpy as np

from app.models import Frame
from app.workflow.pipeline.plugin_stage import PluginAwareStage
from app.workflow.pipeline.stages import (
    OCRStage,
    PreprocessingStage,
    TranslationStage,
    VisionTranslateStage,
)
from app.workflow.pipeline.types import StageResult

from .image_compositor import ImageCompositor

logger = logging.getLogger(__name__)


class ImagePipeline:
    """Simplified pipeline for translating static images.

    Chains ``PreprocessingStage`` -> ``OCRStage`` -> ``TranslationStage``
    (each optionally wrapped with plugin hooks), then feeds results into
    an :class:`ImageCompositor` to produce a translated image.

    Parameters
    ----------
    ocr_layer :
        OCR layer instance (e.g. ``OCRLayer``).  When in vision mode this
        may be *None*; use ``vision_layer`` instead.
    translation_layer :
        Translation layer instance (e.g. ``TranslationFacade``).
    vision_layer :
        Optional vision layer for vision pipeline mode.  When set, the
        pipeline uses :class:`VisionTranslateStage` instead of OCR +
        translation stages.
    config_manager :
        Application config facade for reading settings.
    preprocessing_layer :
        Optional preprocessing layer.  When *None* the preprocessing
        stage acts as a transparent pass-through.
    plugin_map :
        ``{stage_name: {"pre": [plugin, …], "post": [plugin, …]}}``
        — same format produced by
        :meth:`PipelineFactory._build_plugin_map`.
    compositor :
        Optional :class:`ImageCompositor`.  When provided, translated
        text is rendered onto the image automatically.
    """

    def __init__(
        self,
        ocr_layer: Any,
        translation_layer: Any,
        config_manager: Any = None,
        preprocessing_layer: Any = None,
        plugin_map: dict[str, dict[str, list[Any]]] | None = None,
        compositor: ImageCompositor | None = None,
        vision_layer: Any = None,
    ) -> None:
        self._ocr_layer = ocr_layer
        self._translation_layer = translation_layer
        self._vision_layer = vision_layer
        self._config_manager = config_manager
        self._preprocessing_layer = preprocessing_layer
        self._plugin_map: dict[str, dict[str, list[Any]]] = plugin_map or {}
        self._compositor = compositor

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_startup_pipeline(
        cls,
        startup_pipeline: Any,
        config_manager: Any = None,
        compositor: ImageCompositor | None = None,
    ) -> "ImagePipeline":
        """Build an :class:`ImagePipeline` from an existing
        :class:`StartupPipeline`, reusing its layers and plugin hooks.
        """
        ocr_layer = getattr(startup_pipeline, "ocr_layer", None)
        translation_layer = getattr(startup_pipeline, "translation_layer", None)
        vision_layer = getattr(startup_pipeline, "vision_layer", None)
        cm = config_manager or getattr(startup_pipeline, "config_manager", None)

        plugin_map: dict[str, dict[str, list[Any]]] = {}
        base_pipeline = getattr(startup_pipeline, "pipeline", None)
        if base_pipeline is not None:
            for stage in getattr(base_pipeline, "_stages", []):
                if isinstance(stage, PluginAwareStage):
                    plugin_map[stage.name] = {
                        "pre": list(stage.pre_plugins),
                        "post": list(stage.post_plugins),
                    }

        return cls(
            ocr_layer=ocr_layer,
            translation_layer=translation_layer,
            config_manager=cm,
            plugin_map=plugin_map,
            compositor=compositor,
            vision_layer=vision_layer,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image: np.ndarray,
        source_lang: str | None = None,
        target_lang: str | None = None,
        compositor_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a single image through OCR, translation, and compositing.

        Parameters
        ----------
        image :
            BGR ``numpy.ndarray``.
        source_lang :
            Override source language (defaults to config value).
        target_lang :
            Override target language (defaults to config value).
        compositor_config :
            Style overrides forwarded to the :class:`ImageCompositor`.

        Returns
        -------
        dict
            ``success``, ``image``, ``text_blocks``, ``translations``,
            ``source_lang``, ``target_lang``, ``error``, ``duration_ms``.
        """
        start = time.perf_counter()

        try:
            src = source_lang or self._get_setting(
                "translation.source_language", "ja",
            )
            tgt = target_lang or self._get_setting(
                "translation.target_language", "de",
            )

            if self._vision_layer is None and self._ocr_layer is None:
                elapsed = (time.perf_counter() - start) * 1000
                return self._error_result(
                    image,
                    "No pipeline available: vision mode has no vision model; text mode has no OCR.",
                    elapsed,
                )

            if self._vision_layer is not None:
                frame = Frame(
                    data=image, timestamp=time.time(), source_region=None,
                )
                current_data: dict[str, Any] = {
                    "frame": frame,
                    "source_lang": src,
                    "target_lang": tgt,
                }
                stage = VisionTranslateStage(
                    self._vision_layer,
                    source_lang=src,
                    target_lang=tgt,
                )
                try:
                    result: StageResult = stage.execute(current_data)
                except Exception as exc:
                    logger.error("VisionTranslateStage raised: %s", exc)
                    elapsed = (time.perf_counter() - start) * 1000
                    return self._error_result(
                        image, f"VisionTranslateStage: {exc}", elapsed,
                    )
                if not result.success:
                    elapsed = (time.perf_counter() - start) * 1000
                    return self._error_result(
                        image,
                        result.error or "VisionTranslateStage failed",
                        elapsed,
                    )
                current_data.update(result.data)
                text_blocks = current_data.get("text_blocks", [])
                translations = current_data.get("translations", [])
                if self._compositor and text_blocks and translations:
                    composited = self._compositor.composite(
                        image, text_blocks, translations, compositor_config,
                    )
                else:
                    composited = image.copy()
                elapsed = (time.perf_counter() - start) * 1000
                return {
                    "success": True,
                    "image": composited,
                    "text_blocks": text_blocks,
                    "translations": translations,
                    "source_lang": current_data.get("source_lang", src),
                    "target_lang": current_data.get("target_lang", tgt),
                    "error": None,
                    "duration_ms": elapsed,
                }

            stages = self._build_stages(src, tgt)

            frame = Frame(
                data=image, timestamp=time.time(), source_region=None,
            )
            current_data = {
                "frame": frame,
                "source_lang": src,
                "target_lang": tgt,
            }

            for stage in stages:
                stage_name = getattr(stage, "name", type(stage).__name__)
                try:
                    result = stage.execute(current_data)
                except Exception as exc:
                    logger.error("Stage '%s' raised: %s", stage_name, exc)
                    elapsed = (time.perf_counter() - start) * 1000
                    return self._error_result(
                        image, f"{stage_name}: {exc}", elapsed,
                    )

                if not result.success:
                    elapsed = (time.perf_counter() - start) * 1000
                    return self._error_result(
                        image,
                        result.error or f"{stage_name} failed",
                        elapsed,
                    )

                current_data.update(result.data)

            text_blocks = current_data.get("text_blocks", [])
            translations = current_data.get("translations", [])

            if self._compositor and text_blocks and translations:
                composited = self._compositor.composite(
                    image, text_blocks, translations, compositor_config,
                )
            else:
                composited = image.copy()

            elapsed = (time.perf_counter() - start) * 1000
            return {
                "success": True,
                "image": composited,
                "text_blocks": text_blocks,
                "translations": translations,
                "source_lang": current_data.get("source_lang", src),
                "target_lang": current_data.get("target_lang", tgt),
                "error": None,
                "duration_ms": elapsed,
            }

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("ImagePipeline.process_image failed: %s", exc)
            return self._error_result(image, str(exc), elapsed)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_stages(
        self,
        source_lang: str,
        target_lang: str,
    ) -> list[Any]:
        """Construct and optionally plugin-wrap the stage list."""
        confidence = self._get_setting("ocr.confidence_threshold", 0.5)
        preprocessing_on = self._get_setting("ocr.preprocessing_enabled", False)
        preprocessing_intelligent = self._get_setting(
            "ocr.preprocessing_intelligent", True,
        )

        stages: list[Any] = [
            PreprocessingStage(
                self._preprocessing_layer if preprocessing_on else None,
                enabled=preprocessing_on,
                intelligent=preprocessing_intelligent,
            ),
            OCRStage(
                self._ocr_layer,
                confidence_threshold=confidence,
                source_lang=source_lang,
            ),
            TranslationStage(
                self._translation_layer,
                source_lang=source_lang,
                target_lang=target_lang,
            ),
        ]

        if self._plugin_map:
            wrapped: list[Any] = []
            for stage in stages:
                stage_name = getattr(stage, "name", type(stage).__name__)
                hooks = self._plugin_map.get(stage_name)
                if hooks and (hooks.get("pre") or hooks.get("post")):
                    wrapped.append(
                        PluginAwareStage(
                            stage,
                            pre_plugins=hooks.get("pre"),
                            post_plugins=hooks.get("post"),
                            name=stage_name,
                        )
                    )
                else:
                    wrapped.append(stage)
            stages = wrapped

        return stages

    def _get_setting(self, key: str, default: Any) -> Any:
        if self._config_manager is not None:
            return self._config_manager.get_setting(key, default)
        return default

    @staticmethod
    def _error_result(
        image: np.ndarray,
        error: str,
        duration_ms: float,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "image": image,
            "text_blocks": [],
            "translations": [],
            "source_lang": "",
            "target_lang": "",
            "error": error,
            "duration_ms": duration_ms,
        }
