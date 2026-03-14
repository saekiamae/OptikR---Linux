"""
Judge OCR Plugin – Meta-engine that loads N OCR engine plugins, runs them
all on the same frame, and uses configurable voting to determine the best
text for each detected region.
"""

from __future__ import annotations

import logging
import concurrent.futures
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ocr.ocr_engine_interface import (
    IOCREngine,
    OCRProcessingOptions,
    OCREngineType,
    OCREngineStatus,
)
from app.models import Frame, TextBlock, Rectangle

from . import voting

logger = logging.getLogger(__name__)


class OCREngine(IOCREngine):
    """Meta OCR engine that delegates to N sub-engines and votes on results."""

    def __init__(self, engine_name: str = "judge_ocr", engine_type=None):
        if engine_type is None:
            engine_type = OCREngineType.JUDGE_OCR
        super().__init__(engine_name, engine_type)

        self._sub_engines: dict[str, IOCREngine] = {}
        self._engine_thresholds: dict[str, float] = {}
        self._voting_strategy: str = "majority_vote"
        self._quorum_count: int = 2
        self._parallel: bool = True
        self._current_language: str = "en"
        self._plugin_manager = None

    # ------------------------------------------------------------------
    # IOCREngine interface
    # ------------------------------------------------------------------

    def initialize(self, config: dict[str, Any]) -> bool:
        engine_names: list[str] = config.get("engines", [])
        if not engine_names:
            logger.error("judge_ocr: no engines specified in config['engines']")
            self.status = OCREngineStatus.ERROR
            return False

        self.status = OCREngineStatus.INITIALIZING

        self._engine_thresholds = config.get("engine_thresholds", {})
        self._voting_strategy = config.get("voting_strategy", "majority_vote")
        self._quorum_count = config.get("quorum_count", 2)
        self._parallel = config.get("parallel_execution", True)
        self._current_language = config.get("language", "en")

        logger.info(
            "Judge OCR initialising – engines=%s, strategy=%s, quorum=%d",
            engine_names,
            self._voting_strategy,
            self._quorum_count,
        )

        self._load_sub_engines(engine_names, config)

        if not self._sub_engines:
            logger.error("judge_ocr: none of the requested engines could be loaded")
            self.status = OCREngineStatus.ERROR
            return False

        if len(self._sub_engines) == 1:
            logger.warning(
                "judge_ocr: only 1 engine loaded (%s) – operating as passthrough",
                next(iter(self._sub_engines)),
            )

        self.status = OCREngineStatus.READY
        logger.info(
            "Judge OCR ready – %d engine(s) loaded: %s",
            len(self._sub_engines),
            list(self._sub_engines.keys()),
        )
        return True

    def extract_text(
        self, frame: Frame, options: OCRProcessingOptions
    ) -> list[TextBlock]:
        if not self.is_ready():
            return []

        try:
            import numpy as np

            if not isinstance(frame.data, np.ndarray):
                logger.error("Frame data is not a numpy array")
                return []

            engine_results = self._run_all_engines(frame, options)
            if not engine_results:
                return []

            matched_groups = self._match_regions(engine_results)
            return self._vote(matched_groups)

        except Exception:
            logger.error("Judge OCR failed", exc_info=True)
            return []

    def extract_text_batch(
        self, frames: list[Frame], options: OCRProcessingOptions
    ) -> list[list[TextBlock]]:
        return [self.extract_text(f, options) for f in frames]

    def set_language(self, language: str) -> bool:
        self._current_language = language
        ok = True
        for name, engine in self._sub_engines.items():
            try:
                if not engine.set_language(language):
                    logger.warning("Engine %s failed to set language %s", name, language)
                    ok = False
            except Exception:
                logger.warning("Engine %s raised while setting language", name, exc_info=True)
                ok = False
        return ok

    def get_supported_languages(self) -> list[str]:
        if not self._sub_engines:
            return []
        langs: set[str] = set()
        for engine in self._sub_engines.values():
            try:
                langs.update(engine.get_supported_languages())
            except Exception:
                pass
        return sorted(langs)

    def cleanup(self) -> None:
        for name, engine in self._sub_engines.items():
            try:
                engine.cleanup()
                logger.info("Cleaned up sub-engine: %s", name)
            except Exception:
                logger.warning("Error cleaning up sub-engine %s", name, exc_info=True)
        self._sub_engines.clear()

        if self._plugin_manager:
            self._plugin_manager = None

        self.status = OCREngineStatus.UNINITIALIZED
        logger.info("Judge OCR engine cleaned up")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_sub_engines(
        self, engine_names: list[str], parent_config: dict[str, Any]
    ) -> None:
        """
        Create a private OCRPluginManager, discover plugins under
        ``plugins/stages/ocr/``, and load each requested engine.
        """
        from app.ocr.ocr_plugin_manager import OCRPluginManager

        ocr_plugins_dir = str(Path(__file__).parent.parent)
        self._plugin_manager = OCRPluginManager(
            plugin_directories=[ocr_plugins_dir]
        )
        self._plugin_manager.discover_plugins()

        for name in engine_names:
            if name == "judge_ocr":
                logger.warning("judge_ocr cannot load itself as a sub-engine – skipping")
                continue
            try:
                engine_config: dict[str, Any] = {
                    "language": parent_config.get("language", "en"),
                    "gpu": parent_config.get("gpu", True),
                }
                success = self._plugin_manager.load_plugin(name, engine_config)
                if success:
                    engine = self._plugin_manager.get_engine(name)
                    if engine and engine.is_ready():
                        self._sub_engines[name] = engine
                        logger.info("Loaded sub-engine: %s", name)
                    else:
                        logger.warning("Sub-engine %s loaded but not ready", name)
                else:
                    logger.warning("Failed to load sub-engine: %s", name)
            except Exception:
                logger.warning("Exception loading sub-engine %s", name, exc_info=True)

    def _run_all_engines(
        self, frame: Frame, options: OCRProcessingOptions
    ) -> dict[str, list[dict]]:
        """
        Run every sub-engine and collect per-engine results as lists of
        ``{"text", "bbox", "confidence", "engine"}`` dicts.
        """

        def _run_single(name: str, engine: IOCREngine) -> tuple[str, list[dict]]:
            try:
                blocks = engine.extract_text(frame, options)
                threshold = self._engine_thresholds.get(name, 0.0)
                results = []
                for blk in blocks:
                    if blk.confidence >= threshold:
                        results.append(
                            {
                                "text": blk.text,
                                "bbox": blk.position,
                                "confidence": blk.confidence,
                                "engine": name,
                            }
                        )
                return name, results
            except Exception:
                logger.warning("Sub-engine %s raised during extract_text", name, exc_info=True)
                return name, []

        engine_results: dict[str, list[dict]] = {}

        if self._parallel and len(self._sub_engines) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self._sub_engines)
            ) as pool:
                futures = {
                    pool.submit(_run_single, n, e): n
                    for n, e in self._sub_engines.items()
                }
                for fut in concurrent.futures.as_completed(futures):
                    name, res = fut.result()
                    if res:
                        engine_results[name] = res
        else:
            for name, engine in self._sub_engines.items():
                _, res = _run_single(name, engine)
                if res:
                    engine_results[name] = res

        return engine_results

    def _match_regions(
        self, engine_results: dict[str, list[dict]]
    ) -> list[list[dict]]:
        """
        Match overlapping bounding boxes across engines using IoU and group
        them so each group represents the same spatial region.
        """
        all_detections: list[dict] = []
        for results in engine_results.values():
            all_detections.extend(results)

        if not all_detections:
            return []

        used = [False] * len(all_detections)
        groups: list[list[dict]] = []

        for i, det in enumerate(all_detections):
            if used[i]:
                continue
            group = [det]
            used[i] = True
            for j in range(i + 1, len(all_detections)):
                if used[j]:
                    continue
                if self._boxes_overlap(det["bbox"], all_detections[j]["bbox"]):
                    group.append(all_detections[j])
                    used[j] = True
            groups.append(group)

        return groups

    def _vote(self, groups: list[list[dict]]) -> list[TextBlock]:
        """Apply the configured voting strategy to each region group."""
        results: list[TextBlock] = []
        strategy_name = self._voting_strategy

        for group in groups:
            candidates = [
                {"text": d["text"], "confidence": d["confidence"], "engine": d["engine"]}
                for d in group
            ]

            if strategy_name == "quorum":
                outcome = voting.quorum(candidates, self._quorum_count)
                if outcome is None:
                    continue
            elif strategy_name == "majority_vote":
                outcome = voting.majority_vote(candidates)
            elif strategy_name == "weighted_confidence":
                outcome = voting.weighted_confidence(candidates)
            else:
                outcome = voting.best_confidence(candidates)

            text, confidence = outcome
            if not text.strip():
                continue

            best_det = max(group, key=lambda d: d["confidence"])
            results.append(
                TextBlock(
                    text=text,
                    position=best_det["bbox"],
                    confidence=confidence,
                    language=self._current_language,
                )
            )

        return results

    @staticmethod
    def _boxes_overlap(
        box1: Rectangle, box2: Rectangle, threshold: float = 0.3
    ) -> bool:
        """Check if two bounding boxes overlap significantly (IoU-style)."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height

        if area1 <= 0 or area2 <= 0:
            return False

        ratio1 = intersection / area1
        ratio2 = intersection / area2
        return max(ratio1, ratio2) >= threshold
