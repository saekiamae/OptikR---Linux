"""
Benchmark runner module.

Provides a reusable API for running text and vision pipeline benchmarks from
the application UI. The implementation is adapted from
``tests/test_benchmark_combinations.py`` but is independent of pytest and
environment variables so it can be safely used by dialogs and tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union


# Public data structures -----------------------------------------------------


@dataclass
class BenchmarkResult:
    """Single run: one combination on one image."""

    mode: str
    execution: str
    plugins: str
    image_name: str
    success: bool
    time_ms: float
    block_count: int
    error: str | None = None
    translation_engine: str = ""
    ocr_engine: str = ""


@dataclass
class MockCaptureLayer:
    """Capture layer that returns a fixed frame (for benchmark)."""

    _frame: object | None = None

    def set_frame(self, frame) -> None:
        self._frame = frame

    def capture_frame(self, *args, **kwargs):
        if self._frame is None:
            raise RuntimeError("MockCaptureLayer: no frame set")
        return self._frame

    def cleanup(self) -> None:
        pass


# Helpers adapted from tests/test_benchmark_combinations.py ------------------


def _load_image_as_frame(image_path: Path):
    """Load image from path into a Frame (numpy data + CaptureRegion)."""
    from PIL import Image
    import numpy as np
    from app.models import Frame, CaptureRegion, Rectangle

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    region = CaptureRegion(rectangle=Rectangle(0, 0, w, h))
    return Frame(data=arr, timestamp=time.time(), source_region=region)


TEXT_TRANSLATION_ENGINES = ["marianmt", "qwen3", "nllb200"]
TEXT_OCR_ENGINES = ["easyocr", "tesseract", "mokuro"]


def _normalize_lang(lang: str | None, fallback: str) -> str:
    if not lang:
        return fallback
    lang = str(lang).strip().lower()
    if not lang:
        return fallback
    return lang


def _to_tesseract_lang(lang: str) -> str:
    mapping = {
        "ja": "jpn",
        "jpn": "jpn",
        "japanese": "jpn",
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "de": "deu",
        "deu": "deu",
        "german": "deu",
        "fr": "fra",
        "fra": "fra",
        "french": "fra",
        "it": "ita",
        "ita": "ita",
        "italian": "ita",
    }
    return mapping.get(lang, "eng")


def _create_translation_engine(
    engine_id: str,
    gpu: bool = False,
    *,
    source_language: str = "ja",
    target_language: str = "en",
):
    """Create and initialize a translation engine by id."""
    use_gpu = gpu
    cfg = {"gpu": use_gpu, "runtime_mode": "gpu" if use_gpu else "cpu"}

    if engine_id == "marianmt":
        try:
            from plugins.stages.translation.marianmt_gpu.marianmt_engine import (
                TranslationEngine as E,
                TRANSFORMERS_AVAILABLE,
            )

            if not TRANSFORMERS_AVAILABLE:
                return None, "transformers not installed"
            eng = E()
            if not eng.initialize(cfg):
                return None, "MarianMT init failed"
            eng.preload_model(source_language, target_language)
            return eng, None
        except ImportError as exc:
            return None, f"MarianMT: {exc}"

    if engine_id == "qwen3":
        try:
            from plugins.stages.translation.qwen3.worker import (
                TranslationEngine as E,
                TRANSFORMERS_AVAILABLE,
            )

            if not TRANSFORMERS_AVAILABLE:
                return None, "transformers not installed"
            eng = E()
            cfg_q = {"model_name": "Qwen/Qwen3-1.7B", "gpu": use_gpu}
            if not eng.initialize(cfg_q):
                return None, "Qwen3 init failed"
            return eng, None
        except ImportError as exc:
            return None, f"Qwen3: {exc}"

    if engine_id == "nllb200":
        try:
            from plugins.stages.translation.nllb200.worker import (
                TranslationEngine as E,
                TRANSFORMERS_AVAILABLE,
            )

            if not TRANSFORMERS_AVAILABLE:
                return None, "transformers not installed"
            eng = E()
            if not eng.initialize({"gpu": use_gpu}):
                return None, "NLLB200 init failed"
            return eng, None
        except ImportError as exc:
            return None, f"NLLB200: {exc}"

    return None, f"Unknown translation engine: {engine_id}"


def _create_ocr_engine(
    engine_id: str,
    gpu: bool = False,
    *,
    source_language: str = "ja",
):
    """Create and initialize an OCR engine by id."""
    if engine_id == "easyocr":
        try:
            from plugins.stages.ocr.easyocr import OCREngine as E

            eng = E(engine_name="easyocr_gpu")
            if not eng.initialize({"language": source_language, "gpu": gpu}):
                return None, "EasyOCR init failed"
            return eng, None
        except ImportError as exc:
            return None, f"EasyOCR: {exc}"

    if engine_id == "mokuro":
        try:
            from plugins.stages.ocr.mokuro import OCREngine as E

            eng = E(engine_name="mokuro")
            if not eng.initialize({"gpu": gpu}):
                return None, "Mokuro init failed"
            return eng, None
        except ImportError as exc:
            return None, f"Mokuro: {exc}"

    if engine_id == "tesseract":
        try:
            from plugins.stages.ocr.tesseract import OCREngine as E

            eng = E(engine_name="tesseract")
            if not eng.initialize({"language": _to_tesseract_lang(source_language)}):
                return None, "Tesseract init failed"
            return eng, None
        except ImportError as exc:
            return None, f"Tesseract: {exc}"

    return None, f"Unknown OCR engine: {engine_id}"


def _run_single_frame_vision(
    image_path: Path,
    execution: str,
    plugins_on: bool,
    mock_capture: MockCaptureLayer,
    vision_engine_instance: object | None = None,
    *,
    source_language: str = "ja",
    target_language: str = "en",
    config_manager: object | None = None,
    timeout_s: float = 180.0,
) -> tuple[bool, float, int, str | None]:
    """Run vision pipeline on one image."""
    from app.workflow.pipeline_factory import PipelineFactory
    from app.workflow.pipeline.types import PipelineConfig, ExecutionMode

    use_shared = vision_engine_instance is not None
    if use_shared:
        engine = vision_engine_instance
    else:
        try:
            from plugins.stages.vision.qwen3_vl.worker import VisionTranslationEngine
        except ImportError:
            return False, 0.0, 0, "Vision engine not available"
        engine = VisionTranslationEngine()
        cfg = {
            "model_name": "Qwen/Qwen3-VL-2B-Instruct",
            "max_tokens": 256,
            "temperature": 0.3,
            "quantization": "4bit",
            "use_gpu": True,
        }
        if not engine.initialize(cfg) or not engine.is_available():
            return False, 0.0, 0, "Vision engine init failed"

    frame = _load_image_as_frame(image_path)
    mock_capture.set_frame(frame)

    exec_mode = ExecutionMode.ASYNC if execution == "async" else ExecutionMode.SEQUENTIAL
    config = PipelineConfig(
        execution_mode=exec_mode,
        source_language=source_language,
        target_language=target_language,
    )

    factory = PipelineFactory(config_manager=config_manager)
    pipeline = factory.create(
        "vision",
        capture_layer=mock_capture,
        vision_layer=engine,
        overlay_renderer=None,
        config=config,
        enable_all_plugins=plugins_on,
    )

    result_holder: list[dict] = []
    error_holder: list[str] = []
    event = threading.Event()

    def on_translation(data: dict):
        result_holder.append(data)
        event.set()

    def on_error(message: str):
        error_holder.append(str(message))
        event.set()

    pipeline.on_translation = on_translation
    pipeline.on_error = on_error
    pipeline.start()

    ok = event.wait(timeout=timeout_s)
    pipeline.stop()
    try:
        pipeline.cleanup()
    except Exception:
        pass
    if not use_shared and getattr(engine, "cleanup", None):
        engine.cleanup()

    if not ok:
        detail = error_holder[0] if error_holder else "No translation callback (timeout or failure)"
        return False, 0.0, 0, detail
    if not result_holder:
        detail = error_holder[0] if error_holder else "No translation callback (timeout or failure)"
        return False, 0.0, 0, detail

    data = result_holder[0]
    translations = data.get("translations") or []
    text_blocks = data.get("text_blocks") or []
    block_count = len(translations)
    if block_count == 0 and len(text_blocks) == 0:
        detail = error_holder[0] if error_holder else "No translations produced"
        return False, 0.0, 0, detail
    return True, 0.0, block_count, None


def _run_single_frame_text(
    image_path: Path,
    execution: str,
    plugins_on: bool,
    mock_capture: MockCaptureLayer,
    translation_engine_id: str = "marianmt",
    ocr_engine_id: str = "easyocr",
    translation_engine_instance: object | None = None,
    ocr_engine_instance: object | None = None,
    *,
    source_language: str = "ja",
    target_language: str = "en",
    config_manager: object | None = None,
    timeout_s: float = 120.0,
) -> tuple[bool, float, int, str | None]:
    """Run text (OCR+translation) pipeline on one image."""
    from app.workflow.pipeline_factory import PipelineFactory
    from app.workflow.pipeline.types import PipelineConfig, ExecutionMode
    from app.ocr.ocr_engine_interface import OCRProcessingOptions

    use_shared_ocr = ocr_engine_instance is not None
    use_shared_trans = translation_engine_instance is not None

    if not use_shared_ocr:
        ocr_engine, ocr_err = _create_ocr_engine(
            ocr_engine_id,
            gpu=False,
            source_language=source_language,
        )
        if ocr_engine is None:
            return False, 0.0, 0, ocr_err or "OCR init failed"
    else:
        ocr_engine = ocr_engine_instance

    if not use_shared_trans:
        trans_engine, trans_err = _create_translation_engine(
            translation_engine_id,
            gpu=False,
            source_language=source_language,
            target_language=target_language,
        )
        if trans_engine is None:
            if not use_shared_ocr and getattr(ocr_engine, "cleanup", None):
                ocr_engine.cleanup()
            return False, 0.0, 0, trans_err or "Translation init failed"
    else:
        trans_engine = translation_engine_instance

    class _OCRLayerAdapter:
        def __init__(self, eng, cleanup_engine: bool = True):
            self._eng = eng
            self._cleanup_engine = cleanup_engine

        def extract_text(self, frame, engine=None, options=None):
            opts = options or OCRProcessingOptions(
                language=source_language,
                confidence_threshold=0.3,
            )
            return self._eng.extract_text(frame, opts)

        def cleanup(self):
            if self._cleanup_engine and getattr(self._eng, "cleanup", None):
                self._eng.cleanup()

    class _TranslationLayerAdapter:
        def __init__(self, eng, cleanup_engine: bool = True):
            self._eng = eng
            self._cleanup_engine = cleanup_engine

        def translate_batch(self, texts, engine, src_lang, tgt_lang):
            batch = self._eng.translate_batch(texts, src_lang, tgt_lang)
            return [r.translated_text for r in batch.results]

        def cleanup(self):
            if self._cleanup_engine and getattr(self._eng, "cleanup", None):
                self._eng.cleanup()

    ocr_layer = _OCRLayerAdapter(ocr_engine, cleanup_engine=not use_shared_ocr)
    trans_layer = _TranslationLayerAdapter(trans_engine, cleanup_engine=not use_shared_trans)

    frame = _load_image_as_frame(image_path)
    mock_capture.set_frame(frame)

    preset = "async" if execution == "async" else "sequential"
    exec_mode = ExecutionMode.ASYNC if execution == "async" else ExecutionMode.SEQUENTIAL
    config = PipelineConfig(
        execution_mode=exec_mode,
        source_language=source_language,
        target_language=target_language,
    )

    factory = PipelineFactory(config_manager=config_manager)
    pipeline = factory.create(
        preset,
        capture_layer=mock_capture,
        ocr_layer=ocr_layer,
        translation_layer=trans_layer,
        overlay_renderer=None,
        config=config,
        enable_all_plugins=plugins_on,
    )

    result_holder: list[dict] = []
    error_holder: list[str] = []
    event = threading.Event()

    def on_translation(data: dict):
        result_holder.append(data)
        event.set()

    def on_error(message: str):
        error_holder.append(str(message))
        event.set()

    pipeline.on_translation = on_translation
    pipeline.on_error = on_error
    pipeline.start()

    ok = event.wait(timeout=timeout_s)
    pipeline.stop()
    try:
        pipeline.cleanup()
    except Exception:
        pass

    if not ok:
        detail = error_holder[0] if error_holder else "No translation callback (timeout or failure)"
        return False, 0.0, 0, detail
    if not result_holder:
        detail = error_holder[0] if error_holder else "No translation callback (timeout or failure)"
        return False, 0.0, 0, detail

    data = result_holder[0]
    translations = data.get("translations") or []
    block_count = len(translations)
    return True, 0.0, block_count, None


def _run_combination(
    image_path: Path,
    mode: str,
    execution: str,
    plugins_on: bool,
    mock_capture: MockCaptureLayer,
    translation_engine: str = "",
    ocr_engine: str = "",
    vision_engine_instance: object | None = None,
    translation_engine_instance: object | None = None,
    ocr_engine_instance: object | None = None,
    *,
    source_language: str = "ja",
    target_language: str = "en",
    config_manager: object | None = None,
    vision_timeout_s: float = 180.0,
    text_timeout_s: float = 120.0,
) -> BenchmarkResult:
    """Run one (mode, execution, plugins, engines) on one image."""
    t0 = time.perf_counter()
    if mode == "vision":
        success, _, block_count, err = _run_single_frame_vision(
            image_path,
            execution,
            plugins_on,
            mock_capture,
            vision_engine_instance=vision_engine_instance,
            source_language=source_language,
            target_language=target_language,
            config_manager=config_manager,
            timeout_s=vision_timeout_s,
        )
    else:
        success, _, block_count, err = _run_single_frame_text(
            image_path,
            execution,
            plugins_on,
            mock_capture,
            translation_engine_id=translation_engine or "marianmt",
            ocr_engine_id=ocr_engine or "easyocr",
            translation_engine_instance=translation_engine_instance,
            ocr_engine_instance=ocr_engine_instance,
            source_language=source_language,
            target_language=target_language,
            config_manager=config_manager,
            timeout_s=text_timeout_s,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    plugins_str = "plugins_on" if plugins_on else "plugins_off"
    return BenchmarkResult(
        mode=mode,
        execution=execution,
        plugins=plugins_str,
        image_name=image_path.name,
        success=success,
        time_ms=elapsed_ms,
        block_count=block_count,
        error=err,
        translation_engine=translation_engine,
        ocr_engine=ocr_engine,
    )


def _default_progress_printer(msg: str) -> None:
    print(msg, flush=True)


def _run_benchmark_with_reuse(
    images: Sequence[Path],
    combinations: Sequence[Tuple],
    mock_capture: MockCaptureLayer,
    progress: Optional[Callable[[str], None]] = None,
    *,
    source_language: str = "ja",
    target_language: str = "en",
    config_manager: object | None = None,
    vision_engine_config: Optional[dict] = None,
    vision_timeout_s: float = 180.0,
    text_timeout_s: float = 120.0,
) -> List[BenchmarkResult]:
    """
    Core runner that reuses engines per type to avoid repeated model loads.

    This is a slightly generalized version of the test helper that accepts a
    progress callback instead of printing directly.
    """
    log = progress or _default_progress_printer

    results: List[BenchmarkResult] = []
    vision_combos = [c for c in combinations if c[0] == "vision"]
    text_combos = [c for c in combinations if c[0] == "text"]
    total_runs = len(images) * (len(vision_combos) + len(text_combos))
    run_index = 0

    log(f"\n--- Benchmark: {len(images)} image(s), {len(combinations)} combinations ---")

    if vision_combos:
        log("Loading vision engine (Qwen3-VL)...")
        try:
            from plugins.stages.vision.qwen3_vl.worker import VisionTranslationEngine
        except ImportError:
            log("  Vision engine not available (skip).")
            for img in images:
                for c in vision_combos:
                    results.append(
                        BenchmarkResult(
                            mode="vision",
                            execution=c[1],
                            plugins="plugins_off" if not c[2] else "plugins_on",
                            image_name=img.name,
                            success=False,
                            time_ms=0,
                            block_count=0,
                            error="Vision engine not available",
                        )
                    )
        else:
            engine = VisionTranslationEngine()
            cfg = {
                "model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "max_tokens": 256,
                "temperature": 0.3,
                "quantization": "4bit",
                "use_gpu": True,
            }
            if config_manager is not None and hasattr(config_manager, "get_setting"):
                try:
                    cfg.update(
                        {
                            "model_name": config_manager.get_setting(
                                "vision.model_name",
                                cfg["model_name"],
                            ),
                            "max_tokens": config_manager.get_setting(
                                "vision.max_tokens",
                                cfg["max_tokens"],
                            ),
                            "temperature": config_manager.get_setting(
                                "vision.temperature",
                                cfg["temperature"],
                            ),
                            "quantization": config_manager.get_setting(
                                "vision.quantization",
                                cfg["quantization"],
                            ),
                            "use_gpu": config_manager.get_setting(
                                "vision.use_gpu",
                                cfg["use_gpu"],
                            ),
                            "prompt_template": config_manager.get_setting(
                                "vision.prompt_template",
                                "",
                            ),
                            "context": config_manager.get_setting("vision.context", ""),
                            "exclude_sfx": config_manager.get_setting(
                                "vision.exclude_sfx",
                                False,
                            ),
                        }
                    )
                except Exception:
                    pass
            if vision_engine_config:
                cfg.update(vision_engine_config)
            quant = str(cfg.get("quantization", "none")).strip().lower()
            if quant in {"4bit", "8bit"}:
                try:
                    import bitsandbytes  # noqa: F401
                except Exception:
                    log(
                        f"  bitsandbytes not available; falling back vision quantization from {quant} to none."
                    )
                    cfg["quantization"] = "none"
            if engine.initialize(cfg) and engine.is_available():
                log("  Vision engine ready.")
                for image_path in images:
                    for c in vision_combos:
                        run_index += 1
                        pl = "plugins_on" if c[2] else "plugins_off"
                        log(
                            f"  [{run_index}/{total_runs}] Running vision | {c[1]} | {pl} @ {image_path.name} ..."
                        )
                        r = _run_combination(
                            image_path,
                            c[0],
                            c[1],
                            c[2],
                            mock_capture,
                            translation_engine=c[3] if len(c) > 3 else "",
                            ocr_engine=c[4] if len(c) > 4 else "",
                            vision_engine_instance=engine,
                            source_language=source_language,
                            target_language=target_language,
                            config_manager=config_manager,
                            vision_timeout_s=vision_timeout_s,
                            text_timeout_s=text_timeout_s,
                        )
                        results.append(r)
                        log(
                            f"       -> {'OK' if r.success else 'FAIL'}  {r.block_count} blocks  {r.time_ms:.0f} ms"
                            + (f"  ({r.error})" if r.error else "")
                        )
                try:
                    engine.cleanup()
                except Exception:
                    pass
                log("  Vision engine unloaded.")
            else:
                log("  Vision init failed (skip).")
                for img in images:
                    for c in vision_combos:
                        results.append(
                            BenchmarkResult(
                                mode="vision",
                                execution=c[1],
                                plugins="plugins_off" if not c[2] else "plugins_on",
                                image_name=img.name,
                                success=False,
                                time_ms=0,
                                block_count=0,
                                error="Vision init failed",
                            )
                        )

    if text_combos:
        unique_ocr = list({(c[4] if len(c) > 4 else "easyocr") for c in text_combos})
        unique_trans = list({(c[3] if len(c) > 3 else "marianmt") for c in text_combos})
        ocr_engines: dict[str, object] = {}
        for oid in unique_ocr:
            log(f"Loading OCR engine ({oid})...")
            eng, err = _create_ocr_engine(
                oid,
                gpu=False,
                source_language=source_language,
            )
            if eng is not None:
                ocr_engines[oid] = eng
                log(f"  OCR ({oid}) ready.")
            else:
                log(f"  OCR ({oid}) failed: {err}")

        for trans_id in unique_trans:
            log(f"Loading translation engine ({trans_id})...")
            trans_eng, err = _create_translation_engine(
                trans_id,
                gpu=False,
                source_language=source_language,
                target_language=target_language,
            )
            if trans_eng is None:
                log(f"  Translation ({trans_id}) failed: {err}")
                for img in images:
                    for c in text_combos:
                        if (c[3] if len(c) > 3 else "marianmt") == trans_id:
                            results.append(
                                BenchmarkResult(
                                    mode="text",
                                    execution=c[1],
                                    plugins="plugins_off" if not c[2] else "plugins_on",
                                    image_name=img.name,
                                    success=False,
                                    time_ms=0,
                                    block_count=0,
                                    translation_engine=trans_id,
                                    ocr_engine=c[4] if len(c) > 4 else "easyocr",
                                    error="Translation engine init failed",
                                )
                            )
                continue
            else:
                log(f"  Translation ({trans_id}) ready.")

            for image_path in images:
                for c in text_combos:
                    if (c[3] if len(c) > 3 else "") != trans_id:
                        continue
                    oid = c[4] if len(c) > 4 else "easyocr"
                    ocr_eng = ocr_engines.get(oid)
                    if ocr_eng is None:
                        continue
                    run_index += 1
                    pl = "plugins_on" if c[2] else "plugins_off"
                    log(
                        f"  [{run_index}/{total_runs}] Running text | {c[1]} | {pl} | {trans_id} | {oid} @ {image_path.name} ..."
                    )
                    r = _run_combination(
                        image_path,
                        c[0],
                        c[1],
                        c[2],
                        mock_capture,
                        translation_engine=trans_id,
                        ocr_engine=oid,
                        translation_engine_instance=trans_eng,
                        ocr_engine_instance=ocr_eng,
                        source_language=source_language,
                        target_language=target_language,
                        config_manager=config_manager,
                        vision_timeout_s=vision_timeout_s,
                        text_timeout_s=text_timeout_s,
                    )
                    results.append(r)
                    log(
                        f"       -> {'OK' if r.success else 'FAIL'}  {r.block_count} blocks  {r.time_ms:.0f} ms"
                        + (f"  ({r.error})" if r.error else "")
                    )
            if trans_eng is not None:
                try:
                    trans_eng.cleanup()
                except Exception:
                    pass
                log(f"  Translation ({trans_id}) unloaded.")

        for oid, eng in ocr_engines.items():
            try:
                eng.cleanup()
            except Exception:
                pass
            log(f"  OCR ({oid}) unloaded.")

    log("--- Benchmark runs finished ---\n")
    return results


# Public API -----------------------------------------------------------------


def build_default_combinations(
    include_vision: bool = True,
    include_text: bool = True,
    fast: bool = True,
    text_translation_engines: Optional[Sequence[str]] = None,
    text_ocr_engines: Optional[Sequence[str]] = None,
) -> List[Tuple]:
    """
    Build a default combination matrix for benchmarking.

    This mirrors the combinations used in tests but allows callers to
    restrict scope (fast vs full, text-only, vision-only, etc.).
    """
    combos: list[Tuple] = []

    if include_vision:
        if fast:
            vision = [("vision", "sequential", False, "", "")]
        else:
            vision = [
                ("vision", "sequential", True, "", ""),
                ("vision", "sequential", False, "", ""),
                ("vision", "async", True, "", ""),
                ("vision", "async", False, "", ""),
            ]
        combos.extend(vision)

    if include_text:
        trans_engs = list(text_translation_engines or TEXT_TRANSLATION_ENGINES)
        ocr_engs = list(text_ocr_engines or TEXT_OCR_ENGINES)
        if fast:
            for te in trans_engs:
                combos.append(("text", "sequential", False, te, "easyocr"))
        else:
            for execution in ("sequential", "async"):
                for plugins_on in (True, False):
                    for trans_eng in trans_engs:
                        for ocr_eng in ocr_engs:
                            combos.append(
                                ("text", execution, plugins_on, trans_eng, ocr_eng)
                            )

    return combos


def guard_vision_async_combinations(
    combinations: Sequence[Tuple],
    *,
    allow_vision_async: bool,
) -> List[Tuple]:
    """
    Return a guarded combination list where, unless ``allow_vision_async`` is True,
    any vision + async combinations are downgraded to sequential and duplicates are removed.

    This keeps vision benchmarks focused on the more stable sequential execution mode
    while still allowing callers (or configuration) to explicitly opt back into async.
    """
    if allow_vision_async:
        return list(combinations)

    guarded: list[Tuple] = []
    seen: set[Tuple] = set()

    for combo in combinations:
        if len(combo) >= 2 and combo[0] == "vision" and combo[1] == "async":
            downgraded = ("vision", "sequential") + combo[2:]
            key = downgraded
        else:
            downgraded = combo
            key = combo

        if key in seen:
            continue
        seen.add(key)
        guarded.append(downgraded)

    return guarded


def run_benchmark(
    images: Iterable[Union[str, Path]],
    combinations: Optional[Sequence[Tuple]] = None,
    *,
    include_vision: bool = True,
    include_text: bool = True,
    fast: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
    allow_vision_async: bool = True,
    config_manager: object | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
    vision_engine_config: Optional[dict] = None,
    vision_timeout_s: float = 180.0,
    text_timeout_s: float = 120.0,
) -> List[BenchmarkResult]:
    """
    Run a benchmark over the given images and combination matrix.

    Args:
        images: Iterable of image paths (str or Path).
        combinations: Optional explicit combinations; if omitted, a default
            matrix is generated via ``build_default_combinations`` using the
            ``include_*`` and ``fast`` flags.
        include_vision: Whether to include vision-mode combinations when
            generating defaults.
        include_text: Whether to include text-mode combinations when
            generating defaults.
        fast: When ``True`` and ``combinations`` is not provided, use a smaller
            set of combinations suitable for quick experiments.
        progress_callback: Optional callback that receives human-readable
            progress messages. When omitted, messages are printed to stdout.

    Returns:
        List of ``BenchmarkResult`` instances, one per (image, combination).
    """
    image_paths: List[Path] = []
    for p in images:
        path = Path(p)
        if path.is_file():
            image_paths.append(path)
    if not image_paths:
        return []

    if combinations is None:
        default_combos = build_default_combinations(
            include_vision=include_vision,
            include_text=include_text,
            fast=fast,
        )
        combinations = guard_vision_async_combinations(
            default_combos, allow_vision_async=allow_vision_async
        )

    src_lang = _normalize_lang(source_language, "ja")
    tgt_lang = _normalize_lang(target_language, "en")
    if config_manager is not None and hasattr(config_manager, "get_setting"):
        try:
            src_lang = _normalize_lang(
                config_manager.get_setting("translation.source_language", src_lang),
                src_lang,
            )
            tgt_lang = _normalize_lang(
                config_manager.get_setting("translation.target_language", tgt_lang),
                tgt_lang,
            )
        except Exception:
            pass

    mock_capture = MockCaptureLayer()
    return _run_benchmark_with_reuse(
        images=image_paths,
        combinations=list(combinations),
        mock_capture=mock_capture,
        progress=progress_callback,
        source_language=src_lang,
        target_language=tgt_lang,
        config_manager=config_manager,
        vision_engine_config=vision_engine_config,
        vision_timeout_s=vision_timeout_s,
        text_timeout_s=text_timeout_s,
    )

