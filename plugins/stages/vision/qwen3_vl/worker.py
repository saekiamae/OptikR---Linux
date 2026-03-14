"""Qwen3-VL Vision Translation Engine.

Uses the Qwen3-VL vision-language model to perform combined OCR and
translation in a single pass.  Takes a raw frame (numpy array / PIL Image)
plus source/target language, and returns translated text blocks with
bounding boxes parsed from the model output.
"""

import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

logger = logging.getLogger("optikr.vision.qwen3_vl")

_DEFAULT_PROMPT = (
    "Extract all visible text from this image and translate it from "
    "{source_lang} to {target_lang}. For each text region, return the "
    "original (detected) text, the translated text, and its approximate "
    'bounding box as JSON: [{{"original": "...", "text": "...", "bbox": [x, y, w, h]}}]. '
    "Use \"original\" for the source-language text and \"text\" for the translation. "
    "Only return the JSON array, no other text."
)

# Prompt that excludes sound effects (SFX) for accuracy evaluation on dialogue/narration only.
# Use when exclude_sfx=True in config (e.g. for manga dialogue-only translation).
_PROMPT_DIALOGUE_ONLY = (
    "This image is from a manga or comic. Extract ONLY dialogue and narration text "
    "(speech bubbles, thought bubbles, narrative boxes). Do NOT include sound effects "
    "(SFX) such as onomatopoeia like ギシャン, ガシャン, ポッ, ドン, etc. "
    "Translate each extracted text from {source_lang} to {target_lang}. "
    "For each text region, return the original (detected) text, the translated text, "
    'and its approximate bounding box as JSON: [{{"original": "...", "text": "...", "bbox": [x, y, w, h]}}]. '
    "Use \"original\" for the source-language text and \"text\" for the translation. "
    "Only return the JSON array, no other text."
)


class VisionTranslationEngine:
    """Qwen3-VL-based vision translation engine.

    Loads the model lazily on first ``translate_frame`` call and caches it
    for subsequent invocations.

    Public API
    ----------
    initialize(config)   -- configure model parameters
    is_available()       -- check whether dependencies are present
    translate_frame(frame, src_lang, tgt_lang) -- run vision translation
    cleanup()            -- release GPU memory
    """

    name = "qwen3_vl"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._init_lock = threading.RLock()
        self._device: str = "cpu"
        self._model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
        self._max_tokens: int = 512
        self._temperature: float = 0.3
        self._quantization: str = "none"
        self._prompt_template: str = _DEFAULT_PROMPT
        self._context: str = ""
        self._exclude_sfx: bool = False
        self._use_gpu: bool = True
        self._initialized: bool = False

    def initialize(self, config: dict[str, Any]) -> bool:
        """Configure engine parameters from *config*.

        Does NOT load the model yet -- that happens on first use to avoid
        blocking startup.
        """
        raw_name = config.get("model_name", self._model_name)
        # Hugging Face hosts -Instruct (and -Thinking) variants; base IDs like Qwen/Qwen3-VL-4B are invalid
        if raw_name and "/" in raw_name and "-Instruct" not in raw_name and "-Thinking" not in raw_name:
            base = raw_name.rstrip("/").split("/")[-1]
            if base in ("Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B"):
                self._model_name = raw_name.replace(base, base + "-Instruct")
            else:
                self._model_name = raw_name
        else:
            self._model_name = raw_name
        self._max_tokens = config.get("max_tokens", self._max_tokens)
        self._temperature = config.get("temperature", self._temperature)
        self._quantization = config.get("quantization", self._quantization)
        self._exclude_sfx = config.get("exclude_sfx", False)
        self._use_gpu = config.get("use_gpu", True)
        tpl = config.get("prompt_template", "")
        if tpl.strip():
            self._prompt_template = tpl
        else:
            self._prompt_template = _PROMPT_DIALOGUE_ONLY if self._exclude_sfx else _DEFAULT_PROMPT
        self._context = (config.get("context") or "").strip()
        self._initialized = True
        logger.info(
            "VisionTranslationEngine configured: model=%s quantization=%s use_gpu=%s",
            self._model_name,
            self._quantization,
            self._use_gpu,
        )
        return True

    @staticmethod
    def is_available() -> bool:
        """Return True when the required libraries are importable."""
        try:
            import torch  # noqa: F401
            from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
            from qwen_vl_utils import process_vision_info  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self) -> str | None:
        """Load the Qwen3-VL model lazily.  Returns an error string or None."""
        if self._model is not None:
            return None

        with self._init_lock:
            # Another thread may have finished initialization while we were waiting.
            if self._model is not None:
                return None

            try:
                import torch
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            except ImportError as exc:
                logger.error("Vision Qwen3-VL missing dependency during init: %s", exc)
                return f"Missing dependency: {exc}"

            cuda_available = torch.cuda.is_available()
            if self._use_gpu and cuda_available:
                self._device = "cuda"
            elif self._use_gpu and not cuda_available:
                logger.warning(
                    "Vision use_gpu=True but CUDA is not available (PyTorch not built with CUDA or no GPU). Using CPU."
                )
                self._device = "cpu"
            else:
                self._device = "cpu"

            try:
                logger.info(
                    "Loading Qwen3-VL model '%s' on %s (quantization=%s)...",
                    self._model_name,
                    self._device,
                    self._quantization,
                )

                model_kwargs: dict[str, Any] = {
                    "torch_dtype": torch.float16 if self._device == "cuda" else torch.float32,
                    "device_map": "auto" if self._device == "cuda" else None,
                }

                if self._quantization == "4bit":
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif self._quantization == "8bit":
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self._model_name, **model_kwargs
                )
                self._processor = AutoProcessor.from_pretrained(self._model_name)

                if self._device != "cuda" or "device_map" not in model_kwargs:
                    self._model = self._model.to(self._device)

                # Log library versions and processor/tokenizer wiring once model is ready.
                try:
                    import transformers

                    hf_version = getattr(transformers, "__version__", "unknown")
                except Exception:
                    hf_version = "unavailable"

                try:
                    import qwen_vl_utils  # type: ignore

                    qwen_pkg_name = getattr(qwen_vl_utils, "__name__", "qwen_vl_utils")
                    qwen_version = getattr(qwen_vl_utils, "__version__", "unknown")
                except Exception:
                    qwen_pkg_name = "qwen_vl_utils"
                    qwen_version = "unavailable"

                processor_type = type(self._processor).__name__ if self._processor is not None else "None"
                tokenizer = getattr(self._processor, "tokenizer", None)
                tokenizer_type = type(tokenizer).__name__ if tokenizer is not None else "None"
                has_proc_decode = hasattr(self._processor, "decode") or hasattr(self._processor, "batch_decode")
                has_tok_decode = bool(
                    tokenizer is not None
                    and (hasattr(tokenizer, "decode") or hasattr(tokenizer, "batch_decode"))
                )
                decoder_attr = getattr(tokenizer, "decoder", None)
                decoder_type = type(decoder_attr).__name__ if decoder_attr is not None else "None"

                logger.info(
                    "Qwen3-VL model loaded successfully on %s; transformers=%s, %s=%s; "
                    "processor=%s tokenizer=%s has_proc_decode=%s has_tok_decode=%s tokenizer.decoder=%s",
                    self._device,
                    hf_version,
                    qwen_pkg_name,
                    qwen_version,
                    processor_type,
                    tokenizer_type,
                    has_proc_decode,
                    has_tok_decode,
                    decoder_type,
                )
                return None

            except Exception as exc:
                # Ensure we never leave the engine in a half-initialized state.
                self._model = None
                self._processor = None
                logger.error("Failed to load Qwen3-VL model: %s", exc)
                return f"Failed to load model: {exc}"

    def translate_frame(
        self,
        frame: np.ndarray | Any,
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> list[dict[str, Any]]:
        """Run vision translation on a single frame.

        Parameters
        ----------
        frame:
            Raw captured frame as a numpy array (H, W, C) or PIL Image.
        source_lang:
            Source language code.
        target_lang:
            Target language code.

        Returns
        -------
        list[dict]
            Each dict has ``text`` (translated string) and ``bbox``
            ([x, y, w, h] list).  Returns an empty list on failure.
        """
        err = self._ensure_model()
        if err:
            logger.error("Vision engine not ready: %s", err)
            return []

        try:
            from PIL import Image
            import torch
            from qwen_vl_utils import process_vision_info

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    image = Image.fromarray(frame, mode="L").convert("RGB")
                elif frame.shape[2] == 4:
                    image = Image.fromarray(frame[:, :, :3])
                else:
                    image = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                image = frame
            else:
                logger.error("Unsupported frame type: %s", type(frame))
                return []

            prompt = self._prompt_template.format(
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if self._context:
                prompt = f"{self._context}\n\n{prompt}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            images, videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )

            inputs = self._processor(
                text=[text_input],
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                **video_kwargs,
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    temperature=self._temperature,
                    do_sample=self._temperature > 0,
                    repetition_penalty=1.2,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]

            processor = self._processor
            tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None

            logger.debug(
                "Vision decode preflight: processor=%s tokenizer=%s "
                "has_proc_decode=%s has_proc_batch_decode=%s "
                "has_tok_decode=%s has_tok_batch_decode=%s",
                type(processor).__name__ if processor is not None else "None",
                type(tokenizer).__name__ if tokenizer is not None else "None",
                hasattr(processor, "decode") if processor is not None else False,
                hasattr(processor, "batch_decode") if processor is not None else False,
                hasattr(tokenizer, "decode") if tokenizer is not None else False,
                hasattr(tokenizer, "batch_decode") if tokenizer is not None else False,
            )

            if processor is None:
                logger.error("Vision decode aborted: processor is None after model initialization.")
                return []

            raw_output: str | None = None
            try:
                # Prefer tokenizer batch_decode when available; fall back through other safe options.
                if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
                    raw_output = tokenizer.batch_decode(
                        generated.unsqueeze(0), skip_special_tokens=True
                    )[0]
                elif tokenizer is not None and hasattr(tokenizer, "decode"):
                    raw_output = tokenizer.decode(generated, skip_special_tokens=True)
                elif hasattr(processor, "batch_decode"):
                    raw_output = processor.batch_decode(
                        generated.unsqueeze(0), skip_special_tokens=True
                    )[0]
                elif hasattr(processor, "decode"):
                    raw_output = processor.decode(generated, skip_special_tokens=True)
                else:
                    logger.error(
                        "Vision decode aborted: no suitable decode/batch_decode on processor/tokenizer "
                        "(processor=%s tokenizer=%s)",
                        type(processor).__name__,
                        type(tokenizer).__name__ if tokenizer is not None else "None",
                    )
                    return []
            except Exception as dec_exc:
                logger.error(
                    "Vision decode failed before parsing: [%s] %s",
                    type(dec_exc).__name__,
                    dec_exc,
                )
                return []

            if raw_output is None:
                logger.error("Vision decode returned None raw_output; skipping parse.")
                return []

            logger.debug("Qwen3-VL raw output: %s", raw_output[:200])
            results = self._parse_output(raw_output)
            # Scale bbox to original image (capture) coordinates so overlay gets correct positions.
            results = self._scale_bbox_to_image(results, image)
            return results

        except Exception as exc:
            logger.error("Vision translation failed: [%s] %s", type(exc).__name__, exc)
            return []

    @staticmethod
    def _parse_output(raw_output: str) -> list[dict[str, Any]]:
        """Parse the model's JSON output into text blocks.

        Tolerates markdown code fences and partial JSON.
        """
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return _normalize_blocks(parsed)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return _normalize_blocks(parsed)
            except json.JSONDecodeError:
                pass

        if cleaned:
            return [{"original": "", "text": cleaned, "bbox": [0, 0, 100, 30]}]

        return []

    def cleanup(self) -> None:
        """Release the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("VisionTranslationEngine cleaned up")

    def _scale_bbox_to_image(
        self, results: list[dict[str, Any]], image: Any
    ) -> list[dict[str, Any]]:
        """Scale bbox from model output space to original image pixel coordinates.

        If the model returns normalized coords (e.g. 0–1), scale by image size.
        If bbox already looks like pixel coords (values > 1), assume they are
        in the image passed to the model; if image was resized by the processor,
        we still expect the model to return coords in original image space per
        the prompt, so no scaling is applied when values exceed 1.
        """
        if not results:
            return results
        try:
            if hasattr(image, "size"):
                w, h = image.size
            elif hasattr(image, "shape"):
                h, w = image.shape[:2]
            else:
                return results
            if w <= 0 or h <= 0:
                return results
            out = []
            for item in results:
                bbox = list(item.get("bbox", [0, 0, 100, 30]))
                if len(bbox) < 4:
                    bbox = bbox + [100, 30][len(bbox) :]
                # Normalized bbox: all values typically in [0, 1] or [0, 1.x]
                try:
                    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                except (TypeError, ValueError):
                    out.append(item)
                    continue
                if 0 <= x <= 1.5 and 0 <= y <= 1.5 and 0 <= bw <= 1.5 and 0 <= bh <= 1.5:
                    x = int(x * w)
                    y = int(y * h)
                    bw = max(10, int(bw * w))
                    bh = max(10, int(bh * h))
                    bbox = [x, y, bw, bh]
                else:
                    bbox = [int(x), int(y), max(10, int(bw)), max(10, int(bh))]
                out.append({**item, "bbox": bbox})
            return out
        except Exception as e:
            logger.debug("Vision bbox scaling skipped: %s", e)
            return results


def _normalize_blocks(parsed: list) -> list[dict[str, Any]]:
    """Extract original + text + bbox from parsed JSON for cross-pipeline cache.

    Supports both formats: {"original": "...", "text": "...", "bbox": [...]}
    and legacy {"text": "...", "bbox": [...]} (original set to empty).
    """
    results = []
    for item in parsed:
        if not isinstance(item, dict) or "text" not in item:
            continue
        bbox = item.get("bbox", [0, 0, 100, 30])
        if isinstance(bbox, list) and len(bbox) >= 4:
            bbox = [float(v) for v in bbox[:4]]
        else:
            bbox = [0.0, 0.0, 100.0, 30.0]
        original = str(item.get("original", "")).strip()
        text = str(item["text"]).strip()
        results.append(
            {
                "original": original,
                "text": text,
                "bbox": bbox,
            }
        )
    return results

