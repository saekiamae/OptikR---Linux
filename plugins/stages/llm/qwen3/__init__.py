"""
Qwen3 LLM Engine Plugin

LLM engine using Qwen3 causal language models for flexible text processing:
refinement of translations, full LLM-based translation, and custom prompt
execution.  Reuses the same model-loading approach as the Qwen3 translation
plugin but exposes the richer ``ILLMEngine`` interface with configurable
system prompts and processing modes.
"""

import logging
from typing import Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.llm.llm_engine_interface import (
    ILLMEngine,
    LLMEngineCapabilities,
    LLMEngineStatus,
    LLMEngineType,
    LLMProcessingMode,
    LLMProcessingOptions,
)

logger = logging.getLogger("optikr.llm.qwen3")

_LANG_NAMES: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
}

_DEFAULT_SYSTEM_PROMPTS: dict[LLMProcessingMode, str] = {
    LLMProcessingMode.REFINE: (
        "You are a professional editor. Refine the following translated text "
        "so it reads naturally in the target language. Preserve the original "
        "meaning. Output ONLY the refined text, nothing else."
    ),
    LLMProcessingMode.TRANSLATE: (
        "You are a professional translator. Translate the following "
        "{src_lang} text into {tgt_lang}. Output ONLY the translation, "
        "nothing else."
    ),
    LLMProcessingMode.CUSTOM: "",
}


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


class LLMEngine(ILLMEngine):
    """Qwen3 LLM engine implementing ILLMEngine."""

    def __init__(
        self,
        engine_name: str = "qwen3",
        engine_type: LLMEngineType | str = LLMEngineType.QWEN3,
    ) -> None:
        if isinstance(engine_type, str):
            try:
                engine_type = LLMEngineType(engine_type)
            except ValueError:
                engine_type = LLMEngineType.QWEN3

        super().__init__(engine_name, engine_type)

        self.model = None
        self.tokenizer = None
        self.model_name = "Qwen/Qwen3-1.7B"
        self._device: "torch.device | None" = None
        self._max_tokens = 512
        self._temperature = 0.7
        self._system_prompt: str = ""
        self._using_shared_model = False

    # ------------------------------------------------------------------
    # ILLMEngine lifecycle
    # ------------------------------------------------------------------

    def initialize(self, config: dict[str, Any]) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            self._logger.error("transformers / torch not available")
            self.status = LLMEngineStatus.ERROR
            return False

        try:
            self.status = LLMEngineStatus.INITIALIZING

            self.model_name = config.get("model_name", self.model_name)
            self._max_tokens = int(config.get("max_tokens", self._max_tokens))
            self._temperature = float(config.get("temperature", self._temperature))
            quantization = config.get("quantization", "none")
            use_gpu = config.get("gpu", True) or config.get("use_gpu", True)

            self._device = (
                torch.device("cuda")
                if use_gpu and torch.cuda.is_available()
                else torch.device("cpu")
            )

            device_str = str(self._device)

            # Check SharedModelRegistry for an already-loaded instance
            # (e.g. the Qwen3 translation plugin loaded the same model)
            try:
                from app.llm.llm_layer import SharedModelRegistry
                shared = SharedModelRegistry.instance().get(self.model_name, device_str)
                if shared is not None:
                    self.model = shared["model"]
                    self.tokenizer = shared["tokenizer"]
                    self._using_shared_model = True
                    self._logger.info(
                        "Qwen3 already loaded as translation engine — "
                        "LLM stage will share the model instance: %s on %s",
                        self.model_name, device_str,
                    )
                    self.capabilities = LLMEngineCapabilities(
                        supported_modes=[
                            LLMProcessingMode.REFINE,
                            LLMProcessingMode.TRANSLATE,
                            LLMProcessingMode.CUSTOM,
                        ],
                        supports_gpu=self._device.type == "cuda",
                        supports_batch_processing=True,
                        supports_streaming=False,
                        max_context_length=4096,
                        memory_requirements_mb=self._estimate_memory_mb(),
                    )
                    self.status = LLMEngineStatus.READY
                    return True
            except ImportError:
                pass

            self._logger.info(
                "Loading Qwen3 LLM model: %s on %s (quantization=%s)",
                self.model_name,
                self._device,
                quantization,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )

            load_kwargs: dict[str, Any] = {"trust_remote_code": True}

            if quantization == "4bit":
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
            elif quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, use_safetensors=True, **load_kwargs,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, use_safetensors=False, **load_kwargs,
                )

            if "device_map" not in load_kwargs:
                self.model = self.model.to(self._device)

            self.model.eval()

            # Register in SharedModelRegistry so other components can reuse it
            try:
                from app.llm.llm_layer import SharedModelRegistry
                SharedModelRegistry.instance().register(
                    self.model_name, device_str, self.model, self.tokenizer,
                )
            except ImportError:
                pass

            self.capabilities = LLMEngineCapabilities(
                supported_modes=[
                    LLMProcessingMode.REFINE,
                    LLMProcessingMode.TRANSLATE,
                    LLMProcessingMode.CUSTOM,
                ],
                supports_gpu=self._device.type == "cuda",
                supports_batch_processing=True,
                supports_streaming=False,
                max_context_length=4096,
                memory_requirements_mb=self._estimate_memory_mb(),
            )

            self.status = LLMEngineStatus.READY
            self._logger.info("Qwen3 LLM engine ready on %s", self._device)
            return True

        except Exception as e:
            self._logger.error("Failed to initialize Qwen3 LLM engine: %s", e)
            self.status = LLMEngineStatus.ERROR
            return False

    def cleanup(self) -> None:
        should_free = True
        if self._device is not None:
            try:
                from app.llm.llm_layer import SharedModelRegistry
                should_free = SharedModelRegistry.instance().release(
                    self.model_name, str(self._device),
                )
            except ImportError:
                pass

        if should_free:
            self.model = None
            self.tokenizer = None
            try:
                from app.utils.pytorch_manager import release_gpu_memory
                release_gpu_memory()
            except ImportError:
                pass
            self._logger.info("Qwen3 LLM engine cleaned up (model freed)")
        else:
            self.model = None
            self.tokenizer = None
            self._logger.info(
                "Qwen3 LLM engine cleaned up "
                "(model kept alive — still used by another component)"
            )

        self._using_shared_model = False
        self.status = LLMEngineStatus.UNINITIALIZED

    # ------------------------------------------------------------------
    # ILLMEngine processing
    # ------------------------------------------------------------------

    def process_text(self, text: str, options: LLMProcessingOptions) -> str:
        if not self._is_available():
            self._logger.error("Engine not available for processing")
            return text

        start_ms = self._record_processing_start()
        try:
            messages = self._build_messages(text, options)
            output = self._generate(messages, options)
            self._record_processing_end(start_ms, success=True)
            return output
        except Exception as e:
            self._logger.error("Qwen3 LLM processing failed: %s", e)
            self._record_processing_end(start_ms, success=False)
            return text

    def process_batch(
        self, texts: list[str], options: LLMProcessingOptions,
    ) -> list[str]:
        results: list[str] = []
        for i, text in enumerate(texts):
            try:
                results.append(self.process_text(text, options))
            except Exception as e:
                self._logger.error("Qwen3 LLM batch item %d failed: %s", i, e)
                results.append(text)
        return results

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        self._logger.info("System prompt updated (%d chars)", len(prompt))

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(
        self, text: str, options: LLMProcessingOptions,
    ) -> list[dict[str, str]]:
        system_prompt = self._resolve_system_prompt(options)
        user_content = self._resolve_user_content(text, options)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _resolve_system_prompt(self, options: LLMProcessingOptions) -> str:
        if options.system_prompt:
            return options.system_prompt

        if self._system_prompt:
            return self._system_prompt

        template = _DEFAULT_SYSTEM_PROMPTS.get(options.mode, "")
        if not template:
            return ""

        src = _lang_name(options.source_language) if options.source_language else "source"
        tgt = _lang_name(options.target_language) if options.target_language else "target"
        return template.format(src_lang=src, tgt_lang=tgt)

    @staticmethod
    def _resolve_user_content(text: str, options: LLMProcessingOptions) -> str:
        if options.prompt_template and "{text}" in options.prompt_template:
            return options.prompt_template.format(text=text)
        return text

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[dict[str, str]],
        options: LLMProcessingOptions,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        max_tokens = options.max_tokens or self._max_tokens
        temperature = options.temperature if options.temperature > 0 else self._temperature

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.capabilities.max_context_length,
        ).to(self._device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=options.top_p if options.top_p else 0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _estimate_memory_mb(self) -> int:
        name_lower = self.model_name.lower()
        if "0.6b" in name_lower:
            return 1200
        elif "1.7b" in name_lower:
            return 3400
        elif "4b" in name_lower:
            return 8000
        elif "8b" in name_lower:
            return 16000
        return 3400
