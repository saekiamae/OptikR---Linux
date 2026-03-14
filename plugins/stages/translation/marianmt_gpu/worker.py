"""
Translation Worker - Subprocess for text translation.

Runs in a separate process, translates text and sends results back
via the BaseWorker JSON protocol (stdin/stdout).
"""

import sys
from pathlib import Path

# Add project root to path for subprocess execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    from transformers import MarianMTModel, MarianTokenizer
    MARIANMT_AVAILABLE = True
except ImportError:
    MARIANMT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TranslationWorker(BaseWorker):
    """Worker for translation using MarianMT with GPU/CPU support."""

    def initialize(self, config: dict) -> bool:
        """Initialize translation model with device and config settings."""
        try:
            if not MARIANMT_AVAILABLE:
                self.log("MarianMT not available (install transformers + sentencepiece)")
                return False

            source_lang = config.get('source_language', 'en')
            target_lang = config.get('target_language', 'de')
            self._max_length = int(config.get('max_length', 512))
            self._num_beams = int(config.get('num_beams', 4))
            self._source_lang = source_lang
            self._target_lang = target_lang

            # Device selection
            use_gpu = config.get('gpu', True)
            runtime_mode = config.get('runtime_mode', 'auto')
            if runtime_mode == 'cpu':
                use_gpu = False
            elif runtime_mode == 'gpu':
                use_gpu = True

            if TORCH_AVAILABLE and use_gpu and torch.cuda.is_available():
                self._device = torch.device('cuda')
                self.log("Using GPU (CUDA)")
            else:
                self._device = torch.device('cpu') if TORCH_AVAILABLE else None
                self.log("Using CPU")

            # Load model
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            self.log(f"Loading model: {model_name}")

            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            try:
                self.model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
            except Exception:
                self.model = MarianMTModel.from_pretrained(model_name, use_safetensors=False)

            if self._device is not None:
                self.model = self.model.to(self._device)

            self.log(f"MarianMT initialized: {source_lang}->{target_lang} on {self._device}")
            return True

        except Exception as e:
            self.log(f"Failed to initialize: {e}")
            return False

    def process(self, data: dict) -> dict:
        """
        Translate text blocks.

        Args:
            data: {
                'text_blocks': [{'text': str, 'bbox': [...], 'confidence': float}],
                'source_language': str,
                'target_language': str
            }

        Returns:
            {
                'translations': [{'original_text': str, 'translated_text': str, 'bbox': [...]}],
                'count': int
            }
        """
        try:
            text_blocks = data.get('text_blocks', [])
            if not text_blocks:
                return {'translations': [], 'count': 0}

            texts = [block['text'] for block in text_blocks]

            # Tokenize and move to device
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self._max_length,
            )
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate
            translated = self.model.generate(
                **inputs,
                max_length=self._max_length,
                num_beams=self._num_beams,
                early_stopping=True,
            )
            translated_texts = [
                self.tokenizer.decode(t, skip_special_tokens=True) for t in translated
            ]

            translations = []
            for block, translated_text in zip(text_blocks, translated_texts):
                translations.append({
                    'original_text': block['text'],
                    'translated_text': translated_text,
                    'bbox': block.get('bbox', []),
                })

            return {'translations': translations, 'count': len(translations)}

        except Exception as e:
            return {'error': f'Translation failed: {e}'}

    def cleanup(self):
        """Clean up translation resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        self.log("Translation worker shutdown")


if __name__ == '__main__':
    worker = TranslationWorker(name="TranslationWorker")
    worker.run()
