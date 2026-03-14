"""
Intelligent Text Processor Optimizer Plugin

Combines OCR error correction, text validation, and smart dictionary lookup
for parallel OCR/translation processing.

Handles:
- OCR error correction (| -> I, 0 -> O, rn -> m, etc.)
- Context-aware corrections
- Text validation
- Smart dictionary integration
- Parallel processing safety
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

app_path = Path(__file__).parent.parent.parent.parent / "app"
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

from app.ocr.intelligent_text_processor import IntelligentTextProcessor


def _normalize_block(block: Any) -> Dict[str, Any]:
    """Convert a TextBlock object or dict to the dict format expected by
    ``IntelligentTextProcessor.process_batch()`` (keys: text, bbox, confidence).
    """
    if isinstance(block, dict):
        return block

    text = getattr(block, 'text', str(block))
    confidence = getattr(block, 'confidence', 1.0)

    position = getattr(block, 'position', None)
    if position is not None:
        bbox = [
            getattr(position, 'x', 0),
            getattr(position, 'y', 0),
            getattr(position, 'width', 0),
            getattr(position, 'height', 0),
        ]
    else:
        bbox = [0, 0, 0, 0]

    return {
        'text': text,
        'bbox': bbox,
        'confidence': confidence,
    }


class IntelligentTextProcessorOptimizer:
    """Intelligent text processor optimizer for parallel processing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_corrections = config.get('enable_corrections', True)
        self.enable_context = config.get('enable_context', True)
        self.enable_validation = config.get('enable_validation', True)
        self.min_confidence = config.get('min_confidence', 0.3)
        self.auto_learn = config.get('auto_learn', True)

        self.processor = IntelligentTextProcessor(
            dict_engine=None,
            enable_corrections=self.enable_corrections,
            enable_context=self.enable_context
        )

        # Statistics
        self.total_processed = 0
        self.total_corrected = 0
        self.total_validated = 0
        self.total_rejected = 0

        logger.info(
            "[INTELLIGENT_PROCESSOR] Initialized (corrections=%s, context=%s, "
            "validation=%s, min_confidence=%.2f)",
            self.enable_corrections,
            self.enable_context,
            self.enable_validation,
            self.min_confidence,
        )

    def set_dict_engine(self, dict_engine):
        """Set smart dictionary engine reference."""
        self.processor.dict_engine = dict_engine
        logger.info("[INTELLIGENT_PROCESSOR] Smart dictionary connected")

    def configure(self, config: Dict[str, Any]):
        """Update configuration dynamically."""
        if 'min_confidence' in config:
            self.min_confidence = config['min_confidence']
            self.config['min_confidence'] = config['min_confidence']
        if 'min_word_length' in config:
            self.config['min_word_length'] = config['min_word_length']
        if 'enable_corrections' in config:
            self.enable_corrections = config['enable_corrections']
            self.processor.enable_corrections = config['enable_corrections']
        if 'enable_context' in config:
            self.enable_context = config['enable_context']
            self.processor.enable_context = config['enable_context']
        if 'enable_validation' in config:
            self.enable_validation = config['enable_validation']

        logger.info(
            "[INTELLIGENT_PROCESSOR] Configuration updated: min_conf=%.2f, "
            "min_word_len=%d",
            self.min_confidence,
            self.config.get('min_word_length', 2),
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process: Apply intelligent text processing to OCR output."""
        text_blocks = data.get('text_blocks', [])

        if not text_blocks:
            return data

        normalized = [_normalize_block(b) for b in text_blocks]

        processed = self.processor.process_batch(normalized)

        validated_texts: List[Dict[str, Any]] = []
        corrections_applied: List[Dict[str, Any]] = []

        for proc, text_dict in zip(processed, normalized):
            self.total_processed += 1

            if proc.corrections:
                self.total_corrected += 1
                corrections_applied.append({
                    'original': proc.original,
                    'corrected': proc.corrected,
                    'corrections': proc.corrections
                })

            if self.enable_validation:
                min_word_length = self.config.get('min_word_length', 2)
                text_length = len(proc.corrected.strip())

                is_valid_single_letter = text_length == 1 and proc.corrected.strip().upper() in ['I', 'A']

                has_cjk = any(ord(c) > 0x2E80 for c in proc.corrected)

                min_conf_threshold = self.min_confidence
                is_valid_check = proc.is_valid

                if proc.corrections:
                    min_conf_threshold = 0.3
                    if proc.confidence >= 0.3:
                        is_valid_check = True

                if has_cjk or (is_valid_check and proc.confidence >= min_conf_threshold and (text_length >= min_word_length or is_valid_single_letter)):
                    updated = text_dict.copy()
                    updated['text'] = proc.corrected
                    updated['original_text'] = proc.original
                    updated['corrections'] = proc.corrections
                    updated['validation_confidence'] = proc.confidence
                    updated['validation_reason'] = proc.validation_reason
                    validated_texts.append(updated)
                    self.total_validated += 1
                else:
                    self.total_rejected += 1
                    reason = proc.validation_reason
                    if text_length < min_word_length:
                        reason = f"Too short ({text_length} < {min_word_length})"
                    logger.debug(
                        "[INTELLIGENT_PROCESSOR] Rejected: '%.30s...' "
                        "(confidence=%.2f, reason=%s)",
                        proc.original,
                        proc.confidence,
                        reason,
                    )
            else:
                updated = text_dict.copy()
                updated['text'] = proc.corrected
                updated['original_text'] = proc.original
                updated['corrections'] = proc.corrections
                validated_texts.append(updated)
                self.total_validated += 1

        data['text_blocks'] = validated_texts
        data['corrections_applied'] = corrections_applied
        data['filtered_count'] = len(text_blocks) - len(validated_texts)

        if corrections_applied:
            logger.debug(
                "[INTELLIGENT_PROCESSOR] Applied %d corrections",
                len(corrections_applied),
            )

        if self.enable_validation:
            logger.debug(
                "[INTELLIGENT_PROCESSOR] Validated %d/%d texts",
                len(validated_texts),
                len(text_blocks),
            )

        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        correction_rate = (self.total_corrected / self.total_processed * 100) if self.total_processed > 0 else 0
        validation_rate = (self.total_validated / self.total_processed * 100) if self.total_processed > 0 else 0
        rejection_rate = (self.total_rejected / self.total_processed * 100) if self.total_processed > 0 else 0

        return {
            'total_processed': self.total_processed,
            'total_corrected': self.total_corrected,
            'total_validated': self.total_validated,
            'total_rejected': self.total_rejected,
            'correction_rate': f"{correction_rate:.1f}%",
            'validation_rate': f"{validation_rate:.1f}%",
            'rejection_rate': f"{rejection_rate:.1f}%",
            'enable_corrections': self.enable_corrections,
            'enable_context': self.enable_context,
            'enable_validation': self.enable_validation
        }

    def reset(self):
        """Reset optimizer state."""
        self.total_processed = 0
        self.total_corrected = 0
        self.total_validated = 0
        self.total_rejected = 0
        self.processor.reset_stats()


# Plugin interface
def initialize(config: Dict[str, Any]) -> IntelligentTextProcessorOptimizer:
    """Initialize the optimizer plugin."""
    return IntelligentTextProcessorOptimizer(config)
