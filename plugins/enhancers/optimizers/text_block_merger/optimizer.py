"""
Text Block Merger Optimizer Plugin
Intelligently merges nearby text blocks into complete sentences.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _normalize_block(block: Any) -> Dict[str, Any]:
    """Convert a TextBlock object or dict to the standard dict format.

    Handles both ``TextBlock`` dataclass instances (with ``.text``,
    ``.position`` as ``Rectangle``, ``.confidence``) and plain dicts.
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


class TextBlockMerger:
    """Merges nearby text blocks based on proximity and layout."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.horizontal_threshold = config.get('horizontal_threshold', 50)
        self.vertical_threshold = config.get('vertical_threshold', 30)
        self.line_height_tolerance = config.get('line_height_tolerance', 1.5)
        self.merge_strategy = config.get('merge_strategy', 'smart')
        self.respect_punctuation = config.get('respect_punctuation', True)
        self.min_confidence = config.get('min_confidence', 0.3)

        # Statistics
        self.total_blocks_in = 0
        self.total_blocks_out = 0
        self.total_merges = 0

        logger.info(
            "[TEXT_BLOCK_MERGER] Initialized (h=%dpx, v=%dpx, strategy=%s)",
            self.horizontal_threshold,
            self.vertical_threshold,
            self.merge_strategy,
        )

    def configure(self, config: Dict[str, Any]):
        """Update configuration dynamically."""
        if 'horizontal_threshold' in config:
            self.horizontal_threshold = config['horizontal_threshold']
        if 'vertical_threshold' in config:
            self.vertical_threshold = config['vertical_threshold']
        if 'merge_strategy' in config:
            self.merge_strategy = config['merge_strategy']
        if 'line_height_tolerance' in config:
            self.line_height_tolerance = config['line_height_tolerance']
        if 'respect_punctuation' in config:
            self.respect_punctuation = config['respect_punctuation']
        if 'min_confidence' in config:
            self.min_confidence = config['min_confidence']

        logger.info(
            "[TEXT_BLOCK_MERGER] Configuration updated: h=%dpx, v=%dpx, strategy=%s",
            self.horizontal_threshold,
            self.vertical_threshold,
            self.merge_strategy,
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process: Merge nearby text blocks."""
        text_blocks = data.get('text_blocks', [])

        if not text_blocks or len(text_blocks) <= 1:
            return data

        normalized = [_normalize_block(b) for b in text_blocks]
        self.total_blocks_in += len(normalized)

        merged = self._merge_text_blocks(normalized)

        self.total_blocks_out += len(merged)
        self.total_merges += (len(normalized) - len(merged))

        data['text_blocks'] = merged
        data['merge_count'] = len(normalized) - len(merged)

        logger.debug(
            "[TEXT_BLOCK_MERGER] Merged %d blocks -> %d blocks",
            len(normalized),
            len(merged),
        )

        return data

    def _merge_text_blocks(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge nearby text blocks intelligently."""
        if not texts:
            return texts

        valid_texts = [t for t in texts if t.get('confidence', 1.0) >= self.min_confidence]

        if not valid_texts:
            return texts

        sorted_texts = valid_texts

        if not sorted_texts:
            return []

        merged: List[Dict[str, Any]] = []
        current_group = [sorted_texts[0]]

        for i in range(1, len(sorted_texts)):
            prev = current_group[-1]
            curr = sorted_texts[i]

            prev_bbox = prev['bbox']
            curr_bbox = curr['bbox']

            prev_bottom = prev_bbox[1] + prev_bbox[3]
            curr_top = curr_bbox[1]
            vertical_gap = abs(curr_top - prev_bottom)

            prev_right = prev_bbox[0] + prev_bbox[2]
            curr_left = curr_bbox[0]
            horizontal_gap = abs(curr_left - prev_right)

            should_merge = (horizontal_gap <= self.horizontal_threshold or
                            vertical_gap <= self.vertical_threshold)

            if should_merge and not self.respect_punctuation:
                current_group.append(curr)
            elif should_merge and self.respect_punctuation:
                prev_text = current_group[-1]['text'].strip()
                if prev_text and prev_text[-1] not in '.!?。！？':
                    current_group.append(curr)
                else:
                    merged.append(self._combine_group(current_group))
                    current_group = [curr]
            else:
                merged.append(self._combine_group(current_group))
                current_group = [curr]

        if current_group:
            merged.append(self._combine_group(current_group))

        return merged

    def _group_into_lines(self, texts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group text blocks into horizontal lines."""
        if not texts:
            return []

        lines: List[List[Dict[str, Any]]] = []
        current_line = [texts[0]]

        for text in texts[1:]:
            prev_bbox = current_line[-1]['bbox']
            curr_bbox = text['bbox']

            prev_y_center = prev_bbox[1] + prev_bbox[3] / 2
            curr_y_center = curr_bbox[1] + curr_bbox[3] / 2

            avg_height = (prev_bbox[3] + curr_bbox[3]) / 2
            max_y_diff = avg_height * self.line_height_tolerance

            if abs(curr_y_center - prev_y_center) <= max_y_diff:
                current_line.append(text)
            else:
                lines.append(current_line)
                current_line = [text]

        if current_line:
            lines.append(current_line)

        return lines

    def _merge_line(self, line: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks within a single line."""
        if len(line) <= 1:
            return line

        sorted_line = line

        merged: List[Dict[str, Any]] = []
        current_group = [sorted_line[0]]

        for text in sorted_line[1:]:
            prev_bbox = current_group[-1]['bbox']
            curr_bbox = text['bbox']

            prev_right = prev_bbox[0] + prev_bbox[2]
            curr_left = curr_bbox[0]
            horizontal_gap = curr_left - prev_right

            should_merge = False

            if horizontal_gap <= self.horizontal_threshold:
                if self.respect_punctuation:
                    prev_text = current_group[-1]['text'].strip()
                    if prev_text and prev_text[-1] in '.!?。！？':
                        should_merge = False
                    else:
                        should_merge = True
                else:
                    should_merge = True

            if should_merge:
                current_group.append(text)
            else:
                merged.append(self._combine_group(current_group))
                current_group = [text]

        if current_group:
            merged.append(self._combine_group(current_group))

        return merged

    def _merge_across_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks across lines (for multi-line sentences)."""
        if len(lines) <= 1:
            return lines

        merged: List[Dict[str, Any]] = []
        current_group = [lines[0]]

        for text in lines[1:]:
            prev_bbox = current_group[-1]['bbox']
            curr_bbox = text['bbox']

            prev_bottom = prev_bbox[1] + prev_bbox[3]
            curr_top = curr_bbox[1]
            vertical_gap = curr_top - prev_bottom

            should_merge = False

            if vertical_gap <= self.vertical_threshold:
                if self.respect_punctuation:
                    prev_text = current_group[-1]['text'].strip()
                    if prev_text and prev_text[-1] in '.!?。！？':
                        should_merge = False
                    else:
                        should_merge = True
                else:
                    should_merge = True

            if should_merge:
                current_group.append(text)
            else:
                merged.append(self._combine_group(current_group))
                current_group = [text]

        if current_group:
            merged.append(self._combine_group(current_group))

        return merged

    def _combine_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple text blocks into one."""
        if len(group) == 1:
            return group[0]

        combined_text = ' '.join(t['text'] for t in group)

        min_x = min(t['bbox'][0] for t in group)
        min_y = min(t['bbox'][1] for t in group)
        max_x = max(t['bbox'][0] + t['bbox'][2] for t in group)
        max_y = max(t['bbox'][1] + t['bbox'][3] for t in group)

        width = max_x - min_x
        height = max_y - min_y

        combined_bbox = [min_x, min_y, width, height]

        avg_confidence = sum(t.get('confidence', 1.0) for t in group) / len(group)

        return {
            'text': combined_text,
            'bbox': combined_bbox,
            'confidence': avg_confidence,
            'merged_from': len(group)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        reduction_rate = (self.total_merges / self.total_blocks_in * 100) if self.total_blocks_in > 0 else 0

        return {
            'total_blocks_in': self.total_blocks_in,
            'total_blocks_out': self.total_blocks_out,
            'total_merges': self.total_merges,
            'reduction_rate': f"{reduction_rate:.1f}%",
            'horizontal_threshold': self.horizontal_threshold,
            'vertical_threshold': self.vertical_threshold,
            'strategy': self.merge_strategy
        }

    def reset(self):
        """Reset optimizer state."""
        self.total_blocks_in = 0
        self.total_blocks_out = 0
        self.total_merges = 0


# Plugin interface
def initialize(config: Dict[str, Any]) -> TextBlockMerger:
    """Initialize the optimizer plugin."""
    return TextBlockMerger(config)
