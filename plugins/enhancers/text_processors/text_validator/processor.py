"""
Text Validator Processor

Validates OCR text to filter out nonsense before translation.
Checks if text is readable, makes sense, and is worth translating.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TextValidator:
    """
    Validates OCR text quality and readability.
    Filters out garbage text before it gets translated.
    """

    def __init__(self, dict_engine=None, enable_smart_grammar: bool = False):
        self.dict_engine = dict_engine
        self.enable_smart_grammar = enable_smart_grammar

        # Common English words for basic validation
        self.common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'can', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them',
            'what', 'when', 'where', 'who', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'one', 'two',
            'first', 'new', 'good', 'high', 'old', 'great', 'big', 'small'
        }

        # Smart Grammar Mode: Lightweight grammar patterns (English)
        self.grammar_patterns = {
            'subject_verb': [
                r'\b(i|you|we|they)\s+(am|are|have|do|will|can)\b',
                r'\b(he|she|it)\s+(is|has|does|will|can)\b',
            ],
            'article_noun': [
                r'\b(the|a|an)\s+\w+',
                r'\b(this|that|these|those)\s+\w+',
            ],
            'verb_object': [
                r'\b(is|are|was|were)\s+(a|an|the)?\s*\w+',
                r'\b(have|has|had)\s+(a|an|the)?\s*\w+',
            ],
            'prep_noun': [
                r'\b(in|on|at|to|for|with|by|from)\s+(the|a|an)?\s*\w+',
            ],
            'question': [
                r'\b(what|when|where|who|why|how)\s+',
                r'\b(do|does|did|will|can|could|would|should)\s+\w+',
            ],
        }

        # Word order anomalies (scrambled text detection)
        self.scrambled_indicators = [
            r'\b(the|a|an)\s+(is|are|was|were)\b',
            r'\b(is|are|was|were)\s+(the|a|an)\s+(is|are|was|were)\b',
        ]

        # Patterns that indicate valid text
        self.valid_patterns = [
            r'\b(the|a|an)\s+\w+',
            r'\w+\s+(is|are|was|were)\s+\w+',
            r'\w+,\s*\w+',
        ]

        # Patterns that indicate garbage
        self.garbage_patterns = [
            r'^[^a-zA-Z0-9\s]{3,}$',
            r'^[0-9\s\-_\.]{5,}$',
            r'(.)\1{4,}',
        ]

    def is_valid_text(self, text: str, min_confidence: float = 0.3) -> tuple[bool, float, str]:
        """
        Check if text is valid and worth translating.

        Returns:
            (is_valid, confidence, reason)
        """
        if not text or not text.strip():
            return False, 0.0, "Empty text"

        text = text.strip()

        if len(text) < 2:
            return False, 0.0, "Too short (< 2 chars)"

        # Short text with letters — likely a word fragment, continue
        if len(text) <= 8 and re.search(r'[a-zA-Z]{3,}', text):
            pass

        # Check for garbage patterns
        for pattern in self.garbage_patterns:
            if re.search(pattern, text):
                return False, 0.0, "Garbage pattern detected"

        # Must contain at least some letters
        if not re.search(r'[a-zA-Z]', text):
            return False, 0.0, "No letters found"

        # Calculate confidence score
        confidence = 0.0
        reasons = []

        words = text.lower().split()
        common_word_count = sum(1 for word in words if word in self.common_words)

        # Check dictionary for known words
        dict_word_count = 0
        if self.dict_engine and words:
            for word in words:
                word_clean = word.strip('.,!?;:')
                if self._is_in_dictionary(word_clean):
                    dict_word_count += 1

        if words:
            common_ratio = common_word_count / len(words)
            confidence += common_ratio * 0.3
            if common_word_count > 0:
                reasons.append(f"{common_word_count} common words")

            if dict_word_count > 0:
                dict_ratio = dict_word_count / len(words)
                confidence += dict_ratio * 0.4
                reasons.append(f"{dict_word_count} known words")

            if common_word_count == 0 and dict_word_count == 0 and len(words) >= 2:
                confidence += 0.2
                reasons.append(f"{len(words)} words")

        pattern_matches = 0
        for pattern in self.valid_patterns:
            if re.search(pattern, text.lower()):
                pattern_matches += 1
        if pattern_matches > 0:
            confidence += min(0.3, pattern_matches * 0.15)
            reasons.append(f"{pattern_matches} valid patterns")

        if self.enable_smart_grammar and len(words) >= 2:
            grammar_score = self._check_grammar_patterns(text.lower())
            if grammar_score > 0:
                confidence += grammar_score
                reasons.append(f"grammar patterns (+{grammar_score:.2f})")

        if text.isupper():
            if len(words) >= 2:
                confidence += 0.15
                reasons.append("manga-style caps")
            else:
                confidence += 0.35
                reasons.append("single caps word")
        elif text[0].isupper() or text.istitle():
            confidence += 0.1
            reasons.append("proper capitalization")

        if text.endswith('-') or text.endswith('\u2014'):
            confidence += 0.15
            reasons.append("hyphenated (continues)")

        if any(p in text for p in '.!?,;:'):
            confidence += 0.1
            reasons.append("has punctuation")

        if words and len(words) > 1:
            word_lengths = [len(w) for w in words]
            avg_length = sum(word_lengths) / len(word_lengths)
            if 3 <= avg_length <= 8:
                confidence += 0.1
                reasons.append("reasonable word lengths")

        is_valid = confidence >= min_confidence
        reason = ", ".join(reasons) if reasons else "no valid indicators"

        return is_valid, confidence, reason

    def _is_in_dictionary(self, word: str) -> bool:
        """Check if word exists in SmartDictionary."""
        if self.dict_engine and word:
            try:
                word_lower = word.lower()
                if hasattr(self.dict_engine, '_dictionary') and hasattr(self.dict_engine._dictionary, '_dictionaries'):
                    for lang_pair, dictionary in self.dict_engine._dictionary._dictionaries.items():
                        for dict_key in dictionary.keys():
                            if ':' in dict_key:
                                parts = dict_key.split(':', 2)
                                dict_text = parts[2] if len(parts) > 2 else dict_key
                            else:
                                dict_text = dict_key
                            dict_text_lower = dict_text.lower()
                            if word_lower == dict_text_lower or word_lower in dict_text_lower.split():
                                return True
            except Exception:
                pass
        return False

    def set_dictionary_engine(self, dict_engine):
        """Set the dictionary engine reference (SmartDictionary)."""
        self.dict_engine = dict_engine

    def clean_text(self, text: str) -> str:
        """Clean and normalize text with intelligent OCR error correction."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'\|', 'I', text)
        text = re.sub(r'\bl\b', 'I', text)
        text = re.sub(r'\b(when|where|while|if)\s+l\b', r'\1 I', text, flags=re.IGNORECASE)
        text = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1O\2', text)
        text = re.sub(r'\brn\b', 'm', text)
        text = re.sub(r'\bcl\b', 'd', text)
        text = ''.join(char for char in text if ord(char) < 65536)
        return text

    def should_translate(self, text: str, ocr_confidence: float = 1.0) -> tuple[bool, str]:
        """Determine if text should be sent to translation."""
        cleaned = self.clean_text(text)
        is_valid, text_confidence, reason = self.is_valid_text(cleaned)
        combined_confidence = (ocr_confidence + text_confidence) / 2

        if not is_valid:
            return False, f"Invalid text: {reason}"
        if combined_confidence < 0.3:
            return False, f"Low confidence ({combined_confidence:.2f}): {reason}"
        if len(cleaned) < 2:
            return False, "Too short after cleaning"
        if ocr_confidence > 0.85:
            return True, f"High OCR confidence ({ocr_confidence:.2f})"

        return True, f"Valid ({combined_confidence:.2f}): {reason}"

    def _check_grammar_patterns(self, text: str) -> float:
        """Check for basic grammar patterns (Smart Grammar Mode)."""
        for pattern in self.scrambled_indicators:
            if re.search(pattern, text):
                return -0.1

        matches = 0
        for category, patterns in self.grammar_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    matches += 1
                    break

        if matches >= 3:
            return 0.2
        elif matches == 2:
            return 0.15
        elif matches == 1:
            return 0.1
        return 0.0


def initialize(config: dict) -> TextValidator:
    """Initialize the text validator plugin."""
    enable_smart_grammar = config.get('enable_smart_grammar', False)
    return TextValidator(enable_smart_grammar=enable_smart_grammar)
