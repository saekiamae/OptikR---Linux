"""
Translation Quality Filter

Prevents low-quality translations from being saved to the dictionary.
Filters out scuffy, garbage, or unreliable translations.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Languages that don't use spaces between words
_CJK_LANGUAGES = frozenset({'ja', 'zh', 'zh-cn', 'zh-tw', 'ko', 'th', 'my', 'km', 'lo'})

# Languages that don't have upper/lower case distinction
_CASELESS_LANGUAGES = frozenset({
    'ja', 'zh', 'zh-cn', 'zh-tw', 'ko', 'th', 'ar', 'he', 'hi',
    'my', 'km', 'lo', 'ka', 'am', 'ti',
})


def _is_letter_or_digit(ch: str) -> bool:
    """Return True if the character is a Unicode letter or digit (any script)."""
    cat = unicodedata.category(ch)
    return cat.startswith('L') or cat.startswith('N')


def _special_char_ratio(text: str) -> float:
    """Ratio of non-letter/digit/whitespace characters in *text* (Unicode-aware)."""
    if not text:
        return 0.0
    special = sum(1 for ch in text if not _is_letter_or_digit(ch) and not ch.isspace())
    return special / len(text)


class TranslationQualityFilter:
    """
    Filters translations before saving to dictionary.

    Prevents:
    - Very low confidence translations
    - Translations that are identical to source
    - Translations with too many special characters (Unicode-aware)
    - Translations that are too short or too long
    - Translations with suspicious patterns
    """

    def __init__(self, config: dict | None = None):
        """Initialize quality filter with configuration."""
        self.config = config or {}

        # Thresholds
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_length = self.config.get('min_length', 2)
        self.max_special_char_ratio = self.config.get('max_special_char_ratio', 0.5)
        self.min_word_count = self.config.get('min_word_count', 1)

        # Patterns that indicate bad translations (language-independent)
        self.bad_patterns = [
            r'^[\d\s\-_\.]{5,}$',       # Only numbers and punctuation
            r'(.)\1{4,}',               # Repeated character 5+ times (aaaaa)
        ]

    def should_save(self, original: str, translation: str, confidence: float,
                    source_lang: str, target_lang: str) -> tuple[bool, str | None]:
        """
        Check if translation should be saved to dictionary.

        Args:
            original: Original text
            translation: Translated text
            confidence: Translation confidence (0-1)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Tuple of (should_save: bool, reason: str | None)
        """
        # Check 1: Confidence too low
        if confidence < self.min_confidence:
            reason = f"Confidence too low ({confidence:.2f} < {self.min_confidence})"
            logger.debug("Rejected: %s", reason)
            return False, reason

        # Check 2: Translation is empty
        if not translation or not translation.strip():
            logger.debug("Rejected: Translation is empty")
            return False, "Translation is empty"

        # Check 3: Translation is identical to original (no translation happened)
        if translation.strip().lower() == original.strip().lower():
            logger.debug("Rejected: Translation identical to original")
            return False, "Translation identical to original"

        # Check 4: Translation too short
        stripped = translation.strip()
        if len(stripped) < self.min_length:
            reason = f"Translation too short ({len(stripped)} < {self.min_length})"
            logger.debug("Rejected: %s", reason)
            return False, reason

        # Check 5: Too many special characters (Unicode-aware)
        special_ratio = _special_char_ratio(translation)
        if special_ratio > self.max_special_char_ratio:
            reason = f"Too many special characters ({special_ratio:.1%} > {self.max_special_char_ratio:.1%})"
            logger.debug("Rejected: %s", reason)
            return False, reason

        # Check 6: Not enough words (skip for CJK — they don't use spaces)
        if target_lang not in _CJK_LANGUAGES:
            words = translation.split()
            if len(words) < self.min_word_count:
                reason = f"Not enough words ({len(words)} < {self.min_word_count})"
                logger.debug("Rejected: %s", reason)
                return False, reason

        # Check 7: Bad patterns
        for pattern in self.bad_patterns:
            if re.search(pattern, translation):
                reason = f"Matches bad pattern: {pattern}"
                logger.debug("Rejected: %s", reason)
                return False, reason

        # Check 8: Translation has suspicious character repetition
        # Example: "aaaaaaa" or "111111"
        if len(set(translation.replace(' ', ''))) < 3 and len(translation) > 5:
            logger.debug("Rejected: Too few unique characters")
            return False, "Translation has too few unique characters"

        # All checks passed
        logger.debug("Accepted: '%s' -> '%s' (conf=%.2f)", original[:30], translation[:30], confidence)
        return True, None

    def get_quality_score(self, original: str, translation: str, confidence: float) -> float:
        """
        Calculate a quality score for the translation (0-1).

        Args:
            original: Original text
            translation: Translated text
            confidence: Translation confidence

        Returns:
            Quality score (0-1, higher is better)
        """
        score = confidence

        # Bonus for reasonable length
        if 5 <= len(translation) <= 100:
            score += 0.05

        # Bonus for having multiple words
        if len(translation.split()) >= 2:
            score += 0.05

        # Bonus for low special character ratio (Unicode-aware)
        if _special_char_ratio(translation) < 0.2:
            score += 0.05

        # Penalty for all caps (only meaningful for cased scripts)
        if translation.isupper() and len(translation) > 5:
            score -= 0.1

        # Penalty for identical to original
        if translation.lower() == original.lower():
            score -= 0.3

        return min(1.0, max(0.0, score))


# Default instance with standard settings
default_quality_filter = TranslationQualityFilter()

# Strict instance for high-quality only
strict_quality_filter = TranslationQualityFilter({
    'min_confidence': 0.85,
    'min_length': 3,
    'max_special_char_ratio': 0.3,
    'min_word_count': 1,
})
