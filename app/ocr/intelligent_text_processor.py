"""
Intelligent Text Processor

Combines text validation, OCR error correction, and context-aware
processing for parallel OCR/translation pipelines.

Handles common OCR errors like:
- | → I (pipe to capital I)
- l → I (standalone lowercase L to capital I)
- 0 → O (zero embedded in words to capital O)
"""

import re
from typing import Any
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Processed text with corrections and confidence."""
    original: str
    corrected: str
    confidence: float
    corrections: list[str]  # List of corrections applied
    is_valid: bool
    validation_reason: str


class IntelligentTextProcessor:
    """
    Intelligent text processor that combines:
    - OCR error correction
    - Text validation
    - Context-aware processing
    """
    
    def __init__(self, dict_engine=None, enable_corrections=True, enable_context=True):
        """
        Initialize intelligent text processor.
        
        Args:
            dict_engine: SmartDictionary instance for word validation
            enable_corrections: Enable OCR error corrections
            enable_context: Enable context-aware processing
        """
        self.dict_engine = dict_engine
        self.enable_corrections = enable_corrections
        self.enable_context = enable_context
        
        # Encoding normalization (mojibake / smart quotes → ASCII apostrophe)
        self.encoding_fixes = {
            r'æ': "'",
            r'Ã¦': "'",
            r'â€™': "'",
            r'Ã¢â‚¬â„¢': "'",
            r'\u2018': "'",  # LEFT SINGLE QUOTATION MARK
            r'\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
            r'\u2018': "'",  # LEFT SINGLE QUOTATION MARK (literal)
            r'\u2019': "'",  # RIGHT SINGLE QUOTATION MARK (literal)
        }
        
        # Common OCR error patterns (safe — won't mangle real words)
        self.ocr_corrections = {
            # Pipe/vertical bar to I
            r'\|': 'I',
            # Standalone lowercase L to I
            r'\bl\b': 'I',
            # Zero in middle of words to O
            r'([a-zA-Z])0([a-zA-Z])': r'\1O\2',
        }
        
        # Context patterns for better correction
        self.context_patterns = {
            # "When | was" → "When I was"
            r'\b(when|where|while|if)\s+\|': r'\1 I',
            # "| am" → "I am"
            r'^\|\s+(am|was|will|can|have)': r'I \1',
            # "at home" with pipe → "at home" with I
            r'\|\s+(am|was|at|in|on)': r'I \1',
        }
        
        # Common word patterns for validation (including short words and contractions)
        self.common_words = {
            # 1-letter words
            'i', 'a',
            # 2-letter words
            'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'ho', 
            'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'oh', 'on', 'or', 
            'so', 'to', 'uh', 'up', 'us', 'we', 'ye', 'yo',
            # 3-letter words
            'and', 'the', 'for', 'you', 'not', 'but', 'all', 'any', 'can', 'out',
            'get', 'him', 'her', 'who', 'man', 'new', 'now', 'one', 'two', 'she',
            'has', 'had', 'was', 'are', 'see', 'way', 'why', 'how', 'day', 'too',
            'big', 'bad', 'far', 'few', 'say', 'run', 'fun', 'let', 'old', 'own',
            # Longer common words
            'with', 'from', 'were', 'been', 'have', 'does', 'will', 'would', 
            'could', 'should', 'might', 'must', 'this', 'that', 'these', 'those',
            'they', 'your', 'his', 'its', 'our', 'their', 'them', 'what', 'when', 
            'where', 'each', 'every', 'both', 'more', 'most', 'other', 'some', 
            'such', 'only', 'same', 'than', 'very', 'first', 'good', 'high', 
            'great', 'small', 'home', 'did', 'may',
            # Common contractions (without apostrophe for matching)
            "dont", "doesnt", "didnt", "wont", "wouldnt", "couldnt", "shouldnt",
            "cant", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt",
            "im", "ive", "id", "ill", "youre", "youve", "youd", "youll",
            "hes", "shes", "its", "were", "theyre", "theyve", "theyd", "theyll"
        }
        
        # Statistics
        self.total_processed = 0
        self.total_corrected = 0
        self.total_validated = 0
        self.total_rejected = 0
    
    def process_text(self, text: str, context: str | None = None, 
                    ocr_confidence: float = 1.0) -> ProcessedText:
        """
        Process text with intelligent corrections and validation.
        
        Args:
            text: Raw OCR text
            context: Optional context (previous/next text)
            ocr_confidence: OCR engine confidence score
            
        Returns:
            ProcessedText with corrections and validation
        """
        self.total_processed += 1
        
        original = text
        corrected = text
        corrections = []
        
        # Step 1: Normalize encoding (mojibake → ASCII)
        for pattern, replacement in self.encoding_fixes.items():
            if re.search(pattern, corrected):
                new_text = re.sub(pattern, replacement, corrected)
                if new_text != corrected:
                    corrections.append(f"Encoding: '{pattern}' → '{replacement}'")
                    corrected = new_text
        
        # Step 2: Apply context-aware corrections (higher priority)
        if self.enable_context and context:
            for pattern, replacement in self.context_patterns.items():
                if re.search(pattern, corrected, re.IGNORECASE):
                    new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                    if new_text != corrected:
                        corrections.append(f"Context: '{pattern}' → '{replacement}'")
                        corrected = new_text
        
        # Step 3: Apply general OCR corrections
        if self.enable_corrections:
            for pattern, replacement in self.ocr_corrections.items():
                if re.search(pattern, corrected):
                    new_text = re.sub(pattern, replacement, corrected)
                    if new_text != corrected:
                        corrections.append(f"OCR: '{pattern}' → '{replacement}'")
                        corrected = new_text
        
        # Step 4: Validate corrected text
        is_valid, confidence, reason = self._validate_text(corrected, ocr_confidence)
        
        if is_valid:
            self.total_validated += 1
        else:
            self.total_rejected += 1
        
        if corrections:
            self.total_corrected += 1
        
        return ProcessedText(
            original=original,
            corrected=corrected,
            confidence=confidence,
            corrections=corrections,
            is_valid=is_valid,
            validation_reason=reason
        )
    
    def process_batch(self, texts: list[dict[str, Any]]) -> list[ProcessedText]:
        """
        Process a batch of texts with context awareness.
        
        Args:
            texts: List of text dictionaries with 'text', 'bbox', 'confidence'
            
        Returns:
            List of ProcessedText objects
        """
        if not texts:
            return []
        
        # DO NOT SORT - Keep original OCR order (Tesseract provides correct reading order)
        # Sorting by bbox breaks the order for curved/manga text
        sorted_texts = texts
        
        processed = []
        
        for i, text_item in enumerate(sorted_texts):
            text = text_item.get('text', '')
            ocr_conf = text_item.get('confidence', 1.0)
            
            # Get context from adjacent texts
            context = None
            if i > 0:
                prev_text = sorted_texts[i-1].get('text', '')
                context = prev_text
            
            # Process with context
            result = self.process_text(text, context, ocr_conf)
            processed.append(result)
        
        return processed
    
    def _validate_text(self, text: str, ocr_confidence: float) -> tuple[bool, float, str]:
        """
        Validate text quality.
        
        Args:
            text: Text to validate
            ocr_confidence: OCR confidence score
            
        Returns:
            (is_valid, confidence, reason)
        """
        if not text or not text.strip():
            return False, 0.0, "Empty text"
        
        text = text.strip()
        
        # Minimum length check (allow single-letter valid words like "I" and "A")
        if len(text) < 2 and text.upper() not in ['I', 'A']:
            return False, 0.0, "Too short"
        
        # Must contain letters
        if not re.search(r'[a-zA-Z]', text):
            return False, 0.0, "No letters"
        
        # Calculate confidence
        confidence = 0.0
        reasons = []
        
        # Check for common words (also check without apostrophes for contractions)
        words = text.lower().split()
        common_count = 0
        for w in words:
            w_clean = w.strip('.,!?;:')
            # Check as-is
            if w_clean in self.common_words:
                common_count += 1
            # Also check without apostrophes/special chars (for contractions with encoding issues)
            elif w_clean.replace("'", "").replace("æ", "").replace("â€™", "") in self.common_words:
                common_count += 1
        
        if words:
            common_ratio = common_count / len(words)
            confidence += common_ratio * 0.4
            if common_count > 0:
                reasons.append(f"{common_count} common words")
        
        # Capitalization bonus
        if text[0].isupper() or text.isupper():
            confidence += 0.2
            reasons.append("proper capitalization")
        
        # Combine with OCR confidence
        combined_confidence = (confidence + ocr_confidence) / 2
        
        is_valid = combined_confidence >= 0.3
        reason = ", ".join(reasons) if reasons else "no indicators"
        
        return is_valid, combined_confidence, reason
    
    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
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
            'rejection_rate': f"{rejection_rate:.1f}%"
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_processed = 0
        self.total_corrected = 0
        self.total_validated = 0
        self.total_rejected = 0
