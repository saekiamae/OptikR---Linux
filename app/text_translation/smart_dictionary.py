"""
Intelligent Local Dictionary Module

Advanced dictionary-based translation system with:
- Machine learning-based quality scoring
- Context-aware translation selection
- Automatic learning from AI translations
- Smart fuzzy matching with multiple algorithms
- Confidence decay over time
- Usage pattern analysis
- Multi-variant translation support
- Intelligent entry merging
"""

import gzip
import json
import logging
import re
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from collections import defaultdict
import difflib

# Import path utilities for EXE compatibility
from app.utils.path_utils import get_dictionary_dir

# Placeholder pattern used by the Context Manager to mask locked terms.
# Entries containing these markers must be rejected to avoid polluting
# the dictionary with transient placeholder artifacts.
_PLACEHOLDER_RE = re.compile(r"\u27E6[TR].*?\u27E7")  # matches ⟦T000⟧, ⟦R…⟧ etc.


@dataclass
class DictionaryEntry:
    """
    Advanced dictionary entry with machine learning features.
    
    Tracks multiple translation variants, context, quality metrics,
    and temporal decay of confidence.
    """
    source_text: str
    translation: str
    source_language: str
    target_language: str
    usage_count: int
    confidence: float
    last_used: str  # ISO format datetime string
    
    # Advanced features
    variants: list[str] = field(default_factory=list)  # Alternative translations
    context_tags: set[str] = field(default_factory=set)  # Context keywords
    quality_score: float = 0.0  # ML-based quality (0-1)
    source_engine: str = "manual"  # Origin: manual, ai, learned
    creation_date: str = ""  # When first added
    success_rate: float = 1.0  # How often this translation was accepted
    avg_confidence: float = 0.0  # Average confidence over time
    decay_factor: float = 1.0  # Temporal decay (1.0 = fresh, 0.0 = stale)
    
    def __post_init__(self):
        """Initialize computed fields."""
        if not self.creation_date:
            self.creation_date = datetime.now().isoformat()
        if self.avg_confidence == 0.0:
            self.avg_confidence = self.confidence
        self._update_decay_factor()
    
    def _update_decay_factor(self):
        """Update confidence decay based on time since last use."""
        try:
            last_used_dt = datetime.fromisoformat(self.last_used)
            days_since_use = (datetime.now() - last_used_dt).days
            
            # Decay formula: confidence decreases 5% per month of non-use
            # After 6 months: ~75%, after 1 year: ~55%, after 2 years: ~30%
            self.decay_factor = max(0.3, 1.0 - (days_since_use / 30) * 0.05)
        except (ValueError, TypeError):
            self.decay_factor = 1.0
    
    def get_effective_confidence(self) -> float:
        """Get confidence adjusted for decay and success rate."""
        self._update_decay_factor()
        return self.confidence * self.decay_factor * self.success_rate
    
    def add_variant(self, variant: str):
        """Add alternative translation."""
        if variant not in self.variants and variant != self.translation:
            self.variants.append(variant)
    
    def add_context(self, context: str):
        """Add context tag."""
        # Extract keywords from context
        words = re.findall(r'\w+', context.lower())
        self.context_tags.update(words[:5])  # Keep top 5 keywords
    
    def update_usage(self, success: bool = True):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update decay factor
        self._update_decay_factor()
    
    def merge_with(self, other: 'DictionaryEntry'):
        """Intelligently merge with another entry."""
        # Keep higher confidence translation as primary
        if other.confidence > self.confidence:
            self.variants.append(self.translation)
            self.translation = other.translation
            self.confidence = other.confidence
        else:
            self.variants.append(other.translation)
        
        # Merge statistics
        self.usage_count += other.usage_count
        self.context_tags.update(other.context_tags)
        
        # Update quality score (weighted average)
        total_usage = self.usage_count + other.usage_count
        self.quality_score = (
            (self.quality_score * self.usage_count + other.quality_score * other.usage_count) / total_usage
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'translation': self.translation,
            'usage_count': self.usage_count,
            'confidence': self.confidence,
            'last_used': self.last_used,
            'variants': list(self.variants) if self.variants else [],
            'context_tags': list(self.context_tags) if self.context_tags else [],
            'quality_score': self.quality_score,
            'source_engine': self.source_engine,
            'creation_date': self.creation_date,
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence
        }
    
    @classmethod
    def from_dict(cls, source_text: str, data: dict, source_lang: str, target_lang: str):
        """Create from dictionary format."""
        return cls(
            source_text=source_text,
            translation=data.get('translation', ''),
            source_language=source_lang,
            target_language=target_lang,
            usage_count=data.get('usage_count', 0),
            confidence=data.get('confidence', 0.0),
            last_used=data.get('last_used', datetime.now().isoformat()),
            variants=data.get('variants', []),
            context_tags=set(data.get('context_tags', [])),
            quality_score=data.get('quality_score', 0.0),
            source_engine=data.get('source_engine', 'manual'),
            creation_date=data.get('creation_date', datetime.now().isoformat()),
            success_rate=data.get('success_rate', 1.0),
            avg_confidence=data.get('avg_confidence', data.get('confidence', 0.0))
        )


@dataclass
class DictionaryStats:
    """Dictionary statistics."""
    total_entries: int = 0
    total_usage: int = 0
    average_usage: float = 0.0
    total_lookups: int = 0
    cache_hits: int = 0
    most_used: list[dict] = None
    
    def __post_init__(self):
        if self.most_used is None:
            self.most_used = []


# Sentinel for negative cache entries (distinguishes "not in cache" from "looked up, not found")
_CACHE_MISS = object()


class DictionaryLookupCache:
    """Simple LRU cache for dictionary lookups."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.max_size = max_size
        self._cache: dict[str, object] = {}
        self._access_order: list[str] = []
        self._lock = RLock()
    
    def get(self, key: str):
        """Get cached entry. Returns _CACHE_MISS sentinel for negative hits, None if not in cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        return None
    
    def put(self, key: str, entry):
        """Cache entry. Use _CACHE_MISS sentinel for negative cache entries."""
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add to cache
            self._cache[key] = entry
            self._access_order.append(key)
            
            # Evict oldest if over limit
            while len(self._cache) > self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class SmartDictionary:
    """
    Smart Dictionary - Intelligent ML-based translation dictionary.
    
    Features:
    - Machine learning quality scoring
    - Context-aware matching
    - Fuzzy search with 4 algorithms
    - Automatic learning from AI translations
    - Multi-variant support
    - Temporal decay
    
    This is the main dictionary system used by OptikR.
    
    Reads from compressed JSON dictionary files in the format:
    {
        "source_text": {
            "translation": "target_text",
            "usage_count": 15,
            "confidence": 0.95,
            "last_used": "<ISO 8601 timestamp>"
        }
    }
    """
    
    def __init__(self, dictionary_path: str | None = None, cache_size: int = None, config_manager=None):
        """
        Initialize local dictionary.
        
        Args:
            dictionary_path: Path to dictionary file (default: auto-detect)
            cache_size: Size of lookup cache
        """
        self.logger = logging.getLogger(__name__)
        
        # Get cache size from config if not provided
        if cache_size is None and config_manager:
            cache_size = config_manager.get_setting('cache.dictionary_cache_size', 1000)
        elif cache_size is None:
            cache_size = 1000
        
        self.cache = DictionaryLookupCache(cache_size)
        self._lock = RLock()
        
        # Statistics
        self.total_lookups = 0
        self.cache_hits = 0
        
        # Dictionary data
        self._dictionaries: dict[tuple[str, str], dict[str, dict]] = {}
        
        # Track which file path is loaded for each language pair
        self._dictionary_paths: dict[tuple[str, str], str] = {}
        
        # Load dictionary if path provided
        if dictionary_path:
            self.load_dictionary(dictionary_path)
        else:
            # Auto-load all dictionaries from dictionary folder
            self._auto_load_dictionaries()
    
    def _auto_load_dictionaries(self):
        """Auto-load all dictionary files from dictionary folder."""
        dict_dir = get_dictionary_dir()
        if not dict_dir.exists():
            self.logger.info("No dictionary folder found")
            return
        
        # Find all dictionary files
        for dict_file in dict_dir.glob("*.json.gz"):
            try:
                # Parse filename: en_de.json.gz
                filename = dict_file.stem  # Remove .gz
                if filename.endswith('.json'):
                    filename = filename[:-5]  # Remove .json
                
                parts = filename.split('_')
                if len(parts) == 2 and len(parts[0]) <= 3 and len(parts[1]) <= 3:
                    source_lang = parts[0]
                    target_lang = parts[1]
                    self.logger.info(f"Loading dictionary: {source_lang} → {target_lang}")
                    self.load_dictionary(str(dict_file), source_lang, target_lang)
                else:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Failed to load dictionary {dict_file}: {e}")
    
    def load_dictionary(self, dictionary_path: str, source_lang: str = "en", target_lang: str = "de"):
        """
        Load dictionary from file or directory.
        
        Args:
            dictionary_path: Path to dictionary file or directory containing dictionary files
            source_lang: Source language code (used only if loading a specific file)
            target_lang: Target language code (used only if loading a specific file)
        """
        try:
            dict_path = Path(dictionary_path)
            
            if not dict_path.exists():
                self.logger.warning(f"Dictionary path not found: {dictionary_path}")
                return
            
            if dict_path.is_dir():
                self.logger.info(f"Loading dictionaries from directory: {dictionary_path}")
                dict_files = list(dict_path.glob("*.json.gz"))
                
                if not dict_files:
                    self.logger.info(f"No dictionary files found in directory: {dictionary_path}")
                    return
                
                # Read all files first (I/O outside lock)
                loaded_pairs = []
                for dict_file in dict_files:
                    try:
                        filename = dict_file.stem  # Remove .gz
                        if filename.endswith('.json'):
                            filename = filename[:-5]  # Remove .json
                        
                        parts = filename.split('_')
                        if len(parts) == 2 and len(parts[0]) <= 3 and len(parts[1]) <= 3:
                            src_lang = parts[0]
                            tgt_lang = parts[1]
                            
                            with gzip.open(dict_file, 'rt', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if isinstance(data, dict) and 'translations' in data:
                                dictionary_data = data['translations']
                            else:
                                dictionary_data = data
                            
                            loaded_pairs.append(((src_lang, tgt_lang), dictionary_data, str(dict_file)))
                            self.logger.info(f"Loaded dictionary {src_lang}→{tgt_lang}: {len(dictionary_data)} entries from {dict_file}")
                        else:
                            continue
                    except Exception as e:
                        self.logger.error(f"Failed to load dictionary file {dict_file}: {e}")
                
                # Update shared state under lock
                with self._lock:
                    for lang_pair, dictionary_data, file_path in loaded_pairs:
                        self._dictionaries[lang_pair] = dictionary_data
                        self._dictionary_paths[lang_pair] = file_path
                
                self.cache.clear()
                return
            
            # Single file: read outside lock
            with gzip.open(dict_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'translations' in data:
                dictionary_data = data['translations']
                if 'source_language' in data and 'target_language' in data:
                    source_lang = data['source_language']
                    target_lang = data['target_language']
            else:
                dictionary_data = data
            
            lang_pair = (source_lang, target_lang)
            
            # Update shared state under lock
            with self._lock:
                self._dictionaries[lang_pair] = dictionary_data
                self._dictionary_paths[lang_pair] = str(dict_path)
            
            self.logger.info(f"Loaded dictionary {source_lang}→{target_lang}: {len(dictionary_data)} entries from {dict_path}")
            self.cache.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to load dictionary: {e}")
            import traceback
            traceback.print_exc()
    
    def lookup(self, text: str, source_language: str = "en", target_language: str = "de") -> DictionaryEntry | None:
        """
        Look up translation in dictionary.
        
        Args:
            text: Source text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            DictionaryEntry if found, None otherwise
        """
        with self._lock:
            self.total_lookups += 1
            
            # Create cache key
            cache_key = f"{source_language}:{target_language}:{text}"
            
            # Check cache first
            cached = self.cache.get(cache_key)
            if cached is _CACHE_MISS:
                # Negative cache hit — we already know this key isn't in the dictionary
                return None
            if cached is not None:
                self.cache_hits += 1
                return cached
            
            # Look up in dictionary
            lang_pair = (source_language, target_language)
            if lang_pair not in self._dictionaries:
                # Cache negative result
                self.cache.put(cache_key, _CACHE_MISS)
                return None
            
            dictionary = self._dictionaries[lang_pair]
            
            # Try exact match with text
            text_lower = text.lower()
            
            # Try different key formats
            possible_keys = [
                text,  # Exact match
                text_lower,  # Lowercase
                f"{source_language}:{target_language}:{text_lower}",  # Full key format
            ]
            
            for key in possible_keys:
                if key in dictionary:
                    entry_data = dictionary[key]
                    # Dictionary format with metadata
                    entry = DictionaryEntry(
                        source_text=entry_data.get('original', text),
                        translation=entry_data.get('translation', text),
                        source_language=source_language,
                        target_language=target_language,
                        usage_count=entry_data.get('usage_count', 1),
                        confidence=entry_data.get('confidence', 0.9),
                        last_used=entry_data.get('last_used', datetime.now().isoformat()),
                        source_engine=entry_data.get('engine', 'dictionary')
                        )
                    self.cache.put(cache_key, entry)
                    return entry
            
            # Not found — cache negative result
            self.cache.put(cache_key, _CACHE_MISS)
            return None
    
    def fuzzy_lookup(self, text: str, source_language: str = "en", target_language: str = "de", 
                    threshold: float = 0.8, context: str | None = None) -> list[tuple[DictionaryEntry, float]]:
        """
        Advanced fuzzy lookup with multiple similarity algorithms and context awareness.
        
        Uses:
        - Levenshtein distance (edit distance)
        - Sequence matching (difflib)
        - Token-based similarity
        - Context matching
        - Quality-weighted scoring
        
        Args:
            text: Source text
            source_language: Source language code
            target_language: Target language code
            threshold: Minimum similarity threshold (0.0-1.0)
            context: Optional context for better matching
            
        Returns:
            List of (DictionaryEntry, similarity_score) tuples sorted by relevance
        """
        with self._lock:
            lang_pair = (source_language, target_language)
            if lang_pair not in self._dictionaries:
                return []
            
            dictionary = dict(self._dictionaries[lang_pair])
        matches = []
        
        text_lower = text.lower().strip()
        text_tokens = set(re.findall(r'\w+', text_lower))
        context_tokens = set(re.findall(r'\w+', context.lower())) if context else set()
        
        for dict_key, entry_data in dictionary.items():
            # Extract source text from key (format: "en:de:hello" or just "hello")
            if ':' in dict_key:
                parts = dict_key.split(':', 2)
                source_text = parts[2] if len(parts) > 2 else dict_key
            else:
                source_text = dict_key
            
            source_lower = source_text.lower().strip()
            
            # Skip if too different in length
            len_ratio = min(len(text_lower), len(source_lower)) / max(len(text_lower), len(source_lower))
            if len_ratio < 0.3:
                continue
            
            # Calculate multiple similarity scores
            scores = []
            
            # 1. Sequence matching (difflib) - best for typos
            seq_similarity = difflib.SequenceMatcher(None, text_lower, source_lower).ratio()
            scores.append(('sequence', seq_similarity, 0.4))  # 40% weight
            
            # 2. Token-based similarity - good for word order changes
            source_tokens = set(re.findall(r'\w+', source_lower))
            if text_tokens and source_tokens:
                token_similarity = len(text_tokens & source_tokens) / len(text_tokens | source_tokens)
                scores.append(('token', token_similarity, 0.3))  # 30% weight
            
            # 3. Substring matching - good for partial matches
            if text_lower in source_lower or source_lower in text_lower:
                substring_score = min(len(text_lower), len(source_lower)) / max(len(text_lower), len(source_lower))
                scores.append(('substring', substring_score, 0.2))  # 20% weight
            
            # Create entry from data
            if isinstance(entry_data, str):
                # Old format: direct translation string
                entry = DictionaryEntry(
                    source_text=source_text,
                    translation=entry_data,
                    source_language=source_language,
                    target_language=target_language,
                    usage_count=1,
                    confidence=0.9,
                    last_used=datetime.now().isoformat()
                )
            else:
                # New format: dictionary with metadata
                entry = DictionaryEntry(
                    source_text=entry_data.get('original', source_text),
                    translation=entry_data.get('translation', source_text),
                    source_language=source_language,
                    target_language=target_language,
                    usage_count=entry_data.get('usage_count', 1),
                    confidence=entry_data.get('confidence', 0.9),
                    last_used=entry_data.get('last_used', datetime.now().isoformat()),
                    source_engine=entry_data.get('engine', 'dictionary')
                )
            
            # 4. Context matching - bonus for context relevance
            if context_tokens and entry.context_tags:
                context_overlap = len(context_tokens & entry.context_tags) / len(context_tokens | entry.context_tags)
                scores.append(('context', context_overlap, 0.1))  # 10% weight
            
            # Calculate weighted similarity
            weighted_similarity = sum(score * weight for _, score, weight in scores)
            
            # Boost by quality and effective confidence
            quality_boost = 1.0 + (entry.quality_score * 0.2)  # Up to 20% boost
            confidence_boost = 1.0 + (entry.get_effective_confidence() * 0.1)  # Up to 10% boost
            
            final_score = weighted_similarity * quality_boost * confidence_boost
            final_score = min(1.0, final_score)  # Cap at 1.0
            
            if final_score >= threshold:
                matches.append((entry, final_score))
        
        # Sort by score (highest first), then by usage count
        matches.sort(key=lambda x: (x[1], x[0].usage_count), reverse=True)
        
        return matches
    
    @staticmethod
    def _contains_placeholder(text: str) -> bool:
        """Return True if *text* contains a Context Manager placeholder."""
        return bool(_PLACEHOLDER_RE.search(text))

    def add_entry(self, source_text: str, translation: str, source_language: str = "en", 
                 target_language: str = "de", confidence: float = 1.0, context: str | None = None,
                 source_engine: str = "manual", auto_merge: bool = True):
        """
        Intelligently add or update dictionary entry with automatic learning.
        
        Features:
        - Automatic variant detection
        - Smart merging of similar entries
        - Context extraction
        - Quality scoring
        - Duplicate detection
        
        Args:
            source_text: Source text
            translation: Translation
            source_language: Source language code
            target_language: Target language code
            confidence: Confidence score (0.0-1.0)
            context: Optional context for better learning
            source_engine: Origin of translation (manual, ai, learned)
            auto_merge: Automatically merge with similar entries
        """
        if self._contains_placeholder(source_text) or self._contains_placeholder(translation):
            self.logger.debug(
                "Rejected entry containing context placeholder: '%s' → '%s'",
                source_text[:40], translation[:40],
            )
            return

        with self._lock:
            lang_pair = (source_language, target_language)
            
            # Create dictionary if doesn't exist
            if lang_pair not in self._dictionaries:
                self._dictionaries[lang_pair] = {}
            
            dictionary = self._dictionaries[lang_pair]
            
            # Normalize text
            source_normalized = source_text.strip()
            translation_normalized = translation.strip()
            
            # Check for exact match
            if source_normalized in dictionary:
                # Update existing entry
                entry_data = dictionary[source_normalized]
                entry = DictionaryEntry.from_dict(source_normalized, entry_data, source_language, target_language)
                
                # Check if translation is different (variant)
                if translation_normalized != entry.translation:
                    entry.add_variant(translation_normalized)
                    
                    # If new translation has higher confidence, make it primary
                    if confidence > entry.confidence:
                        entry.variants.append(entry.translation)
                        entry.translation = translation_normalized
                        entry.confidence = confidence
                
                # Update usage and context
                entry.update_usage(success=True)
                if context:
                    entry.add_context(context)
                
                # Update quality score based on consistency
                entry.quality_score = min(1.0, entry.quality_score + 0.05)  # Gradual improvement
                
                # Save back
                dictionary[source_normalized] = entry.to_dict()
                
            elif auto_merge:
                # Check for similar entries (fuzzy match)
                similar = self.fuzzy_lookup(source_normalized, source_language, target_language, threshold=0.9, context=context)
                
                if similar:
                    # Merge with most similar entry
                    best_match, similarity = similar[0]
                    
                    self.logger.info(f"Merging '{source_normalized}' with similar entry '{best_match.source_text}' (similarity: {similarity:.2f})")
                    
                    # Add as variant to existing entry
                    entry_data = dictionary[best_match.source_text]
                    entry = DictionaryEntry.from_dict(best_match.source_text, entry_data, source_language, target_language)
                    entry.add_variant(translation_normalized)
                    entry.update_usage(success=True)
                    if context:
                        entry.add_context(context)
                    
                    dictionary[best_match.source_text] = entry.to_dict()
                else:
                    # Create new entry
                    self._create_new_entry(dictionary, source_normalized, translation_normalized, 
                                          source_language, target_language, confidence, context, source_engine)
            else:
                # Create new entry without merging
                self._create_new_entry(dictionary, source_normalized, translation_normalized,
                                      source_language, target_language, confidence, context, source_engine)
            
            # Clear cache
            self.cache.clear()
    
    def _create_new_entry(self, dictionary: dict, source_text: str, translation: str,
                         source_lang: str, target_lang: str, confidence: float,
                         context: str | None, source_engine: str):
        """Create a new dictionary entry with full metadata."""
        entry = DictionaryEntry(
            source_text=source_text,
            translation=translation,
            source_language=source_lang,
            target_language=target_lang,
            usage_count=1,
            confidence=confidence,
            last_used=datetime.now().isoformat(),
            source_engine=source_engine,
            quality_score=confidence * 0.8,  # Initial quality based on confidence
            creation_date=datetime.now().isoformat()
        )
        
        if context:
            entry.add_context(context)
        
        dictionary[source_text] = entry.to_dict()
    
    def learn_from_translation(self, source_text: str, translation: str, 
                              source_language: str, target_language: str,
                              confidence: float, context: str | None = None):
        """
        Learn from AI translation with quality validation.
        
        Only adds high-quality translations to avoid polluting dictionary.
        
        Args:
            source_text: Source text
            translation: AI translation
            source_language: Source language
            target_language: Target language
            confidence: AI confidence score
            context: Optional context
        """
        # Quality threshold for automatic learning
        MIN_CONFIDENCE = 0.85
        MIN_LENGTH = 1  # Minimum word length (1 = allow single words/characters, important for CJK)
        
        # Validate quality
        if confidence < MIN_CONFIDENCE:
            return  # Too low confidence
        
        if len(source_text.split()) < MIN_LENGTH:
            return  # Too short (likely noise)
        
        # Check if translation looks valid (not empty, not same as source)
        if not translation or translation.strip() == source_text.strip():
            return
        
        # Reject entries that still contain context placeholders
        if self._contains_placeholder(source_text) or self._contains_placeholder(translation):
            self.logger.debug("learn_from_translation: rejected placeholder text")
            return
        
        # Add to dictionary with AI source
        self.add_entry(
            source_text=source_text,
            translation=translation,
            source_language=source_language,
            target_language=target_language,
            confidence=confidence,
            context=context,
            source_engine="ai_learned",
            auto_merge=True
        )
        
        self.logger.info(f"Learned: '{source_text}' → '{translation}' (confidence: {confidence:.2f})")
    
    def get_stats(self, source_language: str = "en", target_language: str = "de") -> DictionaryStats:
        """
        Get dictionary statistics.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            DictionaryStats object
        """
        with self._lock:
            lang_pair = (source_language, target_language)
            
            if lang_pair not in self._dictionaries:
                return DictionaryStats()
            
            dictionary = self._dictionaries[lang_pair]
        
        # Calculate stats
        total_entries = len(dictionary)
        total_usage = sum(entry.get('usage_count', 0) for entry in dictionary.values())
        average_usage = total_usage / total_entries if total_entries > 0 else 0.0
        
        # Get most used
        sorted_entries = sorted(
            dictionary.items(),
            key=lambda x: x[1].get('usage_count', 0),
            reverse=True
        )[:10]
        
        most_used = [
            {
                'original': source,
                'translation': data.get('translation', ''),
                'usage_count': data.get('usage_count', 0),
                'confidence': data.get('confidence', 0.0)
            }
            for source, data in sorted_entries
        ]
        
        return DictionaryStats(
            total_entries=total_entries,
            total_usage=total_usage,
            average_usage=average_usage,
            total_lookups=self.total_lookups,
            cache_hits=self.cache_hits,
            most_used=most_used
        )
    
    def get_statistics(self, source_language: str = "en", target_language: str = "de") -> DictionaryStats:
        """
        Alias for get_stats() for backward compatibility.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            DictionaryStats object
        """
        return self.get_stats(source_language, target_language)
    
    def get_available_language_pairs(self) -> list[tuple[str, str, str, int]]:
        """
        Get list of all loaded language-pair dictionaries.
        
        Returns:
            List of tuples: (source_lang, target_lang, file_path, entry_count)
        """
        with self._lock:
            pairs = []
            for (source_lang, target_lang), dictionary in self._dictionaries.items():
                file_path = self._dictionary_paths.get((source_lang, target_lang), "")
                entry_count = len(dictionary)
                pairs.append((source_lang, target_lang, file_path, entry_count))
            
            return pairs
    
    def get_loaded_dictionary_path(self, source_language: str, target_language: str) -> str | None:
        """
        Get the file path of the currently loaded dictionary for a language pair.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            File path string if loaded, None otherwise
        """
        lang_pair = (source_language, target_language)
        return self._dictionary_paths.get(lang_pair)
    
    def get_all_entries(self, source_language: str = "en", target_language: str = "de") -> list[DictionaryEntry]:
        """
        Get all dictionary entries for a language pair.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of DictionaryEntry objects
        """
        with self._lock:
            lang_pair = (source_language, target_language)
            
            if lang_pair not in self._dictionaries:
                return []
            
            dictionary = dict(self._dictionaries[lang_pair])
        
        entries = []
        for source_text, entry_data in dictionary.items():
            try:
                entry = DictionaryEntry.from_dict(source_text, entry_data, source_language, target_language)
                entries.append(entry)
            except Exception as e:
                self.logger.warning(f"Failed to parse entry '{source_text}': {e}")
        
        return entries
    
    def reload_specific_dictionary(self, file_path: str, source_language: str, target_language: str):
        """
        Reload a specific dictionary file for a language pair.
        This allows switching between multiple dictionary files for the same language pair.
        
        Args:
            file_path: Path to the dictionary file to load
            source_language: Source language code
            target_language: Target language code
        """
        try:
            dict_path = Path(file_path)
            
            if not dict_path.exists():
                self.logger.warning(f"Dictionary file not found: {file_path}")
                return
            
            # Read outside lock
            with gzip.open(dict_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'translations' in data:
                dictionary_data = data['translations']
            else:
                dictionary_data = data
            
            # Update state under lock
            lang_pair = (source_language, target_language)
            with self._lock:
                self._dictionaries[lang_pair] = dictionary_data
                self._dictionary_paths[lang_pair] = str(dict_path)
            
            self.cache.clear()
            
            self.logger.info(f"Reloaded dictionary {source_language}→{target_language} from {file_path}: {len(dictionary_data)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to reload dictionary from {file_path}: {e}")
    
    def cleanup_stale_entries(self, source_language: str = "en", target_language: str = "de",
                             min_quality: float = 0.3, max_age_days: int = 365):
        """
        Remove low-quality and stale entries to keep dictionary clean.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            min_quality: Minimum quality score to keep
            max_age_days: Maximum age in days for unused entries
        """
        with self._lock:
            lang_pair = (source_language, target_language)
            if lang_pair not in self._dictionaries:
                return
            
            dictionary = self._dictionaries[lang_pair]
            to_remove = []
            
            for source_text, entry_data in dictionary.items():
                entry = DictionaryEntry.from_dict(source_text, entry_data, source_language, target_language)
                
                # Check quality
                if entry.quality_score < min_quality and entry.usage_count < 3:
                    to_remove.append(source_text)
                    continue
                
                # Check age
                try:
                    last_used = datetime.fromisoformat(entry.last_used)
                    days_old = (datetime.now() - last_used).days
                    
                    if days_old > max_age_days and entry.usage_count < 5:
                        to_remove.append(source_text)
                except (ValueError, TypeError):
                    pass
            
            # Remove stale entries
            for source_text in to_remove:
                del dictionary[source_text]
            
            if to_remove:
                self.logger.info(f"Cleaned up {len(to_remove)} stale entries from {source_language}→{target_language}")
                self.cache.clear()
    
    def get_recommendations(self, source_language: str = "en", target_language: str = "de",
                           limit: int = 10) -> list[dict]:
        """
        Get recommendations for entries that need review or improvement.
        
        Returns entries with:
        - Low quality scores
        - High usage but low confidence
        - Multiple variants (needs consolidation)
        - Stale entries (not used recently)
        
        Args:
            source_language: Source language code
            target_language: Target language code
            limit: Maximum recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        with self._lock:
            lang_pair = (source_language, target_language)
            if lang_pair not in self._dictionaries:
                return []
            
            dictionary = self._dictionaries[lang_pair]
            recommendations = []
            
            for source_text, entry_data in dictionary.items():
                entry = DictionaryEntry.from_dict(source_text, entry_data, source_language, target_language)
                
                # Check for issues
                issues = []
                priority = 0
                
                # Low quality but high usage
                if entry.quality_score < 0.5 and entry.usage_count > 10:
                    issues.append("Low quality despite high usage")
                    priority += 3
                
                # Multiple variants (needs consolidation)
                if len(entry.variants) > 2:
                    issues.append(f"Has {len(entry.variants)} variants - needs review")
                    priority += 2
                
                # Low success rate
                if entry.success_rate < 0.7:
                    issues.append(f"Low success rate: {entry.success_rate:.1%}")
                    priority += 3
                
                # Stale but frequently used
                if entry.decay_factor < 0.7 and entry.usage_count > 20:
                    issues.append("Frequently used but stale - needs refresh")
                    priority += 2
                
                if issues:
                    recommendations.append({
                        'source_text': source_text,
                        'translation': entry.translation,
                        'issues': issues,
                        'priority': priority,
                        'usage_count': entry.usage_count,
                        'quality_score': entry.quality_score,
                        'variants': entry.variants
                    })
        
        # Sort by priority (highest first)
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:limit]
    
    def clear_all_entries(self):
        """Clear all dictionaries and the lookup cache."""
        with self._lock:
            self._dictionaries.clear()
            self._dictionary_paths.clear()
            self.cache.clear()
            self.logger.info("All dictionary entries cleared")

    def discard_unsaved_changes(self) -> None:
        """Revert in-memory dictionaries to the last saved state on disk.

        Clears all in-memory data and reloads from the dictionary files,
        effectively discarding any entries learned during the current session.
        """
        with self._lock:
            self._dictionaries.clear()
            self._dictionary_paths.clear()
            self.cache.clear()
        self._auto_load_dictionaries()
        self.logger.info("Discarded unsaved dictionary changes — reloaded from disk")
    
    def save_dictionary(self, dictionary_path: str, source_language: str = "en", target_language: str = "de",
                       auto_cleanup: bool = True):
        """
        Save dictionary to file with optional automatic cleanup.
        
        Uses atomic write (temp file + rename) to prevent corruption on crash.
        
        Args:
            dictionary_path: Path to save dictionary
            source_language: Source language code
            target_language: Target language code
            auto_cleanup: Automatically clean up stale entries before saving
        """
        try:
            lang_pair = (source_language, target_language)
            
            if auto_cleanup:
                self.cleanup_stale_entries(source_language, target_language)
            
            # Snapshot dictionary data under lock
            with self._lock:
                if lang_pair not in self._dictionaries:
                    self.logger.warning(f"No dictionary to save for {source_language}→{target_language}")
                    return
                dictionary = dict(self._dictionaries[lang_pair])
            
            dict_path = Path(dictionary_path)
            dict_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if dict_path.exists():
                backup_path = Path(str(dict_path) + '.bak')
                import shutil
                shutil.copy2(dict_path, backup_path)
            
            dict_file_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(dictionary),
                "compressed": True,
                "source_language": source_language,
                "target_language": target_language,
                "translations": dictionary
            }
            
            # Atomic write: write to temp file, then rename into place
            fd, tmp_path = tempfile.mkstemp(
                dir=str(dict_path.parent),
                suffix='.tmp'
            )
            os.close(fd)
            try:
                with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
                    json.dump(dict_file_data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, str(dict_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            
            self.logger.info(f"Saved dictionary {source_language}→{target_language}: {len(dictionary)} entries to {dictionary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dictionary: {e}")
            import traceback
            traceback.print_exc()


def create_smart_dictionary(dictionary_path: str | None = None, cache_size: int = 1000) -> 'SmartDictionary':
    """
    Create a SmartDictionary instance.
    
    Args:
        dictionary_path: Path to dictionary file (optional)
        cache_size: Size of lookup cache
        
    Returns:
        SmartDictionary instance
    """
    return SmartDictionary(dictionary_path, cache_size)
