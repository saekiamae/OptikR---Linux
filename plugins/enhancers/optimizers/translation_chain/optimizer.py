"""
Translation Chain Optimizer Plugin
Chains translations through intermediate languages for better quality.
Uses SmartDictionary for intelligent caching and learning.

Example: Japanese → English → German
Instead of: Japanese → German (direct, poor quality)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TranslationChainOptimizer:
    """
    Chain translations through intermediate languages.
    
    Features:
    - Better quality for rare language pairs
    - Saves all intermediate mappings to dictionary
    - Caches intermediate results
    - Configurable chains
    """
    
    def __init__(self, config: dict[str, Any]):
        """Initialize translation chain optimizer."""
        self.config = config
        self.enable_chaining = config.get('enable_chaining', False)
        self.intermediate_language = config.get('intermediate_language', 'en')
        self.chain_pairs = config.get('chain_pairs', {
            'ja->de': 'ja->en->de',
            'ko->de': 'ko->en->de',
            'zh->ja': 'zh->en->ja'
        })
        self.save_all_mappings = config.get('save_all_mappings', True)
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.cache_intermediate = config.get('cache_intermediate', True)
        
        # Intermediate translation cache
        self.intermediate_cache = {}
        
        # Statistics
        self.total_translations = 0
        self.chained_translations = 0
        self.direct_translations = 0
        self.cache_hits = 0
        
        logger.info("Initialized with %d chain pairs", len(self.chain_pairs))
        if self.enable_chaining:
            logger.info("Chaining enabled, intermediate language: %s", self.intermediate_language)
    
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Pre-process: Determine if translation should be chained.
        
        Args:
            data: Translation data with source_lang, target_lang, text
            
        Returns:
            Modified data with chaining instructions
        """
        if not self.enable_chaining:
            return data
        
        self.total_translations += 1
        
        source_lang = data.get('source_lang', 'ja')
        target_lang = data.get('target_lang', 'de')
        text = data.get('text', '')
        
        # Create language pair key
        pair_key = f"{source_lang}->{target_lang}"
        
        # Check if this pair should use chaining
        if pair_key in self.chain_pairs:
            chain_spec = self.chain_pairs[pair_key]
            
            # Parse chain specification (e.g., "ja->en->de")
            chain_languages = chain_spec.split('->')
            
            if len(chain_languages) > 2:
                # Check if we have the FINAL translation in SmartDictionary (ja->de, not ja->en)
                translation_layer = data.get('translation_layer')
                dict_engine = None
                
                if translation_layer and hasattr(translation_layer, '_engine_registry'):
                    dict_engine = translation_layer._engine_registry.get_engine('dictionary')
                
                # Check SmartDictionary for final translation
                if dict_engine and hasattr(dict_engine, '_dictionary'):
                    dict_entry = dict_engine._dictionary.lookup(text, source_lang, target_lang)
                    if dict_entry:
                        # Use direct translation from SmartDictionary
                        data['translated_text'] = dict_entry.translation
                        data['skip_translation'] = True
                        data['translation_source'] = 'smart_dictionary'
                        self.direct_translations += 1
                        logger.debug("Found in SmartDictionary: %s", pair_key)
                        return data
                
                # Enable chaining
                data['use_translation_chain'] = True
                data['chain_languages'] = chain_languages
                data['chain_spec'] = chain_spec
                self.chained_translations += 1
                
                logger.debug("Using chain: %s for '%.30s...'", chain_spec, text)
            else:
                # Direct translation
                data['use_translation_chain'] = False
                self.direct_translations += 1
        else:
            # Direct translation (no chain defined)
            data['use_translation_chain'] = False
            self.direct_translations += 1
        
        return data
    
    def post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Post-process: Execute chained translation if needed.
        Saves all intermediate and final mappings to learning dictionary.
        
        Args:
            data: Translation data with results
            
        Returns:
            Data with final translation and all mappings saved
        """
        # Skip if not using chain
        if not data.get('use_translation_chain', False):
            return data
        
        # Skip if already translated (dictionary hit)
        if data.get('skip_translation', False):
            return data
        
        # Get required components
        translation_layer = data.get('translation_layer')
        
        if not translation_layer:
            logger.warning("Translation layer not available")
            data['use_translation_chain'] = False
            return data
        
        # Get SmartDictionary engine
        dict_engine = None
        if hasattr(translation_layer, '_engine_registry'):
            dict_engine = translation_layer._engine_registry.get_engine('dictionary')
        
        if not dict_engine or not hasattr(dict_engine, '_dictionary'):
            logger.warning("SmartDictionary not available")
            # Continue without dictionary support
        
        try:
            chain_languages = data.get('chain_languages', [])
            original_text = data.get('text', '')
            
            if not original_text or len(chain_languages) < 2:
                return data
            
            # Execute translation chain
            intermediate_texts = [original_text]
            intermediate_translations = []
            
            logger.debug("Executing chain: %s", " -> ".join(chain_languages))
            
            for i in range(len(chain_languages) - 1):
                source = chain_languages[i]
                target = chain_languages[i + 1]
                current_text = intermediate_texts[-1]
                
                # Check intermediate cache first
                cache_key = f"{source}:{target}:{current_text}"
                if self.cache_intermediate and cache_key in self.intermediate_cache:
                    translated = self.intermediate_cache[cache_key]
                    self.cache_hits += 1
                    logger.debug("Step %d (cached): %s -> %s", i + 1, source, target)
                else:
                    # Check SmartDictionary for this step
                    dict_translation = None
                    if dict_engine and hasattr(dict_engine, '_dictionary'):
                        dict_entry = dict_engine._dictionary.lookup(current_text, source, target)
                        if dict_entry:
                            dict_translation = dict_entry.translation
                    
                    if dict_translation:
                        translated = dict_translation
                        logger.debug("Step %d (SmartDictionary): %s -> %s", i + 1, source, target)
                    else:
                        # Translate this step using engine
                        logger.debug("Step %d (engine): %s -> %s", i + 1, source, target)
                        translated = translation_layer.translate(current_text, source, target)
                    
                    # Cache intermediate result
                    if self.cache_intermediate and translated:
                        self.intermediate_cache[cache_key] = translated
                
                if not translated:
                    logger.error("Translation failed at step %d", i + 1)
                    # Fall back to direct translation
                    data['use_translation_chain'] = False
                    return data
                
                intermediate_texts.append(translated)
                intermediate_translations.append({
                    'step': i + 1,
                    'source': source,
                    'target': target,
                    'text': translated
                })
                
                logger.debug("  '%.30s...' -> '%.30s...'", current_text, translated)
            
            # Get final translation
            final_translation = intermediate_texts[-1]
            
            # Save mappings to SmartDictionary
            if dict_engine and hasattr(dict_engine, '_dictionary'):
                num_mappings = (len(intermediate_translations) + 1) if self.save_all_mappings else 1
                logger.debug("Saving %d mapping(s) to SmartDictionary...", num_mappings)
                
                # Save intermediate steps (optional)
                if self.save_all_mappings:
                    for i, step_data in enumerate(intermediate_translations):
                        dict_engine._dictionary.add_entry(
                            source_text=intermediate_texts[i],
                            translation=step_data['text'],
                            source_language=step_data['source'],
                            target_language=step_data['target'],
                            confidence=0.9,
                            source_engine='translation_chain'
                        )
                        logger.debug("  Saved intermediate: %s->%s", step_data['source'], step_data['target'])
                
                # ALWAYS save final direct mapping (ja->de, not ja->en!)
                # This is the most important one!
                dict_engine._dictionary.add_entry(
                    source_text=original_text,
                    translation=final_translation,
                    source_language=chain_languages[0],  # ja
                    target_language=chain_languages[-1],  # de (NOT en!)
                    confidence=0.95,
                    source_engine='translation_chain_final'
                )
                logger.debug("  Saved FINAL: %s->%s ('%.20s...' -> '%.20s...')", chain_languages[0], chain_languages[-1], original_text, final_translation)
            else:
                logger.warning("SmartDictionary not available, translations not saved")
            
            # Store final result
            data['translated_text'] = final_translation
            data['translation_method'] = 'chained'
            data['chain_steps'] = intermediate_translations
            data['skip_translation'] = True  # Don't translate again
            
            logger.debug("Chain complete: '%.30s...' -> '%.30s...'", original_text, final_translation)
            
        except Exception as e:
            logger.error("Chain failed: %s", e, exc_info=True)
            # Fall back to direct translation
            data['use_translation_chain'] = False
        
        return data
    
    def get_stats(self) -> dict[str, Any]:
        """Get translation chain statistics."""
        chain_rate = (self.chained_translations / self.total_translations * 100) if self.total_translations > 0 else 0
        
        return {
            'total_translations': self.total_translations,
            'chained_translations': self.chained_translations,
            'direct_translations': self.direct_translations,
            'chain_rate': f"{chain_rate:.1f}%",
            'intermediate_cache_hits': self.cache_hits,
            'cache_size': len(self.intermediate_cache)
        }
    
    def reset(self):
        """Reset statistics and cache."""
        self.total_translations = 0
        self.chained_translations = 0
        self.direct_translations = 0
        self.cache_hits = 0
        self.intermediate_cache.clear()


# Plugin interface
def initialize(config: dict[str, Any]) -> TranslationChainOptimizer:
    """Initialize the translation chain optimizer plugin."""
    return TranslationChainOptimizer(config)
