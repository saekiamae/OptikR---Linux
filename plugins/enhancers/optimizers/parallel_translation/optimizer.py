"""
Parallel Translation Optimizer Plugin
Translates multiple text blocks simultaneously using worker threads

Features:
- Warm start: Pre-loads translation models to avoid threading issues
- Automatic fallback: Falls back to sequential if parallel processing fails
- Thread-safe: Handles shutdown gracefully
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import threading

logger = logging.getLogger(__name__)


class ParallelTranslationOptimizer:
    """Translates multiple text blocks in parallel using worker threads"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.worker_threads = config.get('worker_threads', 4)
        self.batch_size = config.get('batch_size', 16)
        self.timeout = config.get('timeout_seconds', 15.0)
        self.enable_warm_start = config.get('enable_warm_start', True)
        self.fallback_on_error = config.get('fallback_on_error', True)
        
        # Query hardware capability gate for MarianMT GPU feature availability
        plugin_use_gpu = config.get('use_gpu', True)
        try:
            from app.utils.hardware_capability_gate import get_hardware_gate, GatedFeature
            gate = get_hardware_gate()
            if not gate.is_available(GatedFeature.MARIANMT_GPU):
                self.use_gpu = False
                logger.info("Hardware gate: MarianMT GPU unavailable - falling back to CPU")
            else:
                self.use_gpu = plugin_use_gpu
        except Exception as e:
            self.use_gpu = False
            logger.info("Hardware gate unavailable (%s) - defaulting to CPU", e)
        
        # Thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix="translation_worker"
        )
        self.is_shutdown = False
        self.warm_started = False
        self.fallback_mode = False
        
        # Statistics
        self.total_texts = 0
        self.parallel_operations = 0
        self.total_time_saved = 0.0
        self.fallback_count = 0
        self.warm_start_attempts = 0
        self.lock = threading.Lock()
        
        logger.info("Initialized with %d workers (GPU: %s, warm_start: %s, fallback: %s)",
                    self.worker_threads,
                    'enabled' if self.use_gpu else 'disabled',
                    'enabled' if self.enable_warm_start else 'disabled',
                    'enabled' if self.fallback_on_error else 'disabled')
    
    def _translate_single_text(self, text_data: dict[str, Any], translate_func) -> dict[str, Any]:
        """Translate a single text block"""
        try:
            start_time = time.time()
            
            # Extract text info
            text_id = text_data.get('id', 0)
            source_text = text_data.get('text', '')
            source_lang = text_data.get('source_lang', 'auto')
            target_lang = text_data.get('target_lang', 'en')
            
            # Skip empty text
            if not source_text or not source_text.strip():
                return {
                    'text_id': text_id,
                    'source_text': source_text,
                    'translated_text': '',
                    'success': True,
                    'translation_time': 0.0,
                    'skipped': True
                }
            
            # Perform translation using provided translation function
            if translate_func:
                result = translate_func(source_text, source_lang, target_lang)
            else:
                result = {'translated_text': source_text, 'confidence': 0.0}
            
            elapsed = time.time() - start_time
            
            return {
                'text_id': text_id,
                'source_text': source_text,
                'translated_text': result.get('translated_text', source_text),
                'confidence': result.get('confidence', 1.0),
                'success': True,
                'translation_time': elapsed
            }
            
        except Exception as e:
            logger.warning("Error translating text %s: %s", text_data.get('id'), e)
            return {
                'text_id': text_data.get('id', 0),
                'source_text': text_data.get('text', ''),
                'translated_text': text_data.get('text', ''),  # Fallback to source
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def warm_start(self, source_lang: str, target_lang: str, translate_func) -> bool:
        """
        Warm start: Pre-load translation models in worker threads.
        This helps avoid threading issues with model loading.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            translate_func: Translation function to warm up
            
        Returns:
            True if warm start successful, False otherwise
        """
        if not self.enable_warm_start or self.warm_started:
            return True
        
        with self.lock:
            self.warm_start_attempts += 1
        
        try:
            logger.info("Warm starting %d workers...", self.worker_threads)
            
            # Submit dummy translation to each worker to pre-load models
            dummy_text = "test"
            futures = []
            
            for i in range(self.worker_threads):
                try:
                    future = self.executor.submit(
                        self._warm_start_worker,
                        dummy_text,
                        source_lang,
                        target_lang,
                        translate_func,
                        i
                    )
                    futures.append(future)
                except Exception as e:
                    logger.warning("Failed to submit warm start task %d: %s", i, e)
                    return False
            
            # Wait for all workers to complete warm start
            success_count = 0
            # Get timeout from config
            model_timeout = 30.0
            if hasattr(self, 'config_manager') and self.config_manager:
                model_timeout = self.config_manager.get_setting('timeouts.model_loading', 30.0)
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=model_timeout)
                    if result:
                        success_count += 1
                except Exception as e:
                    logger.warning("Worker %d warm start failed: %s", i, e)
            
            if success_count == self.worker_threads:
                self.warm_started = True
                logger.info("Warm start complete (%d/%d workers ready)", success_count, self.worker_threads)
                return True
            else:
                logger.warning("Partial warm start (%d/%d workers ready)", success_count, self.worker_threads)
                # Still mark as warm started if at least one worker succeeded
                if success_count > 0:
                    self.warm_started = True
                    return True
                return False
                
        except Exception as e:
            logger.error("Warm start failed: %s", e, exc_info=True)
            return False
    
    def _warm_start_worker(self, text: str, source_lang: str, target_lang: str, 
                          translate_func, worker_id: int) -> bool:
        """Warm start a single worker by performing a dummy translation."""
        try:
            logger.debug("Worker %d warming up...", worker_id)
            result = translate_func(text, source_lang, target_lang)
            if result and result.get('translated_text'):
                logger.debug("Worker %d ready", worker_id)
                return True
            return False
        except Exception as e:
            logger.warning("Worker %d warm start error: %s", worker_id, e)
            return False
    
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process: Translate multiple text blocks in parallel with fallback"""
        # Check if shutdown
        if self.is_shutdown:
            return data
        
        # Check if in fallback mode (skip parallel processing)
        if self.fallback_mode:
            return data
        
        texts = data.get('texts', [])
        translate_func = data.get('translation_function')
        
        # If only one text, no need for parallel processing
        if len(texts) <= 1:
            return data
        
        # Attempt warm start on first call (if not already done in background)
        if self.enable_warm_start and not self.warm_started:
            logger.info("Warm start not complete yet, performing inline warm start...")
            source_lang = texts[0].get('source_lang', 'auto') if texts else 'auto'
            target_lang = texts[0].get('target_lang', 'en') if texts else 'en'
            
            warm_start_success = self.warm_start(source_lang, target_lang, translate_func)
            
            if not warm_start_success and self.fallback_on_error:
                logger.warning("Warm start failed, enabling fallback mode")
                self.fallback_mode = True
                with self.lock:
                    self.fallback_count += 1
                return data
        
        # Limit batch size
        texts_to_process = texts[:self.batch_size]
        
        start_time = time.time()
        
        try:
            # Submit all translation tasks
            futures = []
            for text_item in texts_to_process:
                try:
                    future = self.executor.submit(
                        self._translate_single_text,
                        text_item,
                        translate_func
                    )
                    futures.append(future)
                except RuntimeError as e:
                    # Executor is shutdown
                    logger.warning("Executor shutdown, enabling fallback mode")
                    self.is_shutdown = True
                    if self.fallback_on_error:
                        self.fallback_mode = True
                        with self.lock:
                            self.fallback_count += 1
                    return data
            
            # Collect results with improved timeout handling
            results = []
            completed_count = 0
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                    except Exception as e:
                        logger.warning("Future failed: %s", e)
            except TimeoutError as e:
                # Collect partial results from completed futures
                logger.warning("Timeout after %d/%d completed", completed_count, len(futures))
                for future in futures:
                    if future.done() and not future.cancelled():
                        try:
                            result = future.result(timeout=0)
                            if result not in results:
                                results.append(result)
                        except Exception as e:
                            logger.debug("Skipping failed future during timeout recovery: %s", e)
                
                # If we got some results, continue with partial success
                if results:
                    logger.info("Continuing with %d partial results", len(results))
                elif self.fallback_on_error:
                    logger.warning("No results, enabling fallback mode")
                    self.fallback_mode = True
                    with self.lock:
                        self.fallback_count += 1
                    return data
            
            # Verify we got results
            if not results:
                logger.warning("No results obtained")
                if self.fallback_on_error:
                    logger.warning("Enabling fallback mode")
                    self.fallback_mode = True
                    with self.lock:
                        self.fallback_count += 1
                return data
            
            # Calculate time saved
            elapsed = time.time() - start_time
            sequential_time = sum(r.get('translation_time', 0) for r in results 
                                if not r.get('skipped', False))
            time_saved = max(0, sequential_time - elapsed)
            
            # Update statistics
            with self.lock:
                self.total_texts += len(texts_to_process)
                self.parallel_operations += 1
                self.total_time_saved += time_saved
            
            # Update data with results
            data['translation_results'] = results
            data['parallel_translation_time'] = elapsed
            data['time_saved'] = time_saved
            
            speedup = sequential_time / elapsed if elapsed > 0 else 1.0
            success_rate = f"{len(results)}/{len(texts_to_process)}"
            logger.info("Translated %s texts in %.3fs (saved %.3fs, speedup: %.1fx)",
                        success_rate, elapsed, time_saved, speedup)
            
            return data
            
        except Exception as e:
            logger.error("Error during parallel processing: %s", e, exc_info=True)
            
            if self.fallback_on_error:
                logger.warning("Enabling fallback mode due to exception")
                self.fallback_mode = True
                with self.lock:
                    self.fallback_count += 1
            
            return data
    
    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics"""
        with self.lock:
            avg_time_saved = (self.total_time_saved / self.parallel_operations 
                            if self.parallel_operations > 0 else 0)
            
            return {
                'total_texts': self.total_texts,
                'parallel_operations': self.parallel_operations,
                'total_time_saved': f"{self.total_time_saved:.2f}s",
                'avg_time_saved_per_operation': f"{avg_time_saved:.3f}s",
                'worker_threads': self.worker_threads,
                'gpu_enabled': self.use_gpu,
                'warm_started': self.warm_started,
                'warm_start_attempts': self.warm_start_attempts,
                'fallback_mode': self.fallback_mode,
                'fallback_count': self.fallback_count
            }
    
    def reset(self):
        """Reset optimizer state"""
        with self.lock:
            self.total_texts = 0
            self.parallel_operations = 0
            self.total_time_saved = 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor and not self.is_shutdown:
            self.is_shutdown = True
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Thread pool shut down")


# Plugin interface
def initialize(config: dict[str, Any]) -> ParallelTranslationOptimizer:
    """Initialize the optimizer plugin"""
    return ParallelTranslationOptimizer(config)


def shutdown(optimizer: ParallelTranslationOptimizer):
    """Shutdown the optimizer plugin"""
    if optimizer:
        optimizer.cleanup()
