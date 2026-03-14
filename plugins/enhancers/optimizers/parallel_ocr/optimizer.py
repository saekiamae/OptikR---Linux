"""
Parallel OCR Optimizer Plugin
Processes multiple text regions simultaneously using worker threads
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import threading

logger = logging.getLogger(__name__)


class ParallelOCROptimizer:
    """Processes multiple OCR regions in parallel using worker threads"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.worker_threads = config.get('worker_threads', 4)
        self.batch_size = config.get('batch_size', 8)
        self.timeout = config.get('timeout_seconds', 10.0)
        
        # Query hardware capability gate for GPU OCR feature availability
        plugin_use_gpu = config.get('use_gpu', True)
        try:
            from app.utils.hardware_capability_gate import get_hardware_gate, GatedFeature
            gate = get_hardware_gate()
            gpu_ocr_available = (
                gate.is_available(GatedFeature.EASYOCR)
                or gate.is_available(GatedFeature.PADDLEOCR)
                or gate.is_available(GatedFeature.MANGA_OCR)
            )
            if not gpu_ocr_available:
                self.use_gpu = False
                logger.info("Hardware gate: GPU OCR unavailable - falling back to CPU")
            else:
                self.use_gpu = plugin_use_gpu
        except Exception as e:
            # If gate is unavailable, default to CPU-compatible behavior (Req 5.3)
            self.use_gpu = False
            logger.info("Hardware gate unavailable (%s) - defaulting to CPU", e)
        
        # Thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix="ocr_worker"
        )
        
        # Statistics
        self.total_regions = 0
        self.parallel_operations = 0
        self.total_time_saved = 0.0
        self.lock = threading.Lock()
        
        logger.info("Initialized with %d workers (GPU: %s)", self.worker_threads, "enabled" if self.use_gpu else "disabled")
    
    def _ocr_single_region(self, region_data: dict[str, Any], ocr_func) -> dict[str, Any]:
        """Process OCR for a single region"""
        try:
            start_time = time.time()
            
            # Extract region info
            region_id = region_data.get('id', 0)
            frame = region_data.get('frame')
            bbox = region_data.get('bbox', (0, 0, 0, 0))
            
            # Perform OCR using provided OCR function
            if ocr_func and frame is not None:
                result = ocr_func(frame)
            else:
                result = {'text': '', 'confidence': 0.0}
            
            elapsed = time.time() - start_time
            
            return {
                'region_id': region_id,
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'bbox': bbox,
                'success': True,
                'ocr_time': elapsed
            }
            
        except Exception as e:
            logger.error("Error processing region %s: %s", region_data.get('id'), e)
            return {
                'region_id': region_data.get('id', 0),
                'text': '',
                'confidence': 0.0,
                'bbox': region_data.get('bbox'),
                'success': False,
                'error': str(e)
            }
    
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process: OCR multiple regions in parallel"""
        regions = data.get('text_regions', [])
        ocr_func = data.get('ocr_function')
        
        # If only one region, no need for parallel processing
        if len(regions) <= 1:
            return data
        
        # Limit batch size
        regions_to_process = regions[:self.batch_size]
        
        start_time = time.time()
        
        # Submit all OCR tasks
        futures = []
        for region in regions_to_process:
            future = self.executor.submit(
                self._ocr_single_region,
                region,
                ocr_func
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures, timeout=self.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Future failed: %s", e)
        
        # Calculate time saved
        elapsed = time.time() - start_time
        sequential_time = sum(r.get('ocr_time', 0) for r in results)
        time_saved = sequential_time - elapsed
        
        # Update statistics
        with self.lock:
            self.total_regions += len(regions_to_process)
            self.parallel_operations += 1
            self.total_time_saved += time_saved
        
        # Update data with results
        data['ocr_results'] = results
        data['parallel_ocr_time'] = elapsed
        data['time_saved'] = time_saved
        
        logger.info("Processed %d regions in %.3fs (saved %.3fs, speedup: %.1fx)", len(results), elapsed, time_saved, sequential_time / elapsed)
        
        return data
    
    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics"""
        with self.lock:
            avg_time_saved = (self.total_time_saved / self.parallel_operations 
                            if self.parallel_operations > 0 else 0)
            
            return {
                'total_regions': self.total_regions,
                'parallel_operations': self.parallel_operations,
                'total_time_saved': f"{self.total_time_saved:.2f}s",
                'avg_time_saved_per_operation': f"{avg_time_saved:.3f}s",
                'worker_threads': self.worker_threads,
                'gpu_enabled': self.use_gpu
            }
    
    def reset(self):
        """Reset optimizer state"""
        with self.lock:
            self.total_regions = 0
            self.parallel_operations = 0
            self.total_time_saved = 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Thread pool shut down")


# Plugin interface
def initialize(config: dict[str, Any]) -> ParallelOCROptimizer:
    """Initialize the optimizer plugin"""
    return ParallelOCROptimizer(config)


def shutdown(optimizer: ParallelOCROptimizer):
    """Shutdown the optimizer plugin"""
    if optimizer:
        optimizer.cleanup()
