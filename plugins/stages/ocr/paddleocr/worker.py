"""
Paddleocr - OCR engine using paddleocr library

OCR plugin worker script.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker


class OCRWorker(BaseWorker):
    """Worker for Paddleocr."""
    
    def initialize(self, config: dict) -> bool:
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration dictionary with 'language' key
            
        Returns:
            True if successful
        """
        try:
            from paddleocr import PaddleOCR
            import os
            
            # Suppress PaddleOCR verbose logging
            os.environ['GLOG_minloglevel'] = '3'
            os.environ['FLAGS_pir_apply_shape_optimization_pass'] = '0'
            
            from app.utils.language_mapper import LanguageCodeMapper
            
            self.language = config.get('language', 'en')
            self.use_gpu = config.get('gpu', True)
            
            paddle_lang = LanguageCodeMapper.to_paddleocr(self.language)
            
            self.log(f"Initializing PaddleOCR (lang={paddle_lang}, gpu={self.use_gpu})...")
            
            # Initialize PaddleOCR
            try:
                # Try with show_log parameter (newer versions)
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except TypeError:
                # Fallback without show_log parameter (older versions)
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=self.use_gpu
                )
            
            self.log("PaddleOCR initialized successfully")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"Failed to initialize PaddleOCR: {e}\n{traceback.format_exc()}")
            return False
    
    def process(self, data: dict) -> dict:
        """
        Perform OCR on frame.
        
        Args:
            data: {
                'frame': base64_string,
                'shape': [h, w, c],
                'dtype': 'uint8',
                'language': str
            }
            
        Returns:
            {
                'text_blocks': [
                    {'text': str, 'bbox': [x, y, w, h], 'confidence': float}
                ],
                'count': int
            }
        """
        try:
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb
            
            frame, error = decode_frame(data)
            if error:
                return {'error': error}
            
            frame = bgr_to_rgb(frame)
            
            # Perform OCR
            results = self.ocr.ocr(frame, cls=True)
            
            # Parse results
            text_blocks = []
            
            if results and results[0]:
                for line in results[0]:
                    # Each line: [bbox_points, (text, confidence)]
                    bbox_points, (text, confidence) = line
                    
                    # Convert bbox points to [x, y, w, h]
                    # bbox_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    
                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)
                    
                    text_blocks.append({
                        'text': text,
                        'bbox': [x, y, w, h],
                        'confidence': float(confidence)
                    })
            
            return {
                'text_blocks': text_blocks,
                'count': len(text_blocks)
            }
            
        except Exception as e:
            import traceback
            error_msg = f'OCR failed: {e}\n{traceback.format_exc()}'
            self.log(error_msg)
            return {'error': error_msg}
    
    def cleanup(self):
        """Clean up resources."""
        self.ocr = None
        self.log("PaddleOCR cleanup complete")


if __name__ == '__main__':
    worker = OCRWorker(name="paddleocr")
    worker.run()
