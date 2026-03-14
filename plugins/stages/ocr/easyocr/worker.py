"""
OCR Worker - Subprocess for text recognition.

Runs in separate process, performs OCR and sends results back.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[OCR WORKER] Warning: easyocr not available", file=sys.stderr)


class OCRWorker(BaseWorker):
    """Worker for OCR using EasyOCR."""
    
    def initialize(self, config: dict) -> bool:
        """Initialize OCR engine."""
        try:
            if not EASYOCR_AVAILABLE:
                self.log("EasyOCR not available")
                return False
            
            language = config.get('language', 'en')
            use_gpu = config.get('gpu', True)
            
            self.log(f"Initializing EasyOCR for language: {language}, GPU: {use_gpu}")
            
            # Initialize EasyOCR reader with GPU setting from config
            self.reader = easyocr.Reader([language], gpu=use_gpu)
            
            self.log(f"EasyOCR initialized successfully (GPU: {use_gpu})")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize OCR: {e}")
            return False
    
    def process(self, data: dict) -> dict:
        """
        Perform OCR on frame.
        
        Args:
            data: {'frame': base64_encoded_frame, 'language': str}
            
        Returns:
            {'text_blocks': [{'text': str, 'bbox': [x,y,w,h], 'confidence': float}], 'count': int}
        """
        try:
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb
            
            frame, error = decode_frame(data)
            if error:
                return {'error': error}
            
            self.log(f"Frame decoded: {frame.shape}, min={frame.min()}, max={frame.max()}")
            
            frame = bgr_to_rgb(frame)
            
            # Perform OCR with detailed error handling
            self.log("Starting OCR...")
            try:
                # Use paragraph=True to group text blocks together (better for manga)
                results = self.reader.readtext(frame, paragraph=True)
                self.log(f"OCR complete: {len(results)} results")
            except Exception as ocr_error:
                self.log(f"EasyOCR error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                # Return empty results instead of failing
                return {
                    'text_blocks': [],
                    'count': 0
                }
            
            # Convert results to text blocks
            text_blocks = []
            for bbox, text, confidence in results:
                # Convert bbox to x,y,w,h format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)
                
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
            return {'error': f'OCR failed: {e}'}
    
    def cleanup(self):
        """Clean up OCR resources."""
        if hasattr(self, 'reader'):
            del self.reader
        self.log("OCR worker shutdown")


if __name__ == '__main__':
    worker = OCRWorker(name="OCRWorker")
    worker.run()
