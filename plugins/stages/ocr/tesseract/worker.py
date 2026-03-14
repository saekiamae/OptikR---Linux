"""
Tesseract - OCR engine using tesseract library

OCR plugin worker script.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker


class OCRWorker(BaseWorker):
    """Worker for Tesseract."""
    
    def initialize(self, config: dict) -> bool:
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration dictionary with 'language' key
            
        Returns:
            True if successful
        """
        try:
            import pytesseract
            
            from app.utils.language_mapper import LanguageCodeMapper
            
            self.language = config.get('language', 'eng')
            self.tesseract_lang = LanguageCodeMapper.to_tesseract(self.language)
            
            # Test if Tesseract is installed
            try:
                pytesseract.get_tesseract_version()
                self.log(f"Tesseract initialized for language: {self.tesseract_lang}")
                return True
            except Exception as e:
                self.log(f"Tesseract not found. Please install Tesseract OCR: {e}")
                return False
            
        except Exception as e:
            self.log(f"Failed to initialize: {e}")
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
            import pytesseract
            from PIL import Image
            from app.workflow.base.ocr_frame_utils import decode_frame, bgr_to_rgb
            
            frame, error = decode_frame(data)
            if error:
                self.log(f"ERROR: {error}")
                return {'error': error}
            
            self.log(f"Frame decoded: {frame.shape}, min={frame.min()}, max={frame.max()}")
            
            image = Image.fromarray(bgr_to_rgb(frame))
            
            # Perform OCR with detailed data
            self.log(f"Starting Tesseract OCR (lang={self.tesseract_lang})...")
            try:
                ocr_data = pytesseract.image_to_data(
                    image,
                    lang=self.tesseract_lang,
                    output_type=pytesseract.Output.DICT
                )
                self.log(f"Tesseract OCR complete: {len(ocr_data['text'])} boxes detected")
            except Exception as ocr_error:
                self.log(f"Tesseract error: {type(ocr_error).__name__}: {ocr_error}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                # Return empty results instead of failing
                return {
                    'text_blocks': [],
                    'count': 0
                }
            
            # Parse results into text blocks
            text_blocks = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                # Skip empty text or low confidence
                if not text or conf < 0:
                    continue
                
                # Get bounding box
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                text_blocks.append({
                    'text': text,
                    'bbox': [x, y, w, h],
                    'confidence': conf / 100.0  # Convert to 0-1 range
                })
            
            self.log(f"Parsed {len(text_blocks)} valid text blocks")
            
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
        self.log("Tesseract cleanup complete")


if __name__ == '__main__':
    worker = OCRWorker(name="tesseract")
    worker.run()
