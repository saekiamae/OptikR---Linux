"""Image processing package for static image translation.

Provides batch translation of static images by reusing the existing
OCR and translation pipeline stages and rendering translated text
directly onto images via Pillow.
"""

from .image_compositor import ImageCompositor
from .image_pipeline import ImagePipeline
from .batch_processor import BatchProcessor
from .presets import ImageProcessingPreset, PresetManager

__all__ = [
    "ImageCompositor",
    "ImagePipeline",
    "BatchProcessor",
    "ImageProcessingPreset",
    "PresetManager",
]
