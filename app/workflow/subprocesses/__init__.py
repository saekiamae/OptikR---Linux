"""
Pipeline Subprocesses Package

Concrete BaseSubprocess subclasses for crash-isolated stage execution.
"""

from .ocr_subprocess import OCRSubprocess

__all__ = ["OCRSubprocess"]
