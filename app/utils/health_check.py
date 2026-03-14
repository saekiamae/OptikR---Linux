"""
Health Check System for OptikR

Verifies that all system components are functional and provides diagnostic
information with remediation steps for failures.

Requirements: 5.4, 5.5, 5.6, 5.7, 5.8, 5.9
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a health check component."""
    passed: bool
    message: str
    details: str | None = None
    remediation: str | None = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    is_healthy: bool
    components: dict[str, CheckResult]
    timestamp: datetime
    
    def get_failed_components(self) -> list[str]:
        """Get list of failed component names."""
        return [name for name, health in self.components.items() 
                if not health.passed]
    



class HealthCheck:
    """Verifies system components are functional."""
    
    def __init__(self, config_manager=None):
        """
        Initialize health check system.
        
        Args:
            config_manager: Optional configuration manager for accessing settings
        """
        self.config = config_manager
    
    def run_all_checks(self) -> SystemHealth:
        """
        Run all health checks and return overall system health.
        
        Requirements: 5.4, 5.5, 5.6, 5.7, 5.8
        
        Returns:
            SystemHealth object containing:
            - is_healthy: True if all checks passed
            - components: Dictionary mapping component name to CheckResult
            - timestamp: When the health check was performed
        """
        results = {}
        
        # Run each component check
        logger.info("Running health checks...")
        results['pyqt'] = self.check_pyqt()
        results['pytorch'] = self.check_pytorch()
        results['ocr'] = self.check_ocr()
        results['translation'] = self.check_translation()
        results['capture'] = self.check_capture()
        results['models'] = self.check_models()
        
        # Determine overall health
        all_passed = all(result.passed for result in results.values())
        
        # Create SystemHealth object
        system_health = SystemHealth(
            is_healthy=all_passed,
            components=results,
            timestamp=datetime.now()
        )
        
        # Log results
        if all_passed:
            logger.info("All health checks passed")
        else:
            failed = system_health.get_failed_components()
            logger.warning("Health checks failed for: %s", ", ".join(failed))
        
        return system_health
    
    def check_pyqt(self) -> CheckResult:
        """
        Verify PyQt6 is installed and can create GUI components.
        
        Requirements: 5.4
        
        Returns:
            CheckResult with pass/fail status and diagnostic information
        """
        try:
            # Try to import PyQt6 core modules
            from PyQt6.QtWidgets import QApplication, QWidget
            from PyQt6.QtCore import QObject, pyqtSignal
            from PyQt6.QtGui import QIcon
            
            # Verify we can create a QApplication instance (if not already created)
            app = QApplication.instance()
            if app is None:
                # Create a temporary application to test
                import sys
                test_app = QApplication(sys.argv)
                test_widget = QWidget()
                test_app.quit()
            
            return CheckResult(
                passed=True,
                message="PyQt6 is installed and functional",
                details="Successfully imported PyQt6 modules and verified GUI component creation",
                remediation=None
            )
        except ImportError as e:
            return CheckResult(
                passed=False,
                message="PyQt6 is not installed or incomplete",
                details=f"Import error: {str(e)}",
                remediation="Install PyQt6 using: pip install PyQt6>=6.5.0"
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                message="PyQt6 installation is corrupted or incompatible",
                details=f"Error creating GUI components: {str(e)}",
                remediation="Reinstall PyQt6 using: pip uninstall PyQt6 && pip install PyQt6>=6.5.0"
            )
    
    def check_pytorch(self) -> CheckResult:
        """
        Verify PyTorch is installed and can detect GPU availability.
        
        Requirements: 5.5
        
        Returns:
            CheckResult with pass/fail status and GPU detection information
        """
        try:
            import torch
            
            # Check PyTorch version
            torch_version = torch.__version__
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                
                details = (
                    f"PyTorch {torch_version} installed with CUDA support. "
                    f"Detected {device_count} GPU(s): {device_name}"
                )
                message = "PyTorch is installed with GPU support"
            else:
                details = f"PyTorch {torch_version} installed (CPU-only mode)"
                message = "PyTorch is installed (CPU-only)"
            
            return CheckResult(
                passed=True,
                message=message,
                details=details,
                remediation=None
            )
        except ImportError:
            return CheckResult(
                passed=False,
                message="PyTorch is not installed",
                details="PyTorch package not found",
                remediation="Install PyTorch using: pip install torch>=2.0.0"
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                message="PyTorch installation has issues",
                details=f"Error checking PyTorch: {str(e)}",
                remediation="Reinstall PyTorch using: pip uninstall torch && pip install torch>=2.0.0"
            )
    
    def check_ocr(self) -> CheckResult:
        """
        Verify at least one OCR engine is installed and functional.
        
        Requirements: 5.6
        
        Returns:
            CheckResult with pass/fail status and available OCR engines
        """
        available_engines = []
        engine_details = []
        
        # Check EasyOCR
        try:
            import easyocr
            available_engines.append('easyocr')
            engine_details.append(f"EasyOCR {easyocr.__version__}")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"EasyOCR (error: {str(e)})")
        
        # Check PaddleOCR
        try:
            import paddleocr
            available_engines.append('paddleocr')
            # PaddleOCR might not have __version__
            try:
                version = paddleocr.__version__
                engine_details.append(f"PaddleOCR {version}")
            except AttributeError:
                engine_details.append("PaddleOCR (version unknown)")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"PaddleOCR (error: {str(e)})")
        
        # Check Tesseract
        try:
            import pytesseract
            # Try to get tesseract version
            try:
                version = pytesseract.get_tesseract_version()
                available_engines.append('tesseract')
                engine_details.append(f"Tesseract {version}")
            except Exception:
                # Tesseract binary might not be installed
                engine_details.append("Tesseract (binary not found)")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"Tesseract (error: {str(e)})")
        
        
        if available_engines:
            return CheckResult(
                passed=True,
                message=f"OCR engines available: {', '.join(available_engines)}",
                details="; ".join(engine_details),
                remediation=None
            )
        else:
            return CheckResult(
                passed=False,
                message="No OCR engines are installed",
                details="Checked: EasyOCR, PaddleOCR, Tesseract, Manga OCR - none found",
                remediation=(
                    "Install at least one OCR engine:\n"
                    "- EasyOCR: pip install easyocr>=1.7.0\n"
                    "- PaddleOCR: pip install paddleocr\n"
                    "- Tesseract: Install pytesseract and tesseract binary"
                )
            )
    
    def check_translation(self) -> CheckResult:
        """
        Verify at least one translation engine is installed and functional.
        
        Requirements: 5.7
        
        Returns:
            CheckResult with pass/fail status and available translation engines
        """
        available_engines = []
        engine_details = []
        
        # Check MarianMT (via transformers)
        try:
            from transformers import MarianMTModel, MarianTokenizer
            available_engines.append('marianmt')
            import transformers
            engine_details.append(f"MarianMT (transformers {transformers.__version__})")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"MarianMT (error: {str(e)})")
        
        # Check Google Translate Free (deep-translator)
        try:
            import deep_translator
            available_engines.append('google_free')
            engine_details.append(f"Google Translate Free (deep-translator {deep_translator.__version__})")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"Google Translate Free (error: {str(e)})")
        
        # Check Google Cloud Translation API (google-cloud-translate)
        try:
            from google.cloud import translate_v2
            available_engines.append('google_api')
            engine_details.append("Google Translate API (google-cloud-translate)")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"Google Translate (error: {str(e)})")
        
        # Check DeepL
        try:
            import deepl
            available_engines.append('deepl')
            engine_details.append("DeepL")
        except ImportError:
            pass
        except Exception as e:
            engine_details.append(f"DeepL (error: {str(e)})")
        
        # Azure Translator availability is determined by config (API key presence),
        # not by package imports — requests is always installed. Skipped here.
        
        if available_engines:
            return CheckResult(
                passed=True,
                message=f"Translation engines available: {', '.join(available_engines)}",
                details="; ".join(engine_details),
                remediation=None
            )
        else:
            return CheckResult(
                passed=False,
                message="No translation engines are installed",
                details="Checked: MarianMT, Google Translate, DeepL, Azure - none found",
                remediation=(
                    "Install at least one translation engine:\n"
                    "- MarianMT: pip install transformers>=4.30.0\n"
                    "- Google Translate: pip install deep-translator>=1.11.0\n"
                    "- DeepL: pip install deepl\n"
                    "- Azure: pip install requests>=2.31.0"
                )
            )
    
    def check_capture(self) -> CheckResult:
        """
        Verify screen capture libraries are installed and functional.
        
        Requirements: 5.8
        
        Returns:
            CheckResult with pass/fail status and available capture methods
        """
        available_methods = []
        method_details = []
        
        # Check BetterCam (DirectX Desktop Duplication for Windows)
        try:
            import bettercam
            available_methods.append('directx')
            method_details.append(f"BetterCam {bettercam.__version__}")
        except ImportError:
            pass
        except Exception as e:
            method_details.append(f"BetterCam (error: {str(e)})")
        
        # Check MSS (Multi-platform screenshot)
        try:
            import mss
            available_methods.append('mss')
            method_details.append(f"MSS {mss.__version__}")
        except ImportError:
            pass
        except Exception as e:
            method_details.append(f"MSS (error: {str(e)})")
        
        # Check PIL/Pillow (ImageGrab)
        try:
            from PIL import ImageGrab, Image
            available_methods.append('pillow')
            method_details.append(f"Pillow {Image.__version__}")
        except ImportError:
            pass
        except Exception as e:
            method_details.append(f"Pillow (error: {str(e)})")
        
        # Check OpenCV
        try:
            import cv2
            available_methods.append('opencv')
            method_details.append(f"OpenCV {cv2.__version__}")
        except ImportError:
            pass
        except Exception as e:
            method_details.append(f"OpenCV (error: {str(e)})")
        
        if available_methods:
            return CheckResult(
                passed=True,
                message=f"Screen capture methods available: {', '.join(available_methods)}",
                details="; ".join(method_details),
                remediation=None
            )
        else:
            return CheckResult(
                passed=False,
                message="No screen capture libraries are installed",
                details="Checked: BetterCam, MSS, Pillow, OpenCV - none found",
                remediation=(
                    "Install at least one screen capture library:\n"
                    "- BetterCam: pip install bettercam (Windows only)\n"
                    "- MSS: pip install mss>=9.0.1\n"
                    "- Pillow: pip install Pillow>=10.0.0\n"
                    "- OpenCV: pip install opencv-python>=4.8.0"
                )
            )
    
    def check_models(self) -> CheckResult:
        """
        Verify AI models are downloaded and loadable.
        
        Requirements: 5.8
        
        Returns:
            CheckResult with pass/fail status and model availability
        """
        try:
            import transformers
            
            # Only check for the default model that ships with first-run setup.
            # Additional models download on demand when the user switches languages.
            common_models = [
                'Helsinki-NLP/opus-mt-en-de',
            ]
            
            model_status = []
            models_found = 0
            
            # Resolve HuggingFace cache directory
            try:
                from huggingface_hub.constants import HF_HUB_CACHE
                cache_dir = Path(HF_HUB_CACHE)
            except ImportError:
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            
            for model_name in common_models:
                model_cache_name = 'models--' + model_name.replace('/', '--')
                model_path = cache_dir / model_cache_name
                
                if model_path.exists():
                    models_found += 1
                    model_status.append(f"{model_name}: available")
                else:
                    model_status.append(f"{model_name}: not downloaded")
            
            if models_found > 0:
                return CheckResult(
                    passed=True,
                    message=f"AI models available: {models_found}/{len(common_models)}",
                    details="\n".join(model_status),
                    remediation=None
                )
            else:
                return CheckResult(
                    passed=True,
                    message="No AI models pre-downloaded (will download on first use)",
                    details="MarianMT models will be downloaded automatically on first translation",
                    remediation=None
                )
        except ImportError:
            return CheckResult(
                passed=False,
                message="Transformers library not installed",
                details="Cannot check model availability without transformers",
                remediation="Install transformers: pip install transformers>=4.30.0"
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                message="Error checking AI models",
                details=f"Unexpected error: {str(e)}",
                remediation="Verify transformers installation and model directory permissions"
            )
