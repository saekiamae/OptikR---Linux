"""
Preprocessing Layer Implementation

Implements image preprocessing functionality with frame differencing and ROI optimization.
Provides scaling, grayscale conversion, denoising, and adaptive thresholding for OCR accuracy.
"""

import logging
import time
from collections import deque
from typing import Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

from ..models import Frame, Rectangle, PerformanceProfile
from ..interfaces import IPreprocessingLayer
from .frame_differencing import (
    FrameDifferencingSystem, DifferenceConfig, DifferenceMethod,
    SensitivityLevel, DifferenceResult
)
from .roi_detection import TextRegionDetector


@dataclass
class PreprocessingProfile:
    """Configuration profile for preprocessing operations."""
    # Scaling settings
    target_width: int | None = None
    target_height: int | None = None
    maintain_aspect_ratio: bool = True
    interpolation_method: int = cv2.INTER_CUBIC
    
    # Grayscale conversion
    convert_to_grayscale: bool = True
    grayscale_method: str = "weighted"  # "weighted", "average", "luminance", "desaturate", "max_channel", "adaptive"
    
    # Denoising settings
    enable_denoising: bool = True
    denoise_strength: float = 3.0
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21
    
    # Thresholding settings
    enable_adaptive_threshold: bool = True
    threshold_method: int = cv2.THRESH_BINARY
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    block_size: int = 11
    c_constant: float = 2.0
    
    # Frame differencing settings
    enable_frame_differencing: bool = True
    difference_method: DifferenceMethod = DifferenceMethod.ABSOLUTE
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    
    # ROI detection settings
    enable_roi_detection: bool = True
    roi_min_area: int = 100
    roi_max_count: int = 10
    roi_min_width: int = 50
    roi_min_height: int = 20
    roi_max_width: int = 2000
    roi_max_height: int = 1000
    roi_padding: int = 10
    roi_merge_distance: int = 20
    roi_confidence_threshold: float = 0.3
    roi_adaptive_threshold: bool = True
    roi_use_morphology: bool = True
    
    # Content-specific settings
    content_type: str = "mixed"  # "text", "mixed", "graphics", "screenshot"
    
    # Advanced preprocessing options
    enable_contrast_enhancement: bool = False
    enable_sharpening: bool = False
    enable_morphological_operations: bool = False
    
    @classmethod
    def create_for_performance_profile(cls, profile: PerformanceProfile, content_type: str = "mixed") -> 'PreprocessingProfile':
        """
        Create preprocessing profile optimized for specific performance level and content type.
        
        Args:
            profile: Performance profile
            content_type: Type of content being processed
            
        Returns:
            PreprocessingProfile: Optimized preprocessing configuration
        """
        base_config = {}
        
        # Performance-based settings
        if profile == PerformanceProfile.LOW:
            base_config.update({
                'target_width': 640,
                'target_height': 480,
                'interpolation_method': cv2.INTER_LINEAR,
                'denoise_strength': 1.0,
                'denoise_template_window_size': 5,
                'denoise_search_window_size': 15,
                'block_size': 9,
                'sensitivity_level': SensitivityLevel.LOW,
                'roi_max_count': 3,
                'enable_contrast_enhancement': False,
                'enable_sharpening': False,
                'enable_morphological_operations': False
            })
        elif profile == PerformanceProfile.NORMAL:
            base_config.update({
                'target_width': 1280,
                'target_height': 720,
                'interpolation_method': cv2.INTER_CUBIC,
                'denoise_strength': 3.0,
                'sensitivity_level': SensitivityLevel.MEDIUM,
                'roi_max_count': 5,
                'enable_contrast_enhancement': True,
                'enable_sharpening': False,
                'enable_morphological_operations': True
            })
        else:  # HIGH
            base_config.update({
                'target_width': None,  # No scaling for high quality
                'target_height': None,
                'interpolation_method': cv2.INTER_LANCZOS4,
                'denoise_strength': 5.0,
                'denoise_template_window_size': 9,
                'denoise_search_window_size': 25,
                'block_size': 15,
                'sensitivity_level': SensitivityLevel.HIGH,
                'roi_max_count': 10,
                'enable_contrast_enhancement': True,
                'enable_sharpening': True,
                'enable_morphological_operations': True
            })
        
        # Content-specific adjustments
        if content_type == "text":
            base_config.update({
                'grayscale_method': "luminance",
                'enable_adaptive_threshold': True,
                'enable_sharpening': True,
                'c_constant': 3.0
            })
        elif content_type == "graphics":
            base_config.update({
                'grayscale_method': "weighted",
                'enable_adaptive_threshold': False,
                'enable_contrast_enhancement': True,
                'denoise_strength': base_config.get('denoise_strength', 3.0) * 0.7
            })
        elif content_type == "screenshot":
            base_config.update({
                'grayscale_method': "adaptive",
                'enable_adaptive_threshold': True,
                'enable_contrast_enhancement': True,
                'enable_morphological_operations': True
            })
        
        base_config['content_type'] = content_type
        return cls(**base_config)
    
    @classmethod
    def create_for_content_type(cls, content_type: str) -> 'PreprocessingProfile':
        """
        Create preprocessing profile optimized for specific content type.
        
        Args:
            content_type: Type of content ("text", "mixed", "graphics", "screenshot")
            
        Returns:
            PreprocessingProfile: Content-optimized preprocessing configuration
        """
        return cls.create_for_performance_profile(PerformanceProfile.NORMAL, content_type)


class PreprocessingLayer(IPreprocessingLayer):
    """
    Main preprocessing layer implementation.
    
    Provides comprehensive image preprocessing with frame differencing,
    ROI optimization, and configurable filtering operations.
    """
    
    def __init__(self, logger: logging.Logger | None = None, config_manager=None):
        """
        Initialize preprocessing layer.
        
        Args:
            logger: Optional logger for debugging
            config_manager: Optional config manager — when provided, ROI detection
                parameters are read from roi_detection.* config keys, with
                PreprocessingProfile defaults as fallback.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self._config_manager = config_manager

        # Initialize frame differencing system
        self._frame_differencing = FrameDifferencingSystem(logger=logger)
        
        # Initialize small text enhancer
        try:
            from .small_text_enhancer import SmallTextEnhancer
            self._small_text_enhancer = SmallTextEnhancer(logger=logger)
            self._small_text_enhancement_enabled = False
        except ImportError as e:
            self.logger.warning(f"Small text enhancer not available: {e}")
            self._small_text_enhancer = None
            self._small_text_enhancement_enabled = False
        
        # Current configuration
        self._current_profile = PreprocessingProfile.create_for_performance_profile(
            PerformanceProfile.NORMAL
        )
        
        # Read ROI settings from config if available, falling back to profile defaults
        def _cfg(key, fallback):
            if config_manager:
                return config_manager.get_setting(key, fallback)
            return fallback
        
        # Initialize text region detector
        self._text_region_detector = TextRegionDetector(
            min_area=self._current_profile.roi_min_area,
            max_count=self._current_profile.roi_max_count,
            min_width=_cfg('roi_detection.min_region_width', self._current_profile.roi_min_width),
            min_height=_cfg('roi_detection.min_region_height', self._current_profile.roi_min_height),
            max_width=_cfg('roi_detection.max_region_width', self._current_profile.roi_max_width),
            max_height=_cfg('roi_detection.max_region_height', self._current_profile.roi_max_height),
            padding=_cfg('roi_detection.padding', self._current_profile.roi_padding),
            merge_distance=_cfg('roi_detection.merge_distance', self._current_profile.roi_merge_distance),
            confidence_threshold=_cfg('roi_detection.confidence_threshold', self._current_profile.roi_confidence_threshold),
            use_adaptive_threshold=_cfg('roi_detection.adaptive_threshold', self._current_profile.roi_adaptive_threshold),
            use_morphology=_cfg('roi_detection.use_morphology', self._current_profile.roi_use_morphology),
            logger=logger,
        )
        
        # Performance tracking
        self._processing_times: deque = deque(maxlen=100)
        
        # Statistics
        self._stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'roi_optimizations': 0,
            'frame_differences_calculated': 0
        }
        
        self.logger.info("Preprocessing layer initialized")
    
    def preprocess(self, frame: Frame, profile: PerformanceProfile = PerformanceProfile.NORMAL) -> Frame:
        """
        Preprocess frame for optimal OCR accuracy.

        Args:
            frame: Input frame to preprocess
            profile: Performance profile for optimization

        Returns:
            Frame: Preprocessed frame with ROI metadata
        """
        start_time = time.time()

        try:
            # Update profile if needed
            if profile != self._get_current_performance_profile():
                self._update_profile_for_performance(profile)

            # Start with original frame data
            processed_data = frame.data.copy()
            orig_h, orig_w = processed_data.shape[:2]

            full_preprocessing = True
            if self._config_manager:
                full_preprocessing = self._config_manager.get_setting(
                    'ocr.preprocessing_enabled', False,
                )

            # Apply small text enhancement if enabled (before other preprocessing)
            if full_preprocessing and self._small_text_enhancement_enabled and self._small_text_enhancer:
                enhanced_frame = self._small_text_enhancer.enhance_frame(frame)
                processed_data = enhanced_frame.data.copy()
                self.logger.debug("Small text enhancement applied")

            # Apply scaling if configured
            if (self._current_profile.target_width is not None and 
                self._current_profile.target_height is not None):
                processed_data = self._apply_scaling(processed_data)

            if full_preprocessing:
                # Convert to grayscale if enabled
                if self._current_profile.convert_to_grayscale:
                    processed_data = self._convert_to_grayscale(processed_data)

                # Apply denoising if enabled
                if self._current_profile.enable_denoising:
                    processed_data = self._apply_denoising(processed_data)

                # Apply adaptive thresholding if enabled
                if self._current_profile.enable_adaptive_threshold:
                    processed_data = self._apply_adaptive_threshold(processed_data)

            # Detect ROIs on the ORIGINAL unscaled frame.
            # The manga bubble detector relies on clean ink lines that
            # are destroyed by the downscale + sharpening in _apply_scaling.
            # After detection, scale the ROI coordinates to match the
            # processed (scaled) output so OCRStage crops correctly.
            roi_list: list[Rectangle] = []
            if self._current_profile.enable_roi_detection:
                roi_list = self.detect_roi(frame)

                # Scale ROI coordinates when the output was resized
                proc_h, proc_w = processed_data.shape[:2]
                if (proc_w != orig_w or proc_h != orig_h) and roi_list:
                    sx = proc_w / orig_w
                    sy = proc_h / orig_h
                    roi_list = [
                        Rectangle(
                            x=int(r.x * sx),
                            y=int(r.y * sy),
                            width=max(1, int(r.width * sx)),
                            height=max(1, int(r.height * sy)),
                        )
                        for r in roi_list
                    ]

            # Create preprocessed frame
            preprocessed_frame = Frame(
                data=processed_data,
                timestamp=frame.timestamp,
                source_region=frame.source_region,
                metadata={
                    **frame.metadata,
                    'preprocessing_applied': True,
                    'preprocessing_profile': profile.value,
                    'original_shape': frame.data.shape,
                    'processed_shape': processed_data.shape,
                    'roi_regions': roi_list,
                }
            )

            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)

            return preprocessed_frame

        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {e}")
            # Return original frame on error
            return frame
    
    def detect_roi(self, frame: Frame) -> list[Rectangle]:
        """
        Detect regions of interest containing text.

        Uses manga bubble detection (flood-fill) as the primary strategy,
        falling back to frame differencing and edge-based detection.

        Args:
            frame: Input frame for ROI detection

        Returns:
            list[Rectangle]: List of detected ROI rectangles
        """
        try:
            if not self._current_profile.enable_roi_detection:
                return [Rectangle(0, 0, frame.width, frame.height)]

            # Primary: frame differencing (skip on first frame to avoid
            # full-frame ROI that defeats per-bubble OCR)
            if self._current_profile.enable_frame_differencing:
                has_prev = self._frame_differencing._previous_frame is not None
                difference_result, optimized_rois = self._frame_differencing.process_frame(frame)
                self._stats['frame_differences_calculated'] += 1

                # Only use frame-differencing ROIs when we actually have a
                # previous frame to compare against.  On the very first frame
                # the system returns a single full-frame ROI which is useless.
                if has_prev and optimized_rois:
                    self._stats['roi_optimizations'] += 1
                    return optimized_rois

            # Fallback: edge-based text region detection
            return self._text_region_detector.detect(frame.data)

        except Exception as e:
            self.logger.error(f"ROI detection failed: {e}")
            return [Rectangle(0, 0, frame.width, frame.height)]
    
    def _apply_scaling(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intelligent scaling to image for optimal OCR processing.
        
        Uses advanced interpolation methods and maintains text clarity.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Scaled image optimized for text recognition
        """
        target_w = self._current_profile.target_width
        target_h = self._current_profile.target_height
        
        if target_w is None or target_h is None:
            return image
        
        current_h, current_w = image.shape[:2]
        
        # Skip scaling if already at target size
        if current_w == target_w and current_h == target_h:
            return image
        
        if self._current_profile.maintain_aspect_ratio:
            # Calculate scaling factor to fit within target dimensions
            scale_w = target_w / current_w
            scale_h = target_h / current_h
            scale = min(scale_w, scale_h)
            
            new_w = int(current_w * scale)
            new_h = int(current_h * scale)
        else:
            new_w = target_w
            new_h = target_h
        
        # Use different interpolation methods based on scaling direction
        if new_w * new_h > current_w * current_h:
            # Upscaling - use high-quality interpolation
            interpolation = cv2.INTER_CUBIC if self._current_profile.interpolation_method == cv2.INTER_CUBIC else cv2.INTER_LANCZOS4
        else:
            # Downscaling - use area interpolation for better text preservation
            interpolation = cv2.INTER_AREA
        
        try:
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            # Apply sharpening after scaling to enhance text clarity
            if new_w * new_h < current_w * current_h * 0.5:  # Significant downscaling
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                scaled_image = cv2.filter2D(scaled_image, -1, kernel)
                scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
            
            return scaled_image
            
        except Exception as e:
            self.logger.error(f"Scaling failed: {e}")
            return image
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale using optimal method for text recognition.
        
        Implements multiple conversion methods optimized for different content types.
        
        Args:
            image: Input color image
            
        Returns:
            np.ndarray: Grayscale image optimized for OCR
        """
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        try:
            if self._current_profile.grayscale_method == "weighted":
                # OpenCV's optimized weighted conversion (0.299*R + 0.587*G + 0.114*B)
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            elif self._current_profile.grayscale_method == "average":
                # Simple average method
                return np.mean(image, axis=2).astype(np.uint8)
                
            elif self._current_profile.grayscale_method == "luminance":
                # ITU-R BT.709 luminance formula for better text contrast
                if image.shape[2] >= 3:  # BGR or BGRA
                    # OpenCV uses BGR format
                    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
                    return luminance
                else:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
            elif self._current_profile.grayscale_method == "desaturate":
                # Desaturation method - preserves brightness better for text
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return hsv[:, :, 2]  # Return value channel
                
            elif self._current_profile.grayscale_method == "max_channel":
                # Maximum channel method - good for high contrast text
                return np.max(image, axis=2).astype(np.uint8)
                
            elif self._current_profile.grayscale_method == "adaptive":
                # Adaptive method based on image content analysis
                # Analyze image to determine best conversion method
                std_per_channel = np.std(image, axis=(0, 1))
                max_std_channel = np.argmax(std_per_channel)
                
                if max_std_channel == 1:  # Green channel has most variation
                    return image[:, :, 1]  # Use green channel
                else:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Use weighted
                    
            else:
                # Default to OpenCV's weighted conversion
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            self.logger.error(f"Grayscale conversion failed: {e}")
            # Fallback to simple weighted conversion
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced denoising algorithms optimized for text preservation.
        
        Uses multiple denoising techniques based on image characteristics and performance profile.
        
        Args:
            image: Input image with noise
            
        Returns:
            np.ndarray: Denoised image with preserved text clarity
        """
        try:
            # Analyze image noise level to select appropriate denoising method
            noise_level = self._estimate_noise_level(image)
            
            if noise_level < 5:  # Low noise - minimal processing
                return self._apply_light_denoising(image)
            elif noise_level < 15:  # Medium noise - standard denoising
                return self._apply_standard_denoising(image)
            else:  # High noise - aggressive denoising
                return self._apply_aggressive_denoising(image)
                
        except Exception as e:
            self.logger.error(f"Denoising failed: {e}")
            return image
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in image using Laplacian variance method.
        
        Args:
            image: Input image
            
        Returns:
            float: Estimated noise level (higher = more noise)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance as noise estimate
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
            
        except Exception:
            return 10.0  # Default medium noise level
    
    def _apply_light_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply light denoising for low-noise images."""
        return cv2.bilateralFilter(image, 5, 20, 20)
    
    def _apply_standard_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply standard denoising for medium-noise images."""
        if len(image.shape) == 2:
            # Non-local means denoising for grayscale
            return cv2.fastNlMeansDenoising(
                image,
                None,
                self._current_profile.denoise_strength,
                self._current_profile.denoise_template_window_size,
                self._current_profile.denoise_search_window_size
            )
        else:
            # Non-local means denoising for color
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self._current_profile.denoise_strength,
                self._current_profile.denoise_strength,
                self._current_profile.denoise_template_window_size,
                self._current_profile.denoise_search_window_size
            )
    
    def _apply_aggressive_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply aggressive denoising for high-noise images."""
        # Multi-stage denoising approach
        
        # Stage 1: Bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 9, 50, 50)
        
        # Stage 2: Non-local means with higher strength
        strength = min(self._current_profile.denoise_strength * 1.5, 10.0)
        
        if len(denoised.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(
                denoised,
                None,
                strength,
                self._current_profile.denoise_template_window_size,
                self._current_profile.denoise_search_window_size
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                denoised,
                None,
                strength,
                strength,
                self._current_profile.denoise_template_window_size,
                self._current_profile.denoise_search_window_size
            )
        
        # Stage 3: Light morphological operations to clean up artifacts
        if len(denoised.shape) == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intelligent adaptive thresholding based on content analysis.
        
        Analyzes image characteristics to select optimal thresholding parameters
        for maximum text clarity and OCR accuracy.
        
        Args:
            image: Input image for thresholding
            
        Returns:
            np.ndarray: Binary image optimized for text recognition
        """
        try:
            # Ensure grayscale for thresholding
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Analyze image characteristics for optimal thresholding
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Select thresholding method based on image characteristics
            if std_intensity < 30:  # Low contrast image
                return self._apply_low_contrast_threshold(gray, mean_intensity)
            elif std_intensity > 80:  # High contrast image
                return self._apply_high_contrast_threshold(gray)
            else:  # Normal contrast image
                return self._apply_standard_adaptive_threshold(gray)
                
        except Exception as e:
            self.logger.error(f"Adaptive thresholding failed: {e}")
            # Fallback to simple threshold
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    def _apply_low_contrast_threshold(self, gray: np.ndarray, mean_intensity: float) -> np.ndarray:
        """Apply thresholding optimized for low contrast images."""
        # Use CLAHE to enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive threshold with adjusted parameters
        block_size = max(self._current_profile.block_size + 4, 15)  # Larger block for low contrast
        c_constant = self._current_profile.c_constant + 2  # Higher constant for better separation
        
        return cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_constant
        )
    
    def _apply_high_contrast_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply thresholding optimized for high contrast images."""
        # For high contrast, Otsu's method often works well
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try adaptive threshold with smaller block size
        block_size = max(self._current_profile.block_size - 2, 7)  # Smaller block for high contrast
        adaptive_binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self._current_profile.c_constant
        )
        
        # Combine both methods using bitwise operations for best result
        combined = cv2.bitwise_and(otsu_binary, adaptive_binary)
        
        # If combined result is too sparse, use the better individual result
        white_pixels_combined = np.sum(combined == 255)
        white_pixels_otsu = np.sum(otsu_binary == 255)
        
        if white_pixels_combined < white_pixels_otsu * 0.3:  # Too much lost
            return otsu_binary
        else:
            return combined
    
    def _apply_standard_adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply standard adaptive thresholding for normal contrast images."""
        # Try both Gaussian and Mean adaptive methods
        gaussian_binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._current_profile.block_size,
            self._current_profile.c_constant
        )
        
        mean_binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self._current_profile.block_size,
            self._current_profile.c_constant
        )
        
        # Select the method that produces more reasonable text-like regions
        gaussian_score = self._evaluate_threshold_quality(gaussian_binary)
        mean_score = self._evaluate_threshold_quality(mean_binary)
        
        return gaussian_binary if gaussian_score >= mean_score else mean_binary
    
    def _evaluate_threshold_quality(self, binary_image: np.ndarray) -> float:
        """
        Evaluate the quality of a thresholded image for text recognition.
        
        Args:
            binary_image: Binary thresholded image
            
        Returns:
            float: Quality score (higher is better)
        """
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )
            
            if num_labels <= 1:  # No foreground objects
                return 0.0
            
            # Analyze component characteristics
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            widths = stats[1:, cv2.CC_STAT_WIDTH]
            heights = stats[1:, cv2.CC_STAT_HEIGHT]
            
            # Calculate quality metrics
            # 1. Reasonable component sizes (not too small or too large)
            reasonable_sizes = np.sum((areas >= 20) & (areas <= binary_image.size * 0.1))
            
            # 2. Text-like aspect ratios
            aspect_ratios = widths / np.maximum(heights, 1)
            reasonable_aspects = np.sum((aspect_ratios >= 0.1) & (aspect_ratios <= 10))
            
            # 3. Good distribution of component sizes
            size_variance = np.var(areas) if len(areas) > 1 else 0
            
            # Combine metrics into quality score
            quality_score = (
                reasonable_sizes * 0.4 +
                reasonable_aspects * 0.4 +
                min(size_variance / 1000, 10) * 0.2
            )
            
            return quality_score
            
        except Exception:
            return 0.0
    
    def _update_profile_for_performance(self, profile: PerformanceProfile) -> None:
        """Update preprocessing profile based on performance requirements."""
        self._current_profile = PreprocessingProfile.create_for_performance_profile(profile)
        
        # Update frame differencing configuration
        diff_config = DifferenceConfig(
            method=self._current_profile.difference_method,
            sensitivity=self._current_profile.sensitivity_level
        )
        self._frame_differencing.update_config(diff_config)
        
        self.logger.info(f"Preprocessing profile updated for {profile.value} performance")
    
    def _get_current_performance_profile(self) -> PerformanceProfile:
        """Get current performance profile based on configuration."""
        if (self._current_profile.target_width == 640 and 
            self._current_profile.sensitivity_level == SensitivityLevel.LOW):
            return PerformanceProfile.LOW
        elif (self._current_profile.target_width == 1280 and 
              self._current_profile.sensitivity_level == SensitivityLevel.MEDIUM):
            return PerformanceProfile.NORMAL
        else:
            return PerformanceProfile.HIGH
    
    def _update_processing_stats(self, processing_time: float) -> None:
        """Update processing statistics."""
        self._stats['frames_processed'] += 1
        self._stats['total_processing_time'] += processing_time
        self._processing_times.append(processing_time)
    
    def configure_frame_differencing(self, config: DifferenceConfig) -> None:
        """
        Configure frame differencing settings.
        
        Args:
            config: Frame differencing configuration
        """
        self._frame_differencing.update_config(config)
        self.logger.info("Frame differencing configuration updated")
    
    def set_frame_differencing_enabled(self, enabled: bool) -> None:
        """
        Enable or disable frame differencing.
        
        Args:
            enabled: Whether to enable frame differencing
        """
        self._current_profile.enable_frame_differencing = enabled
        self._frame_differencing.set_optimization_enabled(enabled)
        self.logger.info(f"Frame differencing {'enabled' if enabled else 'disabled'}")
    
    def set_small_text_enhancement_enabled(self, enabled: bool, denoise: bool = False, binarize: bool = False) -> None:
        """
        Enable or disable small text enhancement.
        
        Args:
            enabled: Whether to enable small text enhancement
            denoise: Whether to enable noise reduction
            binarize: Whether to enable binarization
        """
        if self._small_text_enhancer is None:
            self.logger.warning("Small text enhancer not available")
            return
        
        self._small_text_enhancement_enabled = enabled
        
        # Configure the enhancer with the options
        if enabled and self._small_text_enhancer:
            self._small_text_enhancer.denoise = denoise
            self._small_text_enhancer.binarize = binarize
            self.logger.info(f"Small text enhancement enabled (denoise={denoise}, binarize={binarize})")
        else:
            self.logger.info("Small text enhancement disabled")
    
    def configure_small_text_enhancement(self, scale_factor: float = 2.0, 
                                        sharpen_strength: float = 1.5,
                                        contrast_enhancement: bool = True,
                                        denoise: bool = True) -> None:
        """
        Configure small text enhancement parameters.
        
        Args:
            scale_factor: Upscaling factor (1.5-3.0 recommended)
            sharpen_strength: Sharpening strength (0.5-2.0)
            contrast_enhancement: Enable contrast enhancement
            denoise: Enable denoising
        """
        if self._small_text_enhancer is None:
            self.logger.warning("Small text enhancer not available")
            return
        
        self._small_text_enhancer.configure(
            scale_factor=scale_factor,
            sharpen_strength=sharpen_strength,
            contrast_enhancement=contrast_enhancement,
            denoise=denoise
        )
        self.logger.info(f"Small text enhancement configured: scale={scale_factor}x")
    
    def get_preprocessing_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive preprocessing statistics.
        
        Returns:
            dict[str, Any]: Preprocessing statistics
        """
        avg_processing_time = (self._stats['total_processing_time'] / 
                             self._stats['frames_processed'] 
                             if self._stats['frames_processed'] > 0 else 0)
        
        stats = {
            **self._stats,
            'average_processing_time': avg_processing_time,
            'current_profile': self._get_current_performance_profile().value,
            'frame_differencing_enabled': self._current_profile.enable_frame_differencing,
            'roi_detection_enabled': self._current_profile.enable_roi_detection,
            'frame_differencing_stats': self._frame_differencing.get_system_statistics()
        }
        
        if self._processing_times:
            stats.update({
                'recent_avg_processing_time': np.mean(self._processing_times),
                'min_processing_time': np.min(self._processing_times),
                'max_processing_time': np.max(self._processing_times)
            })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'roi_optimizations': 0,
            'frame_differences_calculated': 0
        }
        self._processing_times.clear()
        self._frame_differencing.reset_system()
        self.logger.info("Preprocessing statistics reset")
    
    def update_preprocessing_profile(self, profile: PreprocessingProfile) -> None:
        """
        Update the current preprocessing profile.
        
        Args:
            profile: New preprocessing profile to use
        """
        self._current_profile = profile
        
        # Update frame differencing configuration
        if hasattr(profile, 'difference_method') and hasattr(profile, 'sensitivity_level'):
            diff_config = DifferenceConfig(
                method=profile.difference_method,
                sensitivity=profile.sensitivity_level
            )
            self._frame_differencing.update_config(diff_config)
        
        self.logger.info(f"Preprocessing profile updated for {profile.content_type} content")
    
    def get_preprocessing_profile(self) -> PreprocessingProfile:
        """
        Get the current preprocessing profile.
        
        Returns:
            PreprocessingProfile: Current preprocessing configuration
        """
        return self._current_profile
    