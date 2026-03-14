"""
Text Region Detection Module

Detects text regions in images using multi-scale edge detection, heuristic filtering,
and OCR-optimized region merging. Extracted from preprocessing_layer.py.
"""

import logging
from dataclasses import dataclass
import numpy as np
import cv2

from ..models import Rectangle


@dataclass
class _ScoredRegion:
    """Internal wrapper pairing a Rectangle with a detection confidence score."""
    rectangle: Rectangle
    confidence: float = 0.5


class TextRegionDetector:
    """
    Detects text regions in images for OCR optimization.

    Uses multi-scale Canny edge detection, contour analysis, heuristic scoring,
    non-maximum suppression, and region merging to produce OCR-ready ROIs.
    """

    def __init__(self, min_area: int = 100, max_count: int = 10,
                 min_width: int = 50, min_height: int = 20,
                 max_width: int = 2000, max_height: int = 1000,
                 padding: int = 10, merge_distance: int = 20,
                 confidence_threshold: float = 0.3,
                 use_adaptive_threshold: bool = True,
                 use_morphology: bool = True,
                 logger: logging.Logger | None = None):
        self.min_area = min_area
        self.max_count = max_count
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        self.padding = padding
        self.merge_distance = merge_distance
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_morphology = use_morphology
        self.logger = logger or logging.getLogger(__name__)


    def detect(self, frame_data: np.ndarray) -> list[Rectangle]:
        """
        Detect text regions in an image.

        Args:
            frame_data: Image data (BGR or grayscale numpy array).

        Returns:
            List of Rectangle ROIs sorted by confidence (highest first).
        """
        try:
            if len(frame_data.shape) == 3:
                gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_data.copy()

            preprocessed = self._preprocess_for_edge_detection(gray)
            scored_regions = self._detect_text_regions_multiscale(preprocessed)
            refined_regions = self._apply_text_region_heuristics(scored_regions, gray)
            final_scored = self._optimize_rois_for_ocr(refined_regions, gray)

            final_scored.sort(key=lambda sr: sr.confidence, reverse=True)
            return [sr.rectangle for sr in final_scored[:self.max_count]]

        except Exception as e:
            self.logger.error(f"Text region detection failed: {e}", exc_info=True)
            h, w = frame_data.shape[:2]
            return [Rectangle(0, 0, w, h)]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_for_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        try:
            denoised = cv2.bilateralFilter(gray, 5, 50, 50)
            if self.use_adaptive_threshold:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            else:
                enhanced = denoised
            return cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        except Exception as e:
            self.logger.error(f"Edge detection preprocessing failed: {e}", exc_info=True)
            return gray

    # ------------------------------------------------------------------
    # Multi-scale detection
    # ------------------------------------------------------------------

    def _detect_text_regions_multiscale(self, preprocessed: np.ndarray) -> list[_ScoredRegion]:
        try:
            all_regions: list[_ScoredRegion] = []
            scales = [(50, 150), (30, 100), (70, 200)]

            for low_thresh, high_thresh in scales:
                edges = cv2.Canny(preprocessed, low_thresh, high_thresh, apertureSize=3)

                kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
                kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
                if self.use_morphology:
                    h_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
                    v_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
                    combined = cv2.bitwise_or(h_connected, v_connected)
                else:
                    combined = edges

                contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    if (area >= self.min_area
                            and w >= self.min_width and h >= self.min_height
                            and w <= self.max_width and h <= self.max_height):
                        confidence = self._calculate_initial_confidence(contour, area)
                        all_regions.append(_ScoredRegion(Rectangle(x, y, w, h), confidence))

            return self._apply_non_maximum_suppression(all_regions)

        except Exception as e:
            self.logger.error(f"Multi-scale text detection failed: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _calculate_initial_confidence(self, contour: np.ndarray, area: int) -> float:
        try:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            extent = area / max(w * h, 1)

            confidence = 0.0
            if 0.1 <= aspect_ratio <= 20:
                confidence += 0.3 if 0.5 <= aspect_ratio <= 8 else 0.15
            if solidity >= 0.3:
                confidence += 0.25
            if extent >= 0.2:
                confidence += 0.25
            if 100 <= area <= 10000:
                confidence += 0.2
            elif area > 10000:
                confidence += 0.1

            return min(confidence, 1.0)
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Non-maximum suppression
    # ------------------------------------------------------------------

    def _apply_non_maximum_suppression(self, regions: list[_ScoredRegion]) -> list[_ScoredRegion]:
        if not regions:
            return []
        try:
            regions.sort(key=lambda sr: sr.confidence, reverse=True)
            suppressed: list[_ScoredRegion] = []

            for scored in regions:
                should_suppress = False
                for selected in suppressed:
                    if self._calculate_overlap_ratio(scored.rectangle, selected.rectangle) > 0.5:
                        should_suppress = True
                        break
                if not should_suppress:
                    suppressed.append(scored)

            return suppressed
        except Exception as e:
            self.logger.error(f"Non-maximum suppression failed: {e}", exc_info=True)
            return regions

    def _calculate_overlap_ratio(self, rect1: Rectangle, rect2: Rectangle) -> float:
        try:
            x1 = max(rect1.x, rect2.x)
            y1 = max(rect1.y, rect2.y)
            x2 = min(rect1.x + rect1.width, rect2.x + rect2.width)
            y2 = min(rect1.y + rect1.height, rect2.y + rect2.height)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            union = rect1.width * rect1.height + rect2.width * rect2.height - intersection
            return intersection / max(union, 1)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Heuristic filtering
    # ------------------------------------------------------------------

    def _apply_text_region_heuristics(self, regions: list[_ScoredRegion],
                                      gray: np.ndarray) -> list[_ScoredRegion]:
        try:
            refined: list[_ScoredRegion] = []
            for scored in regions:
                r = scored.rectangle
                roi = gray[r.y:r.y + r.height, r.x:r.x + r.width]
                if roi.size == 0:
                    continue
                heuristic = self._calculate_text_heuristic_score(roi, r)
                combined = (scored.confidence + heuristic) / 2
                if combined >= self.confidence_threshold:
                    refined.append(_ScoredRegion(r, combined))
            return refined
        except Exception as e:
            self.logger.error(f"Text region heuristics failed: {e}", exc_info=True)
            return regions

    def _calculate_text_heuristic_score(self, roi: np.ndarray, region: Rectangle) -> float:
        try:
            score = 0.0

            # Intensity variation
            if np.std(roi) > 20:
                score += 0.25
            elif np.std(roi) > 10:
                score += 0.15

            # Edge density
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / roi.size
            if edge_density > 0.1:
                score += 0.25
            elif edge_density > 0.05:
                score += 0.15

            # Horizontal structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(roi.shape[1] // 4, 15), 1))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            if np.sum(h_lines > 0) / roi.size > 0.02:
                score += 0.2

            # Aspect ratio
            aspect = region.width / max(region.height, 1)
            if 1.0 <= aspect <= 6.0:
                score += 0.15
            elif 0.3 <= aspect <= 15.0:
                score += 0.1

            # Size
            area = region.width * region.height
            if 200 <= area <= 5000:
                score += 0.15
            elif 50 <= area <= 15000:
                score += 0.1

            return min(score, 1.0)
        except Exception as e:
            self.logger.error(f"Heuristic score calculation failed: {e}", exc_info=True)
            return 0.5

    # ------------------------------------------------------------------
    # OCR optimization
    # ------------------------------------------------------------------

    def _optimize_rois_for_ocr(self, regions: list[_ScoredRegion],
                               gray: np.ndarray) -> list[_ScoredRegion]:
        try:
            optimized: list[_ScoredRegion] = []
            for scored in regions:
                padded = self._add_ocr_padding(scored, gray.shape)
                aligned = self._align_to_text_baseline(padded, gray)
                if self._validate_ocr_region_quality(aligned.rectangle, gray):
                    optimized.append(aligned)
            return self._merge_nearby_text_regions(optimized)
        except Exception as e:
            self.logger.error(f"ROI OCR optimization failed: {e}", exc_info=True)
            return regions

    def _add_ocr_padding(self, scored: _ScoredRegion,
                         image_shape: tuple[int, int]) -> _ScoredRegion:
        try:
            r = scored.rectangle
            px = max(2, self.padding)
            py = max(2, self.padding)
            new_x = max(0, r.x - px)
            new_y = max(0, r.y - py)
            new_w = min(image_shape[1] - new_x, r.width + 2 * px)
            new_h = min(image_shape[0] - new_y, r.height + 2 * py)
            return _ScoredRegion(Rectangle(new_x, new_y, new_w, new_h), scored.confidence)
        except Exception:
            return scored

    def _align_to_text_baseline(self, scored: _ScoredRegion,
                                gray: np.ndarray) -> _ScoredRegion:
        try:
            r = scored.rectangle
            roi = gray[r.y:r.y + r.height, r.x:r.x + r.width]
            if roi.size == 0:
                return scored

            h_proj = np.sum(roi < 128, axis=1)
            if len(h_proj) == 0:
                return scored

            threshold = np.max(h_proj) * 0.3
            text_rows = np.where(h_proj >= threshold)[0]
            if len(text_rows) == 0:
                return scored

            top = text_rows[0]
            bottom = text_rows[-1]
            margin = max(1, (bottom - top) // 10)
            top = max(0, top - margin)
            bottom = min(roi.shape[0] - 1, bottom + margin)

            return _ScoredRegion(
                Rectangle(r.x, r.y + top, r.width, bottom - top + 1),
                scored.confidence,
            )
        except Exception:
            return scored

    def _validate_ocr_region_quality(self, region: Rectangle,
                                     gray: np.ndarray) -> bool:
        try:
            if region.width < 8 or region.height < 8:
                return False
            max_area = gray.shape[0] * gray.shape[1] * 0.5
            if region.width * region.height > max_area:
                return False

            roi = gray[region.y:region.y + region.height,
                       region.x:region.x + region.width]
            if roi.size == 0:
                return False
            if (np.max(roi) - np.min(roi)) < 30:
                return False
            if np.std(roi) < 10:
                return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Region merging
    # ------------------------------------------------------------------

    def _merge_nearby_text_regions(self, regions: list[_ScoredRegion]) -> list[_ScoredRegion]:
        try:
            if len(regions) <= 1:
                return regions

            merged_regions: list[_ScoredRegion] = []
            used: set = set()

            for i, s1 in enumerate(regions):
                if i in used:
                    continue
                merged_rect = Rectangle(s1.rectangle.x, s1.rectangle.y,
                                        s1.rectangle.width, s1.rectangle.height)
                merged_conf = s1.confidence
                count = 1
                merged_idx = {i}

                for j, s2 in enumerate(regions):
                    if j <= i or j in used:
                        continue
                    if self._should_merge_regions(merged_rect, s2.rectangle):
                        r2 = s2.rectangle
                        min_x = min(merged_rect.x, r2.x)
                        min_y = min(merged_rect.y, r2.y)
                        max_x = max(merged_rect.x + merged_rect.width, r2.x + r2.width)
                        max_y = max(merged_rect.y + merged_rect.height, r2.y + r2.height)
                        merged_rect = Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)
                        merged_conf += s2.confidence
                        count += 1
                        merged_idx.add(j)

                used.update(merged_idx)
                merged_regions.append(_ScoredRegion(merged_rect, merged_conf / count))

            return merged_regions
        except Exception as e:
            self.logger.error(f"Region merging failed: {e}", exc_info=True)
            return regions

    def _should_merge_regions(self, r1: Rectangle, r2: Rectangle) -> bool:
        try:
            h_gap = self._calculate_horizontal_gap(r1, r2)
            v_gap = self._calculate_vertical_gap(r1, r2)

            if h_gap <= self.merge_distance and v_gap <= self.merge_distance * 0.6:
                return True
            if v_gap <= self.merge_distance and h_gap <= self.merge_distance * 0.6:
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _calculate_horizontal_gap(r1: Rectangle, r2: Rectangle) -> float:
        if r1.x <= r2.x + r2.width and r2.x <= r1.x + r1.width:
            return 0.0
        if r1.x < r2.x:
            return r2.x - (r1.x + r1.width)
        return r1.x - (r2.x + r2.width)

    @staticmethod
    def _calculate_vertical_gap(r1: Rectangle, r2: Rectangle) -> float:
        if r1.y <= r2.y + r2.height and r2.y <= r1.y + r1.height:
            return 0.0
        if r1.y < r2.y:
            return r2.y - (r1.y + r1.height)
        return r1.y - (r2.y + r2.height)
