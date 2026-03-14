"""
PIL Screenshot Capture

Lightweight screenshot capture using PIL ImageGrab.
Used as the CPU fallback when GPU capture backends (bettercam) are unavailable.
"""

import logging
import numpy as np

from PIL import ImageGrab

logger = logging.getLogger(__name__)


def capture_screenshot(region_data: dict) -> np.ndarray | None:
    """
    Capture a screenshot of the specified region.

    Args:
        region_data: Dict with keys ``x``, ``y``, ``width``, ``height``.

    Returns:
        BGR numpy array, or *None* on failure.
    """
    try:
        x = region_data['x']
        y = region_data['y']
        w = region_data['width']
        h = region_data['height']

        screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        frame = np.array(screenshot)

        # PIL returns RGB; convert to BGR for consistency with GPU backends
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]

        return frame

    except Exception as e:
        logger.error("PIL screenshot capture failed: %s", e)
        return None
