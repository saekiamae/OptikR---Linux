"""
Shared frame-decoding utilities for OCR plugin workers.

Every OCR subprocess worker receives frames as base64-encoded numpy arrays.
This module extracts the repetitive decode-and-convert boilerplate so each
worker only needs a one-liner.
"""
import base64
from typing import Any

import numpy as np


def decode_frame(data: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    """Decode a base64-encoded frame from the subprocess protocol.

    Args:
        data: Message dict with keys ``frame`` (base64 str),
              ``shape`` (list[int]), and ``dtype`` (str).

    Returns:
        ``(frame, None)`` on success, or ``(None, error_message)`` on failure.
    """
    frame_b64 = data.get("frame")
    if not frame_b64:
        return None, "No frame provided"

    frame_bytes = base64.b64decode(frame_b64)
    shape = data.get("shape", [600, 800, 3])
    dtype = data.get("dtype", "uint8")
    frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
    return frame, None


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB (in-place view, no copy).

    Safe to call on grayscale frames — they are returned unchanged.
    """
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame[:, :, ::-1]
    return frame
