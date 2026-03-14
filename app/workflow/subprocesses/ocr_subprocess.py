"""
OCRSubprocess -- concrete BaseSubprocess for OCR stage isolation.

Bridges the pipeline's Frame/TextBlock objects and the worker's
base64/dict JSON protocol.  Serialises numpy frames on the way in
and deserialises worker result dicts back into TextBlock instances
on the way out.
"""

import base64
import logging
from typing import Any

from app.workflow.base.base_subprocess import BaseSubprocess

try:
    from app.models import Rectangle, TextBlock
except ImportError:
    from models import Rectangle, TextBlock


logger = logging.getLogger(__name__)


class OCRSubprocess(BaseSubprocess):
    """Subprocess wrapper specialised for OCR worker scripts."""

    def __init__(self, plugin_name: str, worker_script: str) -> None:
        super().__init__(f"OCR-{plugin_name}", worker_script)
        self._plugin_name = plugin_name

    # -- BaseSubprocess hooks ------------------------------------------

    def _prepare_message(self, data: Any) -> dict:
        """Serialise a pipeline data dict for the worker.

        Expects *data* to contain a ``frame`` key whose value is either
        a ``Frame`` object (with a ``.data`` ndarray attribute) or a raw
        numpy array.
        """
        frame = data.get("frame")
        if frame is None:
            return {"frame": "", "shape": [0, 0, 0], "dtype": "uint8"}

        frame_array = frame.data if hasattr(frame, "data") else frame

        return {
            "frame": base64.b64encode(frame_array.tobytes()).decode(),
            "shape": list(frame_array.shape),
            "dtype": str(frame_array.dtype),
            "language": data.get("language", "en"),
        }

    def _parse_result(self, result: dict) -> dict:
        """Deserialise a worker result dict into ``TextBlock`` objects."""
        blocks_data = result.get("data", {}).get("text_blocks", [])
        text_blocks: list[TextBlock] = []
        for b in blocks_data:
            bbox = b.get("bbox", [0, 0, 0, 0])
            text_blocks.append(
                TextBlock(
                    text=b.get("text", ""),
                    position=Rectangle(
                        x=int(bbox[0]),
                        y=int(bbox[1]),
                        width=int(bbox[2]),
                        height=int(bbox[3]),
                    ),
                    confidence=b.get("confidence", 0.5),
                )
            )
        return {"text_blocks": text_blocks}
