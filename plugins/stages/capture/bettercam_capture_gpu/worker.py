"""
BetterCam Capture Worker - Subprocess for GPU-accelerated screen capture.

Uses BetterCam (Desktop Duplication API) which supports both AMD and NVIDIA GPUs.
Drop-in replacement for DXCam with better AMD compatibility and higher performance.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.workflow.base.base_worker import BaseWorker

try:
    import bettercam
    import numpy as np
    BETTERCAM_AVAILABLE = True
except ImportError:
    BETTERCAM_AVAILABLE = False
    print("[BETTERCAM CAPTURE WORKER] Warning: bettercam not available, using fallback", file=sys.stderr)


class CaptureWorker(BaseWorker):
    """Worker for screen capture using BetterCam (AMD/NVIDIA compatible)."""

    def initialize(self, config: dict) -> bool:
        """Initialize capture system."""
        try:
            runtime_mode = config.get('runtime_mode', 'auto')

            if runtime_mode == 'cpu':
                self.log("CPU-only mode: BetterCam disabled (requires GPU/DirectX)")
                return False

            if not BETTERCAM_AVAILABLE:
                self.log("BetterCam not available - library not installed")
                return False

            # Create BetterCam camera instance
            color_mode = config.get('color_mode', 'BGR')
            max_buffer_len = config.get('max_buffer_len', 64)

            self.camera = bettercam.create(
                output_color=color_mode,
                max_buffer_len=max_buffer_len,
            )
            if self.camera is None:
                self.log("BetterCam initialization failed - GPU/DirectX not available")
                return False

            self.log(f"BetterCam capture initialized (runtime_mode: {runtime_mode}, color: {color_mode})")
            return True

        except Exception as e:
            self.log(f"Failed to initialize BetterCam capture: {e}")
            return False

    def process(self, data: dict) -> dict:
        """
        Capture a frame.

        Args:
            data: {'region': CaptureRegion or dict}

        Returns:
            {'frame': base64_encoded_frame, 'shape': [h, w, c]}
        """
        try:
            region = data.get('region')
            if not region:
                return {'error': 'No region specified'}

            if hasattr(region, 'x'):
                x, y, w, h = region.x, region.y, region.width, region.height
            else:
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('width', 800)
                h = region.get('height', 600)

            self.log(f"Capturing region: ({x}, {y}, {w}, {h})")

            # Capture frame with region
            frame = self.camera.grab(region=(x, y, x + w, y + h))

            if frame is None:
                self.log("Region capture returned None, trying full screen")
                frame = self.camera.grab()

                if frame is None:
                    return {'error': 'Failed to capture frame (both region and full screen)'}

                # Crop to region manually
                frame = frame[y:y + h, x:x + w]

            self.log(f"Captured frame shape: {frame.shape}")

            import base64
            frame_bytes = frame.tobytes()
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')

            return {
                'frame': frame_b64,
                'shape': list(frame.shape),
                'dtype': str(frame.dtype),
            }

        except Exception as e:
            return {'error': f'Capture failed: {e}'}

    def cleanup(self):
        """Clean up capture resources."""
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        self.log("BetterCam capture worker shutdown")


if __name__ == '__main__':
    worker = CaptureWorker(name="BetterCamCaptureWorker")
    worker.run()
