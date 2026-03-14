"""
Plugin-Based Capture Layer

Adapts the CapturePluginManager to the ICaptureLayer interface used by the
rest of the pipeline.
"""

import logging
import time
import numpy as np
from typing import Any

try:
    from ..models import Frame, CaptureRegion, PerformanceProfile
    from ..interfaces import ICaptureLayer, CaptureSource
    from .capture_plugin_manager import CapturePluginManager
except ImportError:
    from app.models import Frame, CaptureRegion, PerformanceProfile
    from app.interfaces import ICaptureLayer, CaptureSource
    from app.capture.capture_plugin_manager import CapturePluginManager


class PluginCaptureLayer(ICaptureLayer):
    """Plugin-based capture layer.

    Thin adapter between the ``ICaptureLayer`` interface expected by the
    pipeline and the ``CapturePluginManager`` that owns the actual backends.
    """

    def __init__(self, logger: logging.Logger | None = None, config_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager

        # Runtime mode
        self.runtime_mode = 'auto'
        if self.config_manager:
            self.runtime_mode = self.config_manager.get_setting(
                'performance.runtime_mode', 'auto',
            )

        # Plugin manager
        self.plugin_manager = CapturePluginManager()
        plugins = self.plugin_manager.discover_plugins()

        if plugins:
            plugin_names = [p.name for p in plugins]
            self.logger.info("Discovered %d capture plugin(s): %s", len(plugins), plugin_names)
        else:
            self.logger.warning("No capture plugins found")

        # State
        self._frame_rate = 30
        self._performance_profile = PerformanceProfile.NORMAL

        # Statistics
        self._stats: dict[str, Any] = {
            'total_frames': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'average_capture_time': 0.0,
        }

    # ------------------------------------------------------------------
    # ICaptureLayer implementation
    # ------------------------------------------------------------------

    def capture_frame(self, source: CaptureSource, region: CaptureRegion) -> Frame:
        """Capture a frame and return a ``Frame`` object.

        Raises ``RuntimeError`` if all backends fail.
        """
        start_time = time.perf_counter()

        region_data = {
            'x': region.rectangle.x,
            'y': region.rectangle.y,
            'width': region.rectangle.width,
            'height': region.rectangle.height,
            'monitor_id': region.monitor_id,
        }

        frame_data: np.ndarray | None = self.plugin_manager.capture_frame(region_data)

        if frame_data is None:
            self._stats['failed_captures'] += 1
            raise RuntimeError("All capture backends failed")

        frame = Frame(
            data=frame_data,
            timestamp=time.time(),
            source_region=region,
            metadata={'source': source.value if hasattr(source, 'value') else str(source)},
        )

        capture_time = time.perf_counter() - start_time
        self._update_stats(capture_time)
        return frame

    def set_capture_mode(self, mode: str) -> bool:
        plugin_map = {
            'directx': 'bettercam_capture_gpu',
            'dxcam': 'bettercam_capture_gpu',
            'bettercam': 'bettercam_capture_gpu',
            'amd': 'bettercam_capture_gpu',
            'screenshot': 'screenshot_capture_cpu',
            'auto': 'bettercam_capture_gpu',
        }

        plugin_name = plugin_map.get(mode.lower())
        if not plugin_name:
            self.logger.error("Unknown capture mode: %s", mode)
            return False

        if self.plugin_manager.set_active_plugin(plugin_name):
            self.logger.info("Capture mode set to %s (plugin: %s)", mode, plugin_name)
            return True

        # Auto-mode fallback chain: bettercam → screenshot
        if mode.lower() == 'auto':
            for fallback in ('screenshot_capture_cpu',):
                if self.plugin_manager.set_active_plugin(fallback):
                    self.logger.info("Fell back to %s", fallback)
                    return True

        return False

    def get_supported_modes(self) -> list[str]:
        plugins = self.plugin_manager.get_available_plugins(self.config_manager)
        modes: list[str] = []
        if 'bettercam_capture_gpu' in plugins:
            modes.extend(['directx', 'bettercam', 'amd', 'auto'])
        if 'screenshot_capture_cpu' in plugins:
            modes.append('screenshot')
        return modes

    def configure_capture_rate(self, fps: int) -> bool:
        if not 1 <= fps <= 30:
            self.logger.error("Invalid frame rate: %d (must be 1-30)", fps)
            return False
        self._frame_rate = fps
        return True

    def is_available(self) -> bool:
        return len(self.plugin_manager.get_available_plugins(self.config_manager)) > 0

    def set_performance_profile(self, profile: PerformanceProfile) -> None:
        self._performance_profile = profile
        caps = {PerformanceProfile.LOW: 15, PerformanceProfile.NORMAL: 30, PerformanceProfile.HIGH: 30}
        self.configure_capture_rate(min(self._frame_rate, caps.get(profile, 30)))

    def get_capture_statistics(self) -> dict[str, Any]:
        stats = self._stats.copy()
        stats['active_plugin'] = self.plugin_manager.get_active_plugin()
        stats['available_plugins'] = self.plugin_manager.get_available_plugins(self.config_manager)
        stats['frame_rate'] = self._frame_rate
        stats['performance_profile'] = self._performance_profile.value
        return stats

    def cleanup(self) -> None:
        try:
            self.plugin_manager.cleanup()
        except Exception as e:
            self.logger.error("Cleanup error: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_stats(self, capture_time: float) -> None:
        self._stats['total_frames'] += 1
        self._stats['successful_captures'] += 1
        n = self._stats['total_frames']
        avg = self._stats['average_capture_time']
        self._stats['average_capture_time'] = (avg * (n - 1) + capture_time) / n
