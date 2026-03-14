"""
Capture Layer

Component responsible for capturing screen content using plugin-based subprocess isolation.
Uses DirectX (DXCam), BetterCam, or Screenshot capture methods via isolated worker processes.
"""

import logging

logger = logging.getLogger(__name__)

# Plugin-based capture layer (primary system)
try:
    from .plugin_capture_layer import PluginCaptureLayer
except ImportError as e:
    logger.warning("Could not import PluginCaptureLayer: %s", e)
    PluginCaptureLayer = None

# Plugin manager
try:
    from .capture_plugin_manager import CapturePluginManager
except ImportError as e:
    logger.warning("Could not import CapturePluginManager: %s", e)
    CapturePluginManager = None

# DirectX capture components
try:
    from .directx_capture import DirectXCaptureLayer, DirectXDesktopCapture, DirectXCaptureError, CaptureStatus
except ImportError as e:
    DirectXCaptureLayer = None
    DirectXDesktopCapture = None
    DirectXCaptureError = Exception
    CaptureStatus = None

# Multi-monitor support
try:
    from .multi_monitor_support import MultiMonitorManager, MonitorInfo, MonitorOrientation
except ImportError as e:
    logger.warning("Could not import multi-monitor support: %s", e)
    MultiMonitorManager = None

# PIL screenshot (CPU fallback)
from .pil_screenshot import capture_screenshot

__all__ = [
    # Plugin-based system (primary)
    'PluginCaptureLayer',
    'CapturePluginManager',
    
    # DirectX components
    'DirectXCaptureLayer',
    'DirectXDesktopCapture', 
    'DirectXCaptureError',
    'CaptureStatus',
    
    # Multi-monitor support
    'MultiMonitorManager',
    'MonitorInfo',
    'MonitorOrientation',
    
    # PIL screenshot
    'capture_screenshot',
]
