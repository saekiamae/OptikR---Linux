"""
DirectX Desktop Duplication API Capture Implementation

High-performance screen capture using DirectX Desktop Duplication API with fallback support.
"""

import time
import threading
from typing import Callable, Any
import numpy as np
from enum import Enum
import logging

try:
    from ..models import Frame, CaptureRegion, Rectangle, CaptureMode, PerformanceProfile
    from ..interfaces import ICaptureLayer, CaptureSource
    from .multi_monitor_support import MultiMonitorManager, MonitorInfo
except ImportError:
    from app.models import Frame, CaptureRegion, Rectangle, CaptureMode, PerformanceProfile
    from app.interfaces import ICaptureLayer, CaptureSource
    from app.capture.multi_monitor_support import MultiMonitorManager, MonitorInfo

logger = logging.getLogger(__name__)
logger.debug("bettercam import deferred (lazy import to avoid GPU conflicts)")
bettercam = None
BETTERCAM_AVAILABLE = None  # Will be checked on first use


class CaptureStatus(Enum):
    """Capture system status enumeration."""
    READY = "ready"
    CAPTURING = "capturing"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class DirectXCaptureError(Exception):
    """DirectX capture specific exceptions."""
    pass


class DirectXDesktopCapture:
    """
    DirectX Desktop Duplication API capture implementation.
    
    Provides high-performance screen capture using DirectX Desktop Duplication API
    with automatic fallback to GDI+ screenshot capture when DirectX is unavailable.
    """
    
    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize DirectX capture system.
        
        Args:
            logger: Optional logger for debugging and monitoring
        """
        self.logger = logger or logging.getLogger(__name__)
        self._is_initialized = False
        self._capture_active = False
        self._current_region: CaptureRegion | None = None
        self._frame_rate = 30
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_callbacks: list[Callable[[Frame], None]] = []
        self._callbacks_lock = threading.Lock()
        self._fps_window_start: float = 0
        self._performance_profile = PerformanceProfile.NORMAL
        self._use_fullscreen = False  # Flag for full screen capture mode
        self._gpu_capture_failed = False  # Flag to prevent repeated BetterCam initialization attempts
        self.config_manager = None
        
        # Multi-monitor support
        logger.debug("Creating MultiMonitorManager...")
        self._monitor_manager = MultiMonitorManager(logger)
        logger.debug("MultiMonitorManager created")
        
        # DirectX/DXGI objects (will be initialized when needed)
        self._dxgi_factory = None
        self._dxgi_adapter = None
        self._dxgi_output = None
        self._dxgi_duplication = None
        self._d3d_device = None
        self._d3d_context = None
        
        # Per-monitor DirectX objects for multi-monitor support
        self._monitor_duplications: dict[int, Any] = {}
        
        # Capture statistics
        self._stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'last_capture_time': 0,
            'average_fps': 0,
            'capture_errors': 0,
            'active_monitors': 0
        }
        
        # Initialize DirectX capture
        logger.debug("Initializing DirectX capture...")
        self._initialize_directx_capture()
        logger.debug("DirectX capture initialization complete")
    
    def _initialize_directx_capture(self) -> bool:
        """
        Initialize DirectX Desktop Duplication API using bettercam.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        global bettercam, BETTERCAM_AVAILABLE
        
        try:
            self.logger.info("Initializing DirectX Desktop Duplication API via bettercam")
            
            if BETTERCAM_AVAILABLE is None:
                try:
                    import bettercam as bettercam_module
                    bettercam = bettercam_module
                    BETTERCAM_AVAILABLE = True
                    self.logger.info("bettercam imported successfully")
                except ImportError as e:
                    self.logger.info("bettercam not available: %s", e)
                    BETTERCAM_AVAILABLE = False
                    return False
                except Exception as e:
                    self.logger.warning("Error importing bettercam: %s", e)
                    BETTERCAM_AVAILABLE = False
                    return False
            
            if not BETTERCAM_AVAILABLE or bettercam is None:
                self.logger.warning("bettercam not available, DirectX capture will not work")
                return False
            
            self.logger.info("bettercam library available")
            
            if not self._enumerate_display_adapters():
                return False
            
            self._is_initialized = True
            self._bettercam_camera = None  # Will be created on first capture
            self.logger.info("DirectX Desktop Duplication API initialized successfully via bettercam")
            return True
            
        except Exception as e:
            self.logger.error(f"DirectX initialization failed: {e}")
            return False
    
    def _enumerate_display_adapters(self) -> bool:
        """Enumerate display adapters and outputs for multi-monitor support."""
        try:
            self.logger.debug("Enumerating display adapters for multi-monitor support")
            
            # Get monitor information from multi-monitor manager
            monitors = self._monitor_manager.get_all_monitors()
            self._stats['active_monitors'] = len(monitors)
            
            self.logger.info(f"Found {len(monitors)} monitors for DirectX capture")
            for monitor_id, monitor in monitors.items():
                self.logger.debug(f"Monitor {monitor_id}: {monitor.display_name} "
                                f"({monitor.bounds.width}x{monitor.bounds.height})")
            
            # Simplified implementation - would enumerate actual adapters per monitor
            self._dxgi_adapter = True  # Placeholder
            self._dxgi_output = True   # Placeholder
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enumerate display adapters: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if DirectX capture is available.
        
        Returns:
            bool: True if DirectX capture is available and initialized
        """
        return self._is_initialized
    
    def get_supported_modes(self) -> list[str]:
        """
        Get list of supported capture modes.
        
        Returns:
            list[str]: List of supported capture mode names
        """
        modes = []
        if self._is_initialized:
            modes.append(CaptureMode.DIRECTX.value)
            modes.append(CaptureMode.DESKTOP_DUPLICATION.value)
        modes.append(CaptureMode.SCREENSHOT.value)  # Always available as fallback
        return modes
    
    def set_capture_region(self, region: CaptureRegion) -> bool:
        """
        Set the capture region with multi-monitor validation.
        
        Args:
            region: CaptureRegion defining the area to capture
            
        Returns:
            bool: True if region set successfully
        """
        try:
            # Validate monitor ID
            monitor_info = self._monitor_manager.get_monitor_info(region.monitor_id)
            if not monitor_info:
                self.logger.error(f"Invalid monitor ID: {region.monitor_id}")
                return False
            
            # Validate region is within monitor bounds
            monitor_bounds = monitor_info.bounds
            region_rect = region.rectangle
            
            if (region_rect.x < monitor_bounds.x or 
                region_rect.y < monitor_bounds.y or
                region_rect.x + region_rect.width > monitor_bounds.x + monitor_bounds.width or
                region_rect.y + region_rect.height > monitor_bounds.y + monitor_bounds.height):
                
                self.logger.warning(f"Capture region extends beyond monitor {region.monitor_id} bounds, clipping")
                
                # Clip region to monitor bounds
                clipped_region = CaptureRegion(
                    rectangle=Rectangle(
                        x=max(region_rect.x, monitor_bounds.x),
                        y=max(region_rect.y, monitor_bounds.y),
                        width=min(region_rect.width, 
                                monitor_bounds.x + monitor_bounds.width - max(region_rect.x, monitor_bounds.x)),
                        height=min(region_rect.height,
                                 monitor_bounds.y + monitor_bounds.height - max(region_rect.y, monitor_bounds.y))
                    ),
                    monitor_id=region.monitor_id,
                    window_handle=region.window_handle
                )
                self._current_region = clipped_region
            else:
                self._current_region = region
            
            # Release existing BetterCam camera when region changes
            if hasattr(self, '_bettercam_camera') and self._bettercam_camera is not None:
                try:
                    self.logger.info("Releasing old BetterCam camera due to region change")
                    self._bettercam_camera.release()
                    del self._bettercam_camera
                    self._bettercam_camera = None
                    if hasattr(self, '_region_logged'):
                        del self._region_logged
                    if hasattr(self, '_use_fullscreen'):
                        self._use_fullscreen = False
                    
                    self.logger.info("BetterCam camera released successfully")
                except Exception as e:
                    self.logger.warning(f"Error releasing BetterCam camera: {e}")
                    self._bettercam_camera = None
                    self._gpu_capture_failed = True
            
            self.logger.info(f"Capture region set on monitor {region.monitor_id} ({monitor_info.display_name}): "
                           f"{self._current_region.rectangle.width}x{self._current_region.rectangle.height} "
                           f"at ({self._current_region.rectangle.x}, {self._current_region.rectangle.y})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set capture region: {e}")
            return False
    
    def set_frame_rate(self, fps: int) -> bool:
        """
        Set the target frame rate for capture.
        
        Args:
            fps: Target frames per second (1-120)
            
        Returns:
            bool: True if frame rate set successfully
        """
        if not 1 <= fps <= 120:
            self.logger.error(f"Invalid frame rate: {fps}. Must be between 1 and 120")
            return False
        
        self._frame_rate = fps
        self.logger.info(f"Frame rate set to {fps} FPS")
        return True
    
    def set_performance_profile(self, profile: PerformanceProfile) -> None:
        """
        Set performance profile for adaptive optimization.
        
        Args:
            profile: Performance profile (LOW, NORMAL, HIGH)
        """
        self._performance_profile = profile
        
        # Adjust frame rate based on profile
        if profile == PerformanceProfile.LOW:
            self._frame_rate = min(self._frame_rate, 15)
        elif profile == PerformanceProfile.NORMAL:
            self._frame_rate = min(self._frame_rate, 30)
        elif profile == PerformanceProfile.HIGH:
            self._frame_rate = min(self._frame_rate, 60)
        
        self.logger.info(f"Performance profile set to {profile.value}, frame rate: {self._frame_rate}")
    
    def add_frame_callback(self, callback: Callable[[Frame], None]) -> None:
        """
        Add callback function to receive captured frames.
        
        Args:
            callback: Function to call with each captured frame
        """
        with self._callbacks_lock:
            self._frame_callbacks.append(callback)
            self.logger.debug(f"Frame callback added. Total callbacks: {len(self._frame_callbacks)}")
    
    def remove_frame_callback(self, callback: Callable[[Frame], None]) -> None:
        """
        Remove frame callback.
        
        Args:
            callback: Callback function to remove
        """
        with self._callbacks_lock:
            if callback in self._frame_callbacks:
                self._frame_callbacks.remove(callback)
                self.logger.debug(f"Frame callback removed. Total callbacks: {len(self._frame_callbacks)}")
    
    def start_capture(self) -> bool:
        """
        Start continuous frame capture.
        
        Returns:
            bool: True if capture started successfully
        """
        if self._capture_active:
            self.logger.warning("Capture already active")
            return True
        
        if not self._current_region:
            self.logger.error("No capture region set")
            return False
        
        try:
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._capture_active = True
            
            self.logger.info("Frame capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """
        Stop continuous frame capture.
        
        Returns:
            bool: True if capture stopped successfully
        """
        if not self._capture_active:
            return True
        
        try:
            self._stop_event.set()
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            
            self._capture_active = False
            self.logger.info("Frame capture stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop capture: {e}")
            return False
    
    def capture_single_frame(self) -> Frame | None:
        """
        Capture a single frame.
        
        Returns:
            Frame | None: Captured frame or None if capture failed
        """
        if not self._current_region:
            self.logger.error("No capture region set")
            return None
        
        try:
            if self._is_initialized:
                return self._capture_directx_frame()
            else:
                return self._capture_fallback_frame()
                
        except Exception as e:
            self.logger.error(f"Single frame capture failed: {e}")
            return None
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self._frame_rate
        last_capture_time = 0
        
        self.logger.debug(f"Capture loop started with {self._frame_rate} FPS target")
        
        while not self._stop_event.is_set():
            current_time = time.perf_counter()
            
            # Frame rate limiting
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            try:
                frame = self.capture_single_frame()
                if frame:
                    self._update_capture_stats(current_time)
                    
                    # Notify all callbacks (snapshot under lock for thread safety)
                    with self._callbacks_lock:
                        callbacks_snapshot = list(self._frame_callbacks)
                    for callback in callbacks_snapshot:
                        try:
                            callback(frame)
                        except Exception as e:
                            self.logger.error(f"Frame callback error: {e}")
                else:
                    self._stats['frames_dropped'] += 1
                
                last_capture_time = current_time
                
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                self._stats['capture_errors'] += 1
                time.sleep(0.1)  # Brief pause on error
    
    def _capture_directx_frame(self) -> Frame | None:
        """
        Capture frame using DirectX Desktop Duplication API via bettercam.
        
        Returns:
            Frame | None: Captured frame or None if failed
        """
        try:
            if hasattr(self, '_gpu_capture_failed') and self._gpu_capture_failed:
                return self._capture_fallback_frame()
            
            if not hasattr(self, '_bettercam_camera') or self._bettercam_camera is None:
                try:
                    monitor_id = self._current_region.monitor_id if self._current_region else 0
                    
                    self.logger.info(f"Creating BetterCam camera for monitor {monitor_id}")
                    
                    self._bettercam_camera = bettercam.create(output_idx=monitor_id)
                    
                    if self._bettercam_camera is None:
                        self.logger.error("Failed to create BetterCam camera - returned None")
                        return self._capture_fallback_frame()
                    
                    self.logger.info(f"BetterCam camera created (will use direct grab())")
                    
                    time.sleep(0.1)
                    
                    region = self._current_region.rectangle
                    test_region = (region.x, region.y, region.x + region.width, region.y + region.height)
                    
                    self.logger.info(f"Testing BetterCam with region: {test_region}")
                    
                    test_frame = None
                    for attempt in range(5):
                        test_frame = self._bettercam_camera.grab(region=test_region)
                        if test_frame is not None:
                            self.logger.info(f"BetterCam test successful on attempt {attempt + 1}")
                            break
                        time.sleep(0.1)
                    
                    if test_frame is None:
                        self.logger.warning("BetterCam region grab returned None after 5 attempts, trying full screen")
                        for attempt in range(5):
                            test_frame = self._bettercam_camera.grab()
                            if test_frame is not None:
                                self.logger.info(f"BetterCam full screen test successful on attempt {attempt + 1}")
                                break
                            time.sleep(0.1)
                        
                        if test_frame is None:
                            self.logger.error("BetterCam camera not working (both region and full screen failed)")
                            self._bettercam_camera.release()
                            self._bettercam_camera = None
                            self._gpu_capture_failed = True
                            return self._capture_fallback_frame()
                        else:
                            self.logger.warning("BetterCam works with full screen but not with region - will use full screen + crop")
                            self._use_fullscreen = True
                    else:
                        self._use_fullscreen = False
                    
                    self.logger.info(f"BetterCam camera ready for monitor {monitor_id} (FPS: {self._frame_rate}, fullscreen_mode: {self._use_fullscreen})")
                except Exception as e:
                    self.logger.error(f"Failed to create BetterCam camera: {e}")
                    return self._capture_fallback_frame()
            
            region = self._current_region.rectangle
            
            if not hasattr(self, '_region_logged'):
                self.logger.info(f"Capture region: x={region.x}, y={region.y}, w={region.width}, h={region.height}")
                self._region_logged = True
            
            # bettercam region format is (left, top, right, bottom)
            try:
                if hasattr(self, '_use_fullscreen') and self._use_fullscreen:
                    frame_data = self._bettercam_camera.grab()
                    if frame_data is not None:
                        frame_data = frame_data[region.y:region.y+region.height, 
                                              region.x:region.x+region.width]
                else:
                    frame_data = self._bettercam_camera.grab(region=(region.x, region.y, 
                                                                 region.x + region.width, 
                                                                 region.y + region.height))
            except Exception as grab_error:
                self.logger.warning(f"BetterCam grab failed, attempting to recreate camera: {type(grab_error).__name__}")
                try:
                    if hasattr(self, '_bettercam_camera') and self._bettercam_camera is not None:
                        try:
                            self._bettercam_camera.release()
                        except Exception:
                            pass
                    
                    monitor_id = self._current_region.monitor_id if self._current_region else 0
                    self._bettercam_camera = bettercam.create(output_idx=monitor_id)
                    
                    if self._bettercam_camera is None:
                        self.logger.warning("Failed to recreate BetterCam camera, using screenshot fallback")
                        return self._capture_fallback_frame()
                    
                    time.sleep(0.1)
                    
                    frame_data = self._bettercam_camera.grab(region=(region.x, region.y, 
                                                                 region.x + region.width, 
                                                                 region.y + region.height))
                except Exception as recreate_error:
                    self.logger.warning(f"Camera recreation failed: {type(recreate_error).__name__}, using screenshot fallback")
                    return self._capture_fallback_frame()
            
            if frame_data is None:
                retry_count = 0
                max_retries = self.config_manager.get_setting('retry.max_retries', 3) if self.config_manager else 3
                
                while frame_data is None and retry_count < max_retries:
                    retry_count += 1
                    time.sleep(0.01)
                    
                    try:
                        frame_data = self._bettercam_camera.grab()
                        
                        if frame_data is not None:
                            frame_data = frame_data[region.y:region.y+region.height, 
                                                  region.x:region.x+region.width]
                            break
                    except Exception as e:
                        if retry_count == max_retries:
                            self.logger.warning(f"BetterCam grab failed after {max_retries} retries: {e}")
                
                if frame_data is None:
                    if not hasattr(self, '_fallback_warning_count'):
                        self._fallback_warning_count = 0
                    
                    self._fallback_warning_count += 1
                    if self._fallback_warning_count % 10 == 1:
                        self.logger.warning(f"BetterCam returned None after {max_retries} retries (count: {self._fallback_warning_count}), using screenshot fallback")
                    
                    return self._capture_fallback_frame()
            
            frame = Frame(
                data=frame_data,
                timestamp=time.time(),
                source_region=self._current_region,
                metadata={
                    'capture_method': 'directx_bettercam',
                    'frame_rate': self._frame_rate,
                    'performance_profile': self._performance_profile.value
                }
            )
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"DirectX capture unavailable, using screenshot fallback (Reason: {type(e).__name__})")
            return self._capture_fallback_frame()
    
    def _capture_fallback_frame(self) -> Frame | None:
        """
        Fallback frame capture using GDI+ screenshot.
        
        Returns:
            Frame | None: Captured frame or None if failed
        """
        try:
            import win32gui
            import win32ui
            import win32con
            from PIL import Image
            
            # Get device context
            hwnd = win32gui.GetDesktopWindow()
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            region = self._current_region.rectangle
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, region.width, region.height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy screen content
            saveDC.BitBlt((0, 0), (region.width, region.height), mfcDC, 
                         (region.x, region.y), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            # Convert to PIL Image then numpy
            img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), 
                                 bmpstr, 'raw', 'BGRX', 0, 1)
            frame_data = np.array(img)
            
            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            frame = Frame(
                data=frame_data,
                timestamp=time.time(),
                source_region=self._current_region,
                metadata={
                    'capture_method': 'screenshot_fallback',
                    'frame_rate': self._frame_rate,
                    'performance_profile': self._performance_profile.value
                }
            )
            
            return frame
            
        except ImportError:
            self.logger.error("Win32 libraries not available for fallback capture")
            return None
        except Exception as e:
            self.logger.error(f"Fallback frame capture failed: {e}")
            return None
    
    def _update_capture_stats(self, current_time: float) -> None:
        """Update capture statistics."""
        self._stats['frames_captured'] += 1
        
        # Calculate average FPS over last 30 frames using window start time
        if self._stats['frames_captured'] % 30 == 0:
            if self._fps_window_start > 0:
                time_diff = current_time - self._fps_window_start
                if time_diff > 0:
                    self._stats['average_fps'] = 30.0 / time_diff
            self._fps_window_start = current_time
        
        self._stats['last_capture_time'] = current_time
    
    def get_capture_stats(self) -> dict[str, Any]:
        """
        Get capture statistics.
        
        Returns:
            dict[str, Any]: Dictionary containing capture statistics
        """
        return self._stats.copy()
    
    def get_status(self) -> CaptureStatus:
        """
        Get current capture status.
        
        Returns:
            CaptureStatus: Current status of the capture system
        """
        if not self._is_initialized:
            return CaptureStatus.UNAVAILABLE
        elif self._capture_active:
            return CaptureStatus.CAPTURING
        elif self._stats['capture_errors'] > 10:
            return CaptureStatus.ERROR
        else:
            return CaptureStatus.READY
    
    def get_monitor_manager(self) -> MultiMonitorManager:
        """
        Get the multi-monitor manager.
        
        Returns:
            MultiMonitorManager: Monitor manager instance
        """
        return self._monitor_manager
    
    def get_available_monitors(self) -> dict[int, MonitorInfo]:
        """
        Get information about all available monitors.
        
        Returns:
            dict[int, MonitorInfo]: Dictionary of monitor ID to monitor info
        """
        return self._monitor_manager.get_all_monitors()
    
    def get_primary_monitor_id(self) -> int | None:
        """
        Get the ID of the primary monitor.
        
        Returns:
            int | None: Primary monitor ID or None if not found
        """
        return self._monitor_manager.get_primary_monitor_id()
    
    def create_full_screen_region(self, monitor_id: int | None = None) -> CaptureRegion | None:
        """
        Create a capture region for full screen capture.
        
        Args:
            monitor_id: Optional monitor ID, uses primary if None
            
        Returns:
            CaptureRegion | None: Full screen capture region or None if failed
        """
        if monitor_id is None:
            # Use virtual screen (all monitors)
            return self._monitor_manager.create_full_screen_region()
        else:
            # Use specific monitor
            return self._monitor_manager.create_capture_region_for_monitor(monitor_id)
    
    def create_monitor_region(self, monitor_id: int, 
                            custom_region: Rectangle | None = None) -> CaptureRegion | None:
        """
        Create a capture region for a specific monitor.
        
        Args:
            monitor_id: Monitor ID
            custom_region: Optional custom region within the monitor
            
        Returns:
            CaptureRegion | None: Capture region or None if failed
        """
        return self._monitor_manager.create_capture_region_for_monitor(monitor_id, custom_region)
    
    def refresh_monitors(self) -> bool:
        """
        Refresh monitor information (useful when displays are added/removed).
        
        Returns:
            bool: True if refresh successful
        """
        try:
            success = self._monitor_manager.refresh_monitor_info()
            if success:
                self._stats['active_monitors'] = self._monitor_manager.get_monitor_count()
                self.logger.info("Monitor information refreshed")
            return success
        except Exception as e:
            self.logger.error(f"Failed to refresh monitors: {e}")
            return False
    
    def get_monitor_summary(self) -> dict[str, Any]:
        """
        Get a summary of monitor configuration.
        
        Returns:
            dict[str, Any]: Monitor summary
        """
        return self._monitor_manager.get_monitor_summary()
    
    def cleanup(self) -> None:
        """Clean up DirectX resources."""
        try:
            self.stop_capture()
            
            if hasattr(self, '_bettercam_camera') and self._bettercam_camera is not None:
                try:
                    self._bettercam_camera.release()
                    del self._bettercam_camera
                    self._bettercam_camera = None
                    self.logger.debug("Released bettercam camera")
                except Exception as e:
                    self.logger.error(f"Error releasing bettercam camera: {e}")
            
            # Release per-monitor DirectX resources
            for monitor_id, duplication in self._monitor_duplications.items():
                try:
                    self.logger.debug(f"Released DirectX resources for monitor {monitor_id}")
                except Exception as e:
                    self.logger.error(f"Error releasing resources for monitor {monitor_id}: {e}")
            
            self._monitor_duplications.clear()
            
            self._is_initialized = False
            self.logger.info("DirectX capture resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


class DirectXCaptureLayer(ICaptureLayer):
    """
    DirectX capture layer implementation conforming to ICaptureLayer interface.
    
    Provides high-level interface for DirectX-based screen capture with automatic
    fallback support and adaptive frame rate control.
    """
    
    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize DirectX capture layer.
        
        Args:
            logger: Optional logger for debugging and monitoring
        """
        self.logger = logger or logging.getLogger(__name__)
        self._capture_engine = DirectXDesktopCapture(logger)
        self._current_mode = CaptureMode.DIRECTX if self._capture_engine.is_available() else CaptureMode.SCREENSHOT
        
        self.logger.info(f"DirectX Capture Layer initialized with mode: {self._current_mode.value}")
    
    def capture_frame(self, source: CaptureSource, region: CaptureRegion) -> Frame:
        """
        Capture a frame from the specified source and region with multi-monitor support.
        
        Args:
            source: Capture source type (FULL_SCREEN, WINDOW, CUSTOM_REGION)
            region: Region to capture
            
        Returns:
            Frame: Captured frame data
            
        Raises:
            DirectXCaptureError: If capture fails
        """
        try:
            # Set capture region based on source type
            if source == CaptureSource.FULL_SCREEN:
                # Create full screen region for specified monitor or all monitors
                full_screen_region = self._capture_engine.create_full_screen_region(region.monitor_id)
                if not full_screen_region:
                    raise DirectXCaptureError(f"Failed to create full screen region for monitor {region.monitor_id}")
                self._capture_engine.set_capture_region(full_screen_region)
            elif source == CaptureSource.WINDOW:
                # For window capture, use the region as-is but validate monitor
                monitor_info = self._capture_engine.get_monitor_manager().get_monitor_info(region.monitor_id)
                if not monitor_info:
                    raise DirectXCaptureError(f"Invalid monitor ID: {region.monitor_id}")
                self._capture_engine.set_capture_region(region)
            else:  # CUSTOM_REGION
                # Validate custom region against monitor bounds
                monitor_info = self._capture_engine.get_monitor_manager().get_monitor_info(region.monitor_id)
                if not monitor_info:
                    raise DirectXCaptureError(f"Invalid monitor ID: {region.monitor_id}")
                self._capture_engine.set_capture_region(region)
            
            frame = self._capture_engine.capture_single_frame()
            if frame is None:
                raise DirectXCaptureError("Failed to capture frame")
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            raise DirectXCaptureError(f"Capture failed: {e}")
    
    def set_capture_mode(self, mode: str) -> bool:
        """
        Set the capture mode.
        
        Args:
            mode: Capture mode string
            
        Returns:
            bool: True if mode set successfully
        """
        try:
            capture_mode = CaptureMode(mode)
            
            if capture_mode == CaptureMode.DIRECTX and not self._capture_engine.is_available():
                self.logger.warning("DirectX mode requested but not available, using fallback")
                return False
            
            self._current_mode = capture_mode
            self.logger.info(f"Capture mode set to: {mode}")
            return True
            
        except ValueError:
            self.logger.error(f"Invalid capture mode: {mode}")
            return False
    
    def get_supported_modes(self) -> list[str]:
        """
        Get list of supported capture modes.
        
        Returns:
            list[str]: List of supported capture mode names
        """
        return self._capture_engine.get_supported_modes()
    
    def configure_capture_rate(self, fps: int) -> bool:
        """
        Configure the capture frame rate.
        
        Args:
            fps: Target frames per second
            
        Returns:
            bool: True if frame rate configured successfully
        """
        return self._capture_engine.set_frame_rate(fps)
    
    def is_available(self) -> bool:
        """
        Check if capture functionality is available.
        
        Returns:
            bool: True if capture is available
        """
        return self._capture_engine.get_status() != CaptureStatus.UNAVAILABLE
    
    def start_continuous_capture(self, callback: Callable[[Frame], None]) -> bool:
        """
        Start continuous frame capture with callback.
        
        Args:
            callback: Function to call with each captured frame
            
        Returns:
            bool: True if continuous capture started successfully
        """
        self._capture_engine.add_frame_callback(callback)
        return self._capture_engine.start_capture()
    
    def stop_continuous_capture(self) -> bool:
        """
        Stop continuous frame capture.
        
        Returns:
            bool: True if capture stopped successfully
        """
        return self._capture_engine.stop_capture()
    
    def set_performance_profile(self, profile: PerformanceProfile) -> None:
        """
        Set performance profile for adaptive optimization.
        
        Args:
            profile: Performance profile
        """
        self._capture_engine.set_performance_profile(profile)
    
    def get_capture_statistics(self) -> dict[str, Any]:
        """
        Get capture performance statistics.
        
        Returns:
            dict[str, Any]: Capture statistics
        """
        return self._capture_engine.get_capture_stats()
    
    def get_available_monitors(self) -> dict[int, MonitorInfo]:
        """
        Get information about all available monitors.
        
        Returns:
            dict[int, MonitorInfo]: Dictionary of monitor ID to monitor info
        """
        return self._capture_engine.get_available_monitors()
    
    def get_primary_monitor_id(self) -> int | None:
        """
        Get the ID of the primary monitor.
        
        Returns:
            int | None: Primary monitor ID or None if not found
        """
        return self._capture_engine.get_primary_monitor_id()
    
    def create_full_screen_region(self, monitor_id: int | None = None) -> CaptureRegion | None:
        """
        Create a capture region for full screen capture.
        
        Args:
            monitor_id: Optional monitor ID, uses all monitors if None
            
        Returns:
            CaptureRegion | None: Full screen capture region or None if failed
        """
        return self._capture_engine.create_full_screen_region(monitor_id)
    
    def create_monitor_region(self, monitor_id: int, 
                            custom_region: Rectangle | None = None) -> CaptureRegion | None:
        """
        Create a capture region for a specific monitor.
        
        Args:
            monitor_id: Monitor ID
            custom_region: Optional custom region within the monitor
            
        Returns:
            CaptureRegion | None: Capture region or None if failed
        """
        return self._capture_engine.create_monitor_region(monitor_id, custom_region)
    
    def refresh_monitors(self) -> bool:
        """
        Refresh monitor information.
        
        Returns:
            bool: True if refresh successful
        """
        return self._capture_engine.refresh_monitors()
    
    def get_monitor_summary(self) -> dict[str, Any]:
        """
        Get a summary of monitor configuration.
        
        Returns:
            dict[str, Any]: Monitor summary
        """
        return self._capture_engine.get_monitor_summary()
    
    def cleanup(self) -> None:
        """Clean up capture resources."""
        self._capture_engine.cleanup()