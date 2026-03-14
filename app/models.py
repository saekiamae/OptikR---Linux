"""
Core Data Models

Defines the fundamental data structures used throughout the Translation System.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np
from datetime import datetime


class CaptureMode(Enum):
    """Capture mode enumeration."""
    DIRECTX = "directx"
    SCREENSHOT = "screenshot"
    DESKTOP_DUPLICATION = "desktop_duplication"


class RuntimeMode(Enum):
    """Runtime execution mode enumeration."""
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"


class PerformanceProfile(Enum):
    """Performance profile enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class LogLevel(Enum):
    """Logging level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Rectangle:
    """Rectangle coordinates for positioning."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate rectangle area."""
        return self.width * self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within rectangle."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


@dataclass
class CaptureRegion:
    """Defines the region to capture from screen."""
    rectangle: Rectangle
    monitor_id: int = 0
    window_handle: int | None = None
    region_id: str = "default"  # Unique identifier for this region
    enabled: bool = True  # Whether this region is active
    name: str = ""  # User-friendly name for the region
    
    def __post_init__(self):
        """Validate capture region parameters."""
        if self.rectangle.width <= 0 or self.rectangle.height <= 0:
            raise ValueError("Capture region must have positive dimensions")
        
        # Generate default name if not provided
        if not self.name:
            self.name = f"Region {self.region_id} (Monitor {self.monitor_id})"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'region_id': self.region_id,
            'monitor_id': self.monitor_id,
            'x': self.rectangle.x,
            'y': self.rectangle.y,
            'width': self.rectangle.width,
            'height': self.rectangle.height,
            'enabled': self.enabled,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CaptureRegion':
        """Create from dictionary."""
        return cls(
            rectangle=Rectangle(
                x=data.get('x', 0),
                y=data.get('y', 0),
                width=data.get('width', 800),
                height=data.get('height', 600)
            ),
            monitor_id=data.get('monitor_id', 0),
            region_id=data.get('region_id', 'default'),
            enabled=data.get('enabled', True),
            name=data.get('name', '')
        )


@dataclass
class MultiRegionConfig:
    """Configuration for multiple capture regions across monitors."""
    regions: list[CaptureRegion] = field(default_factory=list)
    
    def add_region(self, region: CaptureRegion):
        """Add a new capture region."""
        # Check if region with this ID already exists
        existing = self.get_region(region.region_id)
        if existing:
            # Update existing region
            self.regions = [r if r.region_id != region.region_id else region for r in self.regions]
        else:
            # Add new region
            self.regions.append(region)
    
    def remove_region(self, region_id: str):
        """Remove a capture region by ID."""
        self.regions = [r for r in self.regions if r.region_id != region_id]
    
    def get_region(self, region_id: str) -> CaptureRegion | None:
        """Get a region by ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None
    
    def get_enabled_regions(self) -> list[CaptureRegion]:
        """Get all enabled regions."""
        return [r for r in self.regions if r.enabled]
    
    def get_regions_for_monitor(self, monitor_id: int) -> list[CaptureRegion]:
        """Get all regions for a specific monitor."""
        return [r for r in self.regions if r.monitor_id == monitor_id]
    
    def enable_region(self, region_id: str):
        """Enable a region."""
        region = self.get_region(region_id)
        if region:
            region.enabled = True
    
    def disable_region(self, region_id: str):
        """Disable a region."""
        region = self.get_region(region_id)
        if region:
            region.enabled = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'regions': [r.to_dict() for r in self.regions],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MultiRegionConfig':
        """Create from dictionary."""
        config = cls()
        for region_data in data.get('regions', []):
            region = CaptureRegion.from_dict(region_data)
            config.add_region(region)
        return config


@dataclass
class TextFormatting:
    """Text formatting information."""
    font_family: str = "Arial"
    font_size: int = 12
    is_bold: bool = False
    is_italic: bool = False
    color: tuple[int, int, int] = (0, 0, 0)  # RGB
    background_color: tuple[int, int, int] | None = None


@dataclass
class Frame:
    """Captured screen content data structure."""
    data: np.ndarray
    timestamp: float
    source_region: CaptureRegion
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Optional confidence score for the frame
    text: str = ""  # Optional text content (for compatibility)
    
    def __post_init__(self):
        """Validate frame data."""
        if self.data is None or self.data.size == 0:
            raise ValueError("Frame data cannot be empty")
        if self.timestamp <= 0:
            raise ValueError("Frame timestamp must be positive")
    
    @property
    def width(self) -> int:
        """Get frame width."""
        return self.data.shape[1] if len(self.data.shape) >= 2 else 0
    
    @property
    def height(self) -> int:
        """Get frame height."""
        return self.data.shape[0] if len(self.data.shape) >= 1 else 0
    
    @property
    def channels(self) -> int:
        """Get number of color channels."""
        return self.data.shape[2] if len(self.data.shape) == 3 else 1


@dataclass
class TextBlock:
    """Extracted text with position and formatting information."""
    text: str
    position: Rectangle
    confidence: float
    language: str = "unknown"
    formatting: TextFormatting = field(default_factory=TextFormatting)
    estimated_font_size: int | None = None  # Estimated font size in pixels
    
    def __post_init__(self):
        """Validate text block data."""
        self.text = self.text.strip() if self.text else ""
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if text block has high confidence."""
        return self.confidence >= 0.8


@dataclass
class Translation:
    """Translation result with metadata."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    position: Rectangle
    confidence: float
    engine_used: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    estimated_font_size: int | None = None  # Estimated font size in pixels from OCR
    
    def __post_init__(self):
        """Validate translation data."""
        self.original_text = self.original_text.strip() if self.original_text else ""
        self.translated_text = self.translated_text.strip() if self.translated_text else ""
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def is_reliable(self) -> bool:
        """Check if translation is reliable."""
        return self.confidence >= 0.7


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    fps: float
    cpu_usage: float  # Percentage 0-100
    gpu_usage: float  # Percentage 0-100
    memory_usage: float  # MB
    latency_ms: float
    accuracy: float  # Percentage 0-100
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        """Validate performance metrics."""
        if self.fps < 0:
            raise ValueError("FPS cannot be negative")
        if not 0 <= self.cpu_usage <= 100:
            raise ValueError("CPU usage must be between 0 and 100")
        if not 0 <= self.gpu_usage <= 100:
            raise ValueError("GPU usage must be between 0 and 100")
        if self.memory_usage < 0:
            raise ValueError("Memory usage cannot be negative")
        if self.latency_ms < 0:
            raise ValueError("Latency cannot be negative")
        if not 0 <= self.accuracy <= 100:
            raise ValueError("Accuracy must be between 0 and 100")
    
    @property
    def is_performing_well(self) -> bool:
        """Check if system is performing within acceptable parameters."""
        return (self.fps >= 30 and 
                self.cpu_usage <= 80 and 
                self.latency_ms <= 100 and 
                self.accuracy >= 90)


@dataclass
class SystemStatus:
    """Overall system status information."""
    is_running: bool = False
    current_mode: RuntimeMode = RuntimeMode.AUTO
    current_profile: PerformanceProfile = PerformanceProfile.NORMAL
    active_engines: dict[str, str] = field(default_factory=dict)
    error_count: int = 0
    last_error: str | None = None
    uptime_seconds: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        return self.is_running and self.error_count < 10 and self.last_error is None

@dataclass
class OverlayConfig:
    """Configuration for overlay rendering."""
    enabled: bool = True
    opacity: float = 0.9
    font_family: str = "Arial"
    font_size: int = 14
    text_color: tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    background_color: tuple[int, int, int, int] = (0, 0, 0, 128)  # RGBA
    border_enabled: bool = True
    border_width: int = 1
    border_color: tuple[int, int, int, int] = (255, 255, 255, 200)  # RGBA
    shadow_enabled: bool = True
    animation_enabled: bool = True
    positioning_strategy: str = "smart"
    
    def __post_init__(self):
        """Validate overlay configuration."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("Opacity must be between 0.0 and 1.0")
        if self.font_size <= 0:
            raise ValueError("Font size must be positive")
        if self.border_width < 0:
            raise ValueError("Border width cannot be negative")