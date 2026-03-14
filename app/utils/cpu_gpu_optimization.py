"""
CPU and GPU Optimization — backward-compatibility shim.

The active hardware detection code has been extracted to:
    app.utils.hardware_detection

This module re-exports the public symbols so existing imports continue to work.
"""

# Re-export everything from hardware_detection so old imports still resolve
from app.utils.hardware_detection import (  # noqa: F401
    GPUBackend,
    HardwareCapabilities,
    HardwareDetector,
    ProcessorArchitecture,
    SIMDInstructionSet,
)
