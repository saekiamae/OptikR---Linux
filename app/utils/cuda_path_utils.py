"""
CUDA installation path validation.

Used so the user can choose a CUDA toolkit root folder (e.g. from another drive
or non-standard install). Validates by checking for known files that exist in
a real CUDA installation.
"""

import os
import sys
from pathlib import Path


# Files/dirs that indicate a valid CUDA toolkit root (relative to the chosen folder).
# We require at least one of these to exist.
_WINDOWS_VALIDATORS = [
    "bin/nvcc.exe",           # Compiler - always present in toolkit
    "bin/cudart64_12.dll",   # CUDA 12 runtime
    "bin/cudart64_11.dll",   # CUDA 11 runtime
]
_LINUX_VALIDATORS = [
    "bin/nvcc",
    "lib64/libcudart.so",
]


def validate_cuda_installation(path: str | Path) -> tuple[bool, str]:
    """
    Check that the given path is a valid CUDA toolkit root.

    The path should be the CUDA root (e.g. containing bin/, lib/, include/).
    We validate by looking for nvcc or cudart DLL/so in the expected subpaths.

    Returns:
        (True, "") if valid.
        (False, "reason") if path is missing, not a directory, or no validator file found.
    """
    p = Path(path).resolve()
    if not p.exists():
        return False, "Path does not exist."
    if not p.is_dir():
        return False, "Path is not a directory."
    if sys.platform == "win32":
        for rel in _WINDOWS_VALIDATORS:
            if (p / rel).exists():
                return True, ""
        # Also accept any cudart64_*.dll in bin (different CUDA versions)
        bin_dir = p / "bin"
        if bin_dir.is_dir():
            for f in bin_dir.glob("cudart64_*.dll"):
                if f.is_file():
                    return True, ""
        return False, (
            "No CUDA toolkit files found. The folder should contain "
            "'bin\\nvcc.exe' or 'bin\\cudart64_*.dll'. "
            "Typical location: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x"
        )
    # Linux (and other Unix)
    for rel in _LINUX_VALIDATORS:
        if (p / rel).exists():
            return True, ""
    return False, (
        "No CUDA toolkit files found. The folder should contain "
        "'bin/nvcc' or 'lib64/libcudart.so'. Typical location: /usr/local/cuda"
    )


def get_cuda_path_hint() -> str:
    """Return a short hint for the user where CUDA is usually installed."""
    if sys.platform == "win32":
        return "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x"
    return "/usr/local/cuda"
