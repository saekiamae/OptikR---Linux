"""
Hardware Detection for OptikR

Standalone hardware detection module. Detects CPU architecture, SIMD support,
GPU capabilities (CUDA/OpenCL/ROCm), and memory configuration.

Used by:
- First-run wizard (GPU vendor auto-detection)
- Diagnostics tab (hardware info display)
- Pipeline startup (execution strategy selection)
"""

import logging
import multiprocessing
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

# Optional GPU imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    cl = None


class ProcessorArchitecture(Enum):
    """CPU architecture types."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


class SIMDInstructionSet(Enum):
    """SIMD instruction set support levels."""
    SSE = "sse"
    SSE2 = "sse2"
    SSE3 = "sse3"
    SSSE3 = "ssse3"
    SSE4_1 = "sse4_1"
    SSE4_2 = "sse4_2"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"
    NEON = "neon"


class GPUBackend(Enum):
    """GPU acceleration backends."""
    ROCM = "rocm"
    CUDA = "cuda"
    OPENCL = "opencl"
    NONE = "none"


@dataclass
class HardwareCapabilities:
    """Hardware capabilities detection results."""
    cpu_count: int
    cpu_architecture: ProcessorArchitecture
    simd_support: list[SIMDInstructionSet]
    gpu_backend: GPUBackend
    gpu_memory_mb: int
    total_memory_mb: int
    cache_sizes: dict[str, int]
    numa_nodes: int
    timestamp: float = field(default_factory=time.time)


class HardwareDetector:
    """
    Hardware capabilities detection system.
    Detects CPU architecture, SIMD support, GPU capabilities, and memory config.
    """

    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        self._cpu_info_cache: dict[str, Any] | None = None
        self.config_manager = config_manager

        if self.config_manager:
            self._load_cached_hardware_info()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_capabilities(self) -> HardwareCapabilities:
        """Detect comprehensive hardware capabilities."""
        return HardwareCapabilities(
            cpu_count=self._detect_cpu_count(),
            cpu_architecture=self._detect_cpu_architecture(),
            simd_support=self._detect_simd_support(),
            gpu_backend=self._detect_gpu_backend(),
            gpu_memory_mb=self._detect_gpu_memory(),
            total_memory_mb=self._detect_total_memory(),
            cache_sizes=self._detect_cache_sizes(),
            numa_nodes=self._detect_numa_nodes(),
        )

    def get_amd_cpu_diagnostics(self) -> dict[str, Any] | None:
        """Get AMD CPU diagnostic info, or None if not AMD."""
        cpu_info = self._detect_cpu_info()
        if not cpu_info.get("is_amd", False):
            return None
        return {
            "vendor": cpu_info.get("vendor", "Unknown"),
            "model": cpu_info.get("model", "Unknown"),
            "zen_generation": cpu_info.get("zen_generation", "Unknown"),
            "cores": self._detect_cpu_count(),
            "simd_support": self._get_simd_features_list(
                cpu_info.get("simd_support", [])
            ),
        }

    def get_amd_gpu_diagnostics(self) -> dict[str, Any] | None:
        """Get AMD GPU diagnostic info, or None if no AMD GPU."""
        # Try ROCm (PyTorch + HIP)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device_name = torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name:
                    try:
                        memory_mb = (
                            torch.cuda.get_device_properties(0).total_memory
                            // (1024 * 1024)
                        )
                    except Exception:
                        memory_mb = 0
                    return {
                        "model": device_name,
                        "memory_mb": memory_mb,
                        "device_count": torch.cuda.device_count(),
                        "rocm_available": True,
                    }
        except ImportError:
            pass

        # Try OpenCL
        if PYOPENCL_AVAILABLE:
            try:
                for plat in cl.get_platforms():
                    if "AMD" in plat.name or "Advanced Micro Devices" in plat.name:
                        devices = plat.get_devices(device_type=cl.device_type.GPU)
                        if devices:
                            dev = devices[0]
                            return {
                                "model": dev.name,
                                "memory_mb": dev.global_mem_size // (1024 * 1024),
                                "device_count": len(devices),
                                "rocm_available": False,
                                "opencl_version": dev.version,
                            }
            except Exception:
                pass

        return None

    def get_active_gpu_backend(self) -> str:
        """Return active GPU backend name for display."""
        backend = self._detect_gpu_backend()
        names = {
            GPUBackend.ROCM: "ROCm",
            GPUBackend.OPENCL: "OpenCL",
            GPUBackend.CUDA: "CUDA",
            GPUBackend.NONE: "None",
        }
        return names.get(backend, "Unknown")

    def get_enabled_simd_instructions(self) -> list[str]:
        """Return list of enabled SIMD instruction set names."""
        simd_support = self._detect_simd_support()
        names = {
            SIMDInstructionSet.SSE: "SSE",
            SIMDInstructionSet.SSE2: "SSE2",
            SIMDInstructionSet.SSE3: "SSE3",
            SIMDInstructionSet.SSSE3: "SSSE3",
            SIMDInstructionSet.SSE4_1: "SSE4.1",
            SIMDInstructionSet.SSE4_2: "SSE4.2",
            SIMDInstructionSet.AVX: "AVX",
            SIMDInstructionSet.AVX2: "AVX2",
            SIMDInstructionSet.AVX512: "AVX512",
            SIMDInstructionSet.NEON: "NEON",
        }
        return [names.get(s, str(s)) for s in simd_support]

    def save_hardware_info_to_config(self) -> None:
        """Persist detected hardware info to config for caching."""
        if not self.config_manager:
            return
        try:
            cpu_info = self._detect_cpu_info()
            if cpu_info.get("is_amd"):
                self.config_manager.update_amd_cpu_config(
                    {
                        "is_amd": True,
                        "vendor": cpu_info.get("vendor"),
                        "model": cpu_info.get("model"),
                        "cores": self._detect_cpu_count(),
                        "zen_generation": cpu_info.get("zen_generation"),
                        "simd_support": self._get_simd_features_list(
                            cpu_info.get("simd_support", [])
                        ),
                    }
                )
            gpu_info = self.get_amd_gpu_diagnostics()
            backend = self.get_active_gpu_backend()
            self.config_manager.update_amd_gpu_config(gpu_info, backend)
            self.config_manager.save_config()
            self.logger.info("Saved hardware info to config")
        except Exception as e:
            self.logger.error(f"Failed to save hardware info to config: {e}")

    def check_hardware_changes(self) -> bool:
        """Return True if hardware changed since last cached detection."""
        if not self.config_manager:
            return True
        try:
            amd_config = self.config_manager.get_amd_hardware_config()
            cpu_info = self._detect_cpu_info()
            gpu_info = self.get_amd_gpu_diagnostics()

            if cpu_info.get("is_amd") != amd_config.get("cpu_detected"):
                return True
            if cpu_info.get("model") != amd_config.get("cpu_model"):
                return True
            if (gpu_info is not None) != amd_config.get("gpu_detected"):
                return True
            if gpu_info and gpu_info.get("model") != amd_config.get("gpu_model"):
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check hardware changes: {e}")
            return True

    # ------------------------------------------------------------------
    # CPU detection (private)
    # ------------------------------------------------------------------

    def _detect_cpu_info(self) -> dict[str, Any]:
        """Detect detailed CPU information including vendor, model, AMD details."""
        if self._cpu_info_cache is not None:
            return self._cpu_info_cache

        cpu_info: dict[str, Any] = {
            "vendor": "Unknown",
            "model": "Unknown",
            "is_amd": False,
            "zen_generation": None,
            "simd_support": [],
        }

        try:
            try:
                import cpuinfo

                info = cpuinfo.get_cpu_info()
                cpu_info["vendor"] = info.get("vendor_id_raw", "Unknown")
                cpu_info["model"] = info.get("brand_raw", "Unknown")
                cpu_info["simd_support"] = info.get("flags", [])
            except ImportError:
                if platform.system() == "Linux":
                    cpu_info.update(self._detect_cpu_info_linux())
                elif platform.system() == "Windows":
                    cpu_info.update(self._detect_cpu_info_windows())
                else:
                    cpu_info["model"] = platform.processor()

            # AMD-specific enrichment
            if "AuthenticAMD" in cpu_info["vendor"] or "AMD" in cpu_info["vendor"]:
                cpu_info["is_amd"] = True
                cpu_info["zen_generation"] = self._detect_zen_generation(
                    cpu_info["model"]
                )
                if not cpu_info["simd_support"]:
                    cpu_info["simd_support"] = self._get_cpu_flags()

        except Exception as e:
            self.logger.error(f"CPU info detection failed: {e}", exc_info=True)

        self._cpu_info_cache = cpu_info
        return cpu_info

    def _detect_cpu_info_linux(self) -> dict[str, Any]:
        """Parse /proc/cpuinfo on Linux."""
        result: dict[str, Any] = {
            "vendor": "Unknown",
            "model": "Unknown",
            "simd_support": [],
        }
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("vendor_id"):
                        result["vendor"] = line.split(":")[1].strip()
                    elif line.startswith("model name"):
                        result["model"] = line.split(":")[1].strip()
                    elif line.startswith("flags"):
                        result["simd_support"] = line.split(":")[1].strip().split()
                        break
        except Exception as e:
            self.logger.warning(f"Failed to read /proc/cpuinfo: {e}")
        return result

    def _detect_cpu_info_windows(self) -> dict[str, Any]:
        """Detect CPU info on Windows via wmic."""
        result: dict[str, Any] = {
            "vendor": "Unknown",
            "model": "Unknown",
            "simd_support": [],
        }
        try:
            import subprocess

            proc = subprocess.run(
                ["wmic", "cpu", "get", "Name,Manufacturer"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                lines = proc.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    if parts:
                        result["vendor"] = parts[0]
                        result["model"] = (
                            " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
                        )
                        if "AMD" in result["vendor"]:
                            result["vendor"] = "AuthenticAMD"
        except Exception as e:
            self.logger.warning(f"Failed to run wmic: {e}")
        return result

    def _detect_zen_generation(self, model_name: str) -> str | None:
        """Infer Zen architecture generation from CPU model name."""
        m = model_name.lower()

        zen4 = ["7950x", "7900x", "7700x", "7600x", "genoa", "bergamo",
                 "9654", "9554", "9374"]
        zen3 = ["5950x", "5900x", "5800x", "5600x", "milan",
                 "7763", "7713", "7643", "7543"]
        zen2 = ["3950x", "3900x", "3700x", "3600x", "rome",
                 "3990x", "3970x", "7742", "7702"]
        zenp = ["2700x", "2600x", "2400g"]
        zen1 = ["1800x", "1700x", "1600x", "naples", "7601", "7551"]

        if any(x in m for x in zen4):
            return "Zen 4"
        if any(x in m for x in zen3):
            return "Zen 3"
        if any(x in m for x in zen2):
            return "Zen 2"
        if any(x in m for x in zenp):
            return "Zen+"
        if any(x in m for x in zen1):
            return "Zen"
        if "ryzen" in m or "epyc" in m or "threadripper" in m:
            return "Unknown Zen"
        return None

    def _get_cpu_flags(self) -> list[str]:
        """Get CPU feature flags from the system."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("flags"):
                            return line.split(":")[1].strip().split()
        except Exception as e:
            self.logger.warning(f"Failed to get CPU flags: {e}")
        return []

    def _get_simd_features_list(self, flags: list[str]) -> list[str]:
        """Extract human-readable SIMD feature names from CPU flags."""
        features = []
        if "avx512f" in flags:
            features.append("AVX512")
        elif "avx2" in flags:
            features.append("AVX2")
        elif "avx" in flags:
            features.append("AVX")
        if "sse4_2" in flags:
            features.append("SSE4.2")
        elif "sse4_1" in flags:
            features.append("SSE4.1")
        if "neon" in flags:
            features.append("NEON")
        return features if features else ["None"]

    # ------------------------------------------------------------------
    # Hardware primitives (private)
    # ------------------------------------------------------------------

    def _detect_cpu_count(self) -> int:
        return multiprocessing.cpu_count()

    def _detect_cpu_architecture(self) -> ProcessorArchitecture:
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return ProcessorArchitecture.X86_64
        if machine in ("aarch64", "arm64"):
            return ProcessorArchitecture.ARM64
        if machine.startswith("arm"):
            return ProcessorArchitecture.ARM32
        return ProcessorArchitecture.UNKNOWN

    def _detect_simd_support(self) -> list[SIMDInstructionSet]:
        supported: list[SIMDInstructionSet] = []
        try:
            if platform.machine().lower() in ("x86_64", "amd64"):
                try:
                    import cpuinfo

                    flags = cpuinfo.get_cpu_info().get("flags", [])
                    flag_map = {
                        "sse": SIMDInstructionSet.SSE,
                        "sse2": SIMDInstructionSet.SSE2,
                        "sse3": SIMDInstructionSet.SSE3,
                        "ssse3": SIMDInstructionSet.SSSE3,
                        "sse4_1": SIMDInstructionSet.SSE4_1,
                        "sse4_2": SIMDInstructionSet.SSE4_2,
                        "avx": SIMDInstructionSet.AVX,
                        "avx2": SIMDInstructionSet.AVX2,
                        "avx512f": SIMDInstructionSet.AVX512,
                    }
                    for flag, simd in flag_map.items():
                        if flag in flags:
                            supported.append(simd)
                except ImportError:
                    supported.extend(
                        [SIMDInstructionSet.SSE, SIMDInstructionSet.SSE2]
                    )
            elif platform.machine().lower().startswith("arm"):
                supported.append(SIMDInstructionSet.NEON)
        except Exception as e:
            self.logger.warning(f"SIMD detection failed: {e}")
        return supported

    def _detect_gpu_backend(self) -> GPUBackend:
        """Detect available GPU backend. Prioritises ROCm for AMD systems."""
        cpu_info = self._detect_cpu_info()
        is_amd = cpu_info.get("is_amd", False)

        if is_amd:
            backend = self._try_rocm()
            if backend:
                return backend
            backend = self._try_opencl_amd()
            if backend:
                return backend
            return GPUBackend.NONE

        # Non-AMD: CUDA then OpenCL
        if CUPY_AVAILABLE:
            try:
                if cp.cuda.runtime.getDeviceCount() > 0:
                    return GPUBackend.CUDA
            except Exception:
                pass

        if PYOPENCL_AVAILABLE:
            try:
                if cl.get_platforms():
                    return GPUBackend.OPENCL
            except Exception:
                pass

        return GPUBackend.NONE

    # ------------------------------------------------------------------
    # GPU backend helpers
    # ------------------------------------------------------------------

    def _try_rocm(self) -> GPUBackend | None:
        """Attempt ROCm detection via PyTorch HIP."""
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                name = torch.cuda.get_device_name(0)
                if "AMD" in name or "Radeon" in name:
                    self.logger.info(f"ROCm backend detected: {name}")
                    return GPUBackend.ROCM
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"ROCm detection failed: {e}")
        return None

    def _try_opencl_amd(self) -> GPUBackend | None:
        """Attempt OpenCL detection for AMD GPUs."""
        if not PYOPENCL_AVAILABLE:
            return None
        try:
            for plat in cl.get_platforms():
                if "AMD" in plat.name or "Advanced Micro Devices" in plat.name:
                    devices = plat.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        self.logger.info(
                            f"OpenCL backend detected for AMD GPU: {devices[0].name}"
                        )
                        return GPUBackend.OPENCL
        except Exception as e:
            self.logger.warning(f"OpenCL AMD detection failed: {e}")
        return None

    def _detect_gpu_memory(self) -> int:
        """Detect GPU memory in MB."""
        try:
            if CUPY_AVAILABLE:
                return cp.cuda.Device().mem_info[1] // (1024 * 1024)
        except Exception:
            pass
        try:
            if PYOPENCL_AVAILABLE:
                platforms = cl.get_platforms()
                if platforms:
                    devices = platforms[0].get_devices()
                    if devices:
                        return devices[0].global_mem_size // (1024 * 1024)
        except Exception:
            pass
        return 0

    def _detect_total_memory(self) -> int:
        """Detect total system memory in MB."""
        return psutil.virtual_memory().total // (1024 * 1024)

    @staticmethod
    def _parse_cache_string(value: str) -> int | None:
        """Parse a cache size string like '256 KiB' or '6144 KB' into bytes."""
        if not value:
            return None
        value = value.strip()
        multipliers = {
            "B": 1, "KB": 1024, "KIB": 1024,
            "MB": 1024 ** 2, "MIB": 1024 ** 2,
        }
        for suffix, mult in multipliers.items():
            if value.upper().endswith(suffix):
                try:
                    return int(float(value[: -len(suffix)].strip()) * mult)
                except ValueError:
                    return None
        try:
            return int(value)
        except ValueError:
            return None

    def _detect_cache_sizes(self) -> dict[str, int]:
        """Return CPU cache sizes (best-effort, falls back to typical values)."""
        defaults = {"L1": 32 * 1024, "L2": 256 * 1024, "L3": 8 * 1024 * 1024}

        # Try py-cpuinfo first
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            l2 = self._parse_cache_string(str(info.get("l2_cache_size", "")))
            l3 = self._parse_cache_string(str(info.get("l3_cache_size", "")))
            if l2:
                defaults["L2"] = l2
            if l3:
                defaults["L3"] = l3
        except Exception:
            pass

        # On Linux, sysfs gives authoritative per-level sizes
        if platform.system() == "Linux":
            try:
                base = "/sys/devices/system/cpu/cpu0/cache"
                for entry in os.listdir(base):
                    level_path = os.path.join(base, entry, "level")
                    size_path = os.path.join(base, entry, "size")
                    type_path = os.path.join(base, entry, "type")
                    if not (os.path.isfile(level_path) and os.path.isfile(size_path)):
                        continue
                    level = open(level_path).read().strip()
                    size_str = open(size_path).read().strip()
                    cache_type = open(type_path).read().strip() if os.path.isfile(type_path) else ""
                    parsed = self._parse_cache_string(size_str)
                    if not parsed:
                        continue
                    if level == "1" and cache_type in ("Data", "Unified", ""):
                        defaults["L1"] = parsed
                    elif level == "2":
                        defaults["L2"] = parsed
                    elif level == "3":
                        defaults["L3"] = parsed
            except Exception:
                pass

        # On Windows, use WMI via subprocess
        if platform.system() == "Windows":
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get",
                     "L2CacheSize,L3CacheSize", "/format:list"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("L2CacheSize="):
                        val = line.split("=", 1)[1].strip()
                        if val.isdigit() and int(val) > 0:
                            defaults["L2"] = int(val) * 1024  # WMI reports KB
                    elif line.startswith("L3CacheSize="):
                        val = line.split("=", 1)[1].strip()
                        if val.isdigit() and int(val) > 0:
                            defaults["L3"] = int(val) * 1024
            except Exception:
                pass

        return defaults

    def _detect_numa_nodes(self) -> int:
        """Detect NUMA node count (Linux only, defaults to 1)."""
        try:
            entries = [
                d
                for d in os.listdir("/sys/devices/system/node/")
                if d.startswith("node") and d[4:].isdigit()
            ]
            return len(entries)
        except (OSError, FileNotFoundError):
            return 1

    # ------------------------------------------------------------------
    # Config caching (private)
    # ------------------------------------------------------------------

    def _load_cached_hardware_info(self) -> None:
        """Load cached hardware info from config to skip re-detection."""
        if not self.config_manager:
            return
        try:
            amd_config = self.config_manager.get_amd_hardware_config()
            if amd_config and amd_config.get("last_detection_timestamp"):
                from datetime import datetime, timedelta

                try:
                    last = datetime.fromisoformat(
                        amd_config["last_detection_timestamp"]
                    )
                    if datetime.now() - last < timedelta(hours=24):
                        if amd_config.get("cpu_detected"):
                            self._cpu_info_cache = {
                                "vendor": amd_config.get("cpu_vendor", "Unknown"),
                                "model": amd_config.get("cpu_model", "Unknown"),
                                "is_amd": True,
                                "zen_generation": amd_config.get(
                                    "cpu_zen_generation"
                                ),
                                "simd_support": amd_config.get(
                                    "cpu_simd_support", []
                                ),
                                "cores": amd_config.get("cpu_cores"),
                            }
                            self.logger.info("Loaded hardware info from cache")
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            self.logger.warning(f"Failed to load cached hardware info: {e}")
