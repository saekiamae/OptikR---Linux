"""
Dependency Checker for OptikR

Verifies that all required Python packages are installed with correct versions.
"""

import importlib.metadata
from packaging import version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
import logging

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Checks Python package dependencies with version validation."""
    
    # Required packages with version specifications
    REQUIRED_PACKAGES = {
        'PyQt6': '>=6.5.0',
        'torch': '>=2.0.0',
        'transformers': '>=4.40.0',
        'easyocr': '>=1.7.0',
        'opencv-python': '>=4.8.0',
        'Pillow': '>=10.0.0',
        'numpy': '>=1.24.0',
        'requests': '>=2.31.0',
        'cryptography': '>=42.0.0',
        'pywin32': '>=306',
        'psutil': '>=5.9.0',
        'sentencepiece': '>=0.1.99',
        'huggingface-hub': '>=0.23.0',
        'mss': '>=9.0.1',
        'pystray': '>=0.19.0',
    }
    
    def check_all(self) -> tuple[bool, list[str]]:
        """
        Check all required dependencies.
        
        Returns:
            Tuple of (all_satisfied, missing_packages)
        """
        missing_packages = []
        
        for package_name, version_spec in self.REQUIRED_PACKAGES.items():
            if not self.check_package(package_name, version_spec):
                missing_packages.append(f"{package_name}{version_spec}")
        
        return len(missing_packages) == 0, missing_packages
    
    def check_package(self, package: str, version_spec: str) -> bool:
        """
        Check if specific package meets version requirement.
        
        Args:
            package: Package name (e.g., 'PyQt6')
            version_spec: Version specification (e.g., '>=6.5.0')
            
        Returns:
            True if package is installed and meets version requirement
        """
        installed_version = self.get_installed_version(package)
        
        if installed_version is None:
            logger.warning("Package %s is not installed", package)
            return False
        
        is_compatible = self.check_version_compatibility(installed_version, version_spec)
        
        if not is_compatible:
            logger.warning(
                "Package %s version %s does not meet requirement %s",
                package, installed_version, version_spec
            )
        
        return is_compatible
    
    def get_installed_version(self, package: str) -> str | None:
        """
        Get installed version of package.
        
        Args:
            package: Package name
            
        Returns:
            Version string if installed, None otherwise
        """
        package_variations = [
            package,
            package.lower(),
            package.replace('-', '_'),
            package.replace('_', '-'),
        ]
        
        for pkg_name in package_variations:
            try:
                return importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                continue
        
        return None
    
    def check_version_compatibility(self, installed: str, required: str) -> bool:
        """
        Check if installed version meets requirement.
        
        Args:
            installed: Installed version string (e.g., '6.5.2')
            required: Version requirement (e.g., '>=6.5.0')
            
        Returns:
            True if installed version meets requirement
        """
        try:
            installed_ver = version.parse(installed)
            specifier = SpecifierSet(required)
            return installed_ver in specifier
        except (version.InvalidVersion, InvalidSpecifier) as e:
            logger.error("Error parsing version: %s", e)
            return False

    def check_with_mode(self, mode: str, gpu_vendor: str | None = None) -> tuple[str, bool, str]:
        """Check dependencies with mode awareness.

        Determines the correct requirements file and validates that the
        installed PyTorch variant matches the expected one.
        Delegates PyTorch detection to PyTorchManager (single source of truth).

        Args:
            mode: "cpu" or "gpu"
            gpu_vendor: "nvidia" or "amd" (required when mode is "gpu")

        Returns:
            Tuple of (requirements_file, pytorch_correct, pytorch_info)
        """
        if mode == "gpu":
            if gpu_vendor == "amd":
                requirements_file = "requirements-gpu-rocm.txt"
            else:
                requirements_file = "requirements-gpu.txt"
        else:
            requirements_file = "requirements-cpu.txt"

        pytorch_correct = False
        pytorch_info = "PyTorch not installed"

        try:
            from app.utils.pytorch_manager import get_pytorch_manager
            mgr = get_pytorch_manager()
            info = mgr.get_pytorch_info()

            if not info['installed']:
                return requirements_file, False, "PyTorch not installed"

            torch_version = info['version']
            has_cuda = info['cuda_available']
            # ROCm detection: get_pytorch_info doesn't expose hip directly,
            # so fall back to checking torch.version.hip if CUDA is absent
            has_hip = False
            if not has_cuda:
                try:
                    import torch
                    has_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
                except ImportError:
                    pass

            if mode == "cpu":
                if not has_cuda and not has_hip:
                    pytorch_correct = True
                    pytorch_info = f"PyTorch {torch_version} (CPU) - correct"
                elif has_cuda:
                    cuda_ver = info.get('cuda_version', '?')
                    pytorch_info = f"PyTorch {torch_version} (CUDA {cuda_ver}) - GPU variant installed, CPU expected"
                elif has_hip:
                    pytorch_info = f"PyTorch {torch_version} (ROCm) - ROCm variant installed, CPU expected"

            elif mode == "gpu" and gpu_vendor == "nvidia":
                if has_cuda:
                    cuda_ver = info.get('cuda_version', '?')
                    pytorch_correct = True
                    pytorch_info = f"PyTorch {torch_version} (CUDA {cuda_ver}) - correct"
                else:
                    pytorch_info = f"PyTorch {torch_version} - CUDA variant expected but not found"

            elif mode == "gpu" and gpu_vendor == "amd":
                if has_hip:
                    pytorch_correct = True
                    pytorch_info = f"PyTorch {torch_version} (ROCm) - correct"
                else:
                    pytorch_info = f"PyTorch {torch_version} - ROCm variant expected but not found"

        except ImportError:
            pytorch_info = "PyTorch not installed"
        except Exception as e:
            pytorch_info = f"PyTorch check error: {e}"

        return requirements_file, pytorch_correct, pytorch_info
