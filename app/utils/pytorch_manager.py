"""
PyTorch Version Manager

Handles switching between CPU and CUDA versions of PyTorch.
"""

import subprocess
import sys
import os
from enum import Enum


class PyTorchVersion(Enum):
    """PyTorch version types."""
    CPU = "cpu"
    CUDA_118 = "cu118"
    CUDA_121 = "cu121"
    CUDA_124 = "cu124"


class PyTorchManager:
    """Manages PyTorch installation and version switching."""
    
    def __init__(self, config_manager=None):
        """Initialize PyTorch manager.
        
        Args:
            config_manager: Optional configuration manager for reading URL settings
        """
        self.config_manager = config_manager
        self.current_version = self.detect_current_version()
    
    def detect_current_version(self) -> tuple[str, bool]:
        """Detect currently installed PyTorch version.
        
        Returns:
            Tuple of (version_string, cuda_available)
        """
        try:
            import torch
            version = getattr(torch, '__version__', 'Unknown')
            cuda_available = torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False
            return version, cuda_available
        except (ImportError, AttributeError) as e:
            return "Not installed", False
    
    def get_pytorch_info(self) -> dict:
        """Get detailed PyTorch information.
        
        Returns:
            Dictionary with PyTorch details
        """
        try:
            import torch
            
            info = {
                'installed': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'devices': []
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    info['devices'].append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'capability': torch.cuda.get_device_capability(i)
                    })
            
            # Determine version type
            if '+cpu' in info['version']:
                info['type'] = 'CPU-only'
            elif '+cu' in info['version']:
                cuda_ver = info['version'].split('+cu')[1][:3]
                info['type'] = f'CUDA {cuda_ver}'
            else:
                info['type'] = 'Unknown'
            
            return info
            
        except ImportError:
            return {
                'installed': False,
                'version': None,
                'cuda_available': False,
                'type': 'Not installed'
            }
    
    def check_cuda_toolkit(self) -> dict:
        """Check if CUDA toolkit is installed.
        
        Returns:
            Dictionary with CUDA toolkit information
        """
        cuda_info = {
            'installed': False,
            'versions': [],
            'driver_version': None
        }
        
        # Check for CUDA toolkit
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
            "/usr/local/cuda"
        ]
        
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                cuda_info['installed'] = True
                try:
                    versions = [d for d in os.listdir(cuda_path) if os.path.isdir(os.path.join(cuda_path, d))]
                    cuda_info['versions'] = versions
                except OSError:
                    pass
        
        # Check NVIDIA driver
        try:
            # Get timeout from config
            timeout = 5.0
            if self.config_manager:
                timeout = self.config_manager.get_setting('timeouts.nvidia_smi_seconds', 5.0)
            
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                cuda_info['driver_version'] = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        
        return cuda_info
    
    def get_install_command(self, version_type: PyTorchVersion) -> list:
        """Get pip install command for specified PyTorch version.

        Always uses ``--force-reinstall --no-deps`` so that:
        - A CPU-only build that was dragged in by another package
          (easyocr, surya-ocr, etc.) is unconditionally replaced.
        - No unrelated packages are touched by the reinstall.
        - ``--index-url`` (not ``--extra-index-url``) ensures pip can
          ONLY see wheels from the target index, preventing it from
          preferring a higher-versioned CPU wheel on PyPI.

        Args:
            version_type: PyTorch version to install

        Returns:
            List of command arguments
        """
        base_packages = ['torch', 'torchvision', 'torchaudio']

        # Get URLs from config if available, otherwise use defaults
        if self.config_manager:
            cpu_url = self.config_manager.get_setting('urls.pytorch_cpu_index', 'https://download.pytorch.org/whl/cpu')
            cuda118_url = self.config_manager.get_setting('urls.pytorch_cuda118_index', 'https://download.pytorch.org/whl/cu118')
            cuda121_url = self.config_manager.get_setting('urls.pytorch_cuda121_index', 'https://download.pytorch.org/whl/cu121')
            cuda124_url = self.config_manager.get_setting('urls.pytorch_cuda124_index', 'https://download.pytorch.org/whl/cu124')
        else:
            cpu_url = 'https://download.pytorch.org/whl/cpu'
            cuda118_url = 'https://download.pytorch.org/whl/cu118'
            cuda121_url = 'https://download.pytorch.org/whl/cu121'
            cuda124_url = 'https://download.pytorch.org/whl/cu124'

        install_flags = ['--force-reinstall', '--no-deps']

        url_map = {
            PyTorchVersion.CPU: cpu_url,
            PyTorchVersion.CUDA_118: cuda118_url,
            PyTorchVersion.CUDA_121: cuda121_url,
            PyTorchVersion.CUDA_124: cuda124_url,
        }

        index_url = url_map.get(version_type)
        if index_url is None:
            return []

        return [sys.executable, '-m', 'pip', 'install'] + install_flags + \
               base_packages + ['--index-url', index_url]
    
    def uninstall_pytorch(self, callback=None) -> tuple[bool, str]:
        """Uninstall current PyTorch installation.
        
        Args:
            callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if callback:
                callback("Uninstalling PyTorch...")
            
            packages = ['torch', 'torchvision', 'torchaudio']
            cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y'] + packages
            
            # Get timeout from config
            timeout = 120.0
            if self.config_manager:
                timeout = self.config_manager.get_setting('timeouts.pip_uninstall_seconds', 120.0)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                return True, "PyTorch uninstalled successfully"
            else:
                return False, f"Uninstall failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Uninstall timed out"
        except Exception as e:
            return False, f"Uninstall error: {str(e)}"
    
    def install_pytorch(self, version_type: PyTorchVersion, callback=None) -> tuple[bool, str]:
        """Install specified PyTorch version.
        
        Args:
            version_type: PyTorch version to install
            callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if callback:
                callback(f"Installing PyTorch ({version_type.value})...")
            
            cmd = self.get_install_command(version_type)
            
            if not cmd:
                return False, "Invalid version type"
            
            # Run with real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if callback:
                    # Extract progress information from pip output
                    line_clean = line.strip()
                    if 'Downloading' in line_clean:
                        callback(f"Downloading packages...")
                    elif 'Installing' in line_clean:
                        callback(f"Installing packages...")
                    elif 'Successfully installed' in line_clean:
                        callback(f"Installation complete!")
                    elif '%' in line_clean or 'MB' in line_clean or 'GB' in line_clean:
                        # Show download progress
                        callback(line_clean[:80])  # Limit length
            
            # Get timeout from config
            timeout = 600.0
            if self.config_manager:
                timeout = self.config_manager.get_setting('timeouts.pip_install_seconds', 600.0)
            
            process.wait(timeout=timeout)
            
            if process.returncode == 0:
                return True, f"PyTorch ({version_type.value}) installed successfully"
            else:
                error_output = '\n'.join(output_lines[-10:])  # Last 10 lines
                return False, f"Install failed:\n{error_output}"
                
        except subprocess.TimeoutExpired:
            return False, "Install timed out (this can take several minutes)"
        except Exception as e:
            return False, f"Install error: {str(e)}"
    
    def switch_version(self, version_type: PyTorchVersion, callback=None) -> tuple[bool, str]:
        """Switch to specified PyTorch version.
        
        Args:
            version_type: PyTorch version to switch to
            callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (success, message)
        """
        # Step 1: Uninstall current version
        if callback:
            callback("Step 1/2: Uninstalling current PyTorch...")
        
        success, message = self.uninstall_pytorch(callback)
        if not success:
            return False, f"Uninstall failed: {message}"
        
        # Step 2: Install new version
        if callback:
            callback("Step 2/2: Installing new PyTorch version...")
        
        success, message = self.install_pytorch(version_type, callback)
        if not success:
            return False, f"Install failed: {message}"
        
        return True, f"Successfully switched to PyTorch {version_type.value}"
    


def release_gpu_memory() -> None:
    """Force-release GPU memory held by PyTorch / CUDA.

    Call after deleting model references (setting them to ``None``) so
    that Python's garbage collector finalises the tensors and the CUDA
    caching allocator returns the freed blocks to the driver.

    Safe to call even when PyTorch or CUDA are unavailable — the
    function silently does nothing in that case.
    """
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


_pytorch_manager_instance: PyTorchManager | None = None


def get_pytorch_manager(config_manager=None) -> PyTorchManager:
    """Get or create the singleton PyTorchManager instance.

    Args:
        config_manager: Optional config manager (used only on first call)

    Returns:
        Shared PyTorchManager instance
    """
    global _pytorch_manager_instance
    if _pytorch_manager_instance is None:
        _pytorch_manager_instance = PyTorchManager(config_manager)
    return _pytorch_manager_instance

