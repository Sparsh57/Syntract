"""
GPU utilities for graceful GPU/CPU fallback in the synthesis package.

This module provides centralized GPU library detection and management,
allowing functions to use whatever GPU acceleration is available without
requiring all GPU dependencies to be installed.
"""

from types import ModuleType
from typing import Any, Dict, Optional, Tuple


class GPUSupport:
    """Manages GPU library availability and provides graceful fallbacks."""

    def __init__(self):
        self.cupy_available = False
        self.numba_cuda_available = False
        self.cupy: Optional[ModuleType] = None
        self.cuda: Optional[ModuleType] = None
        self._initialized = False

    def initialize(self, verbose: bool = True) -> None:
        """Initialize GPU support detection."""
        if self._initialized:
            return

        # Try to import CuPy
        try:
            import cupy
            self.cupy_available = True
            self.cupy = cupy
            if verbose:
                print("CuPy available for GPU array operations")
        except ImportError:
            if verbose:
                print("ERROR: CuPy not available (array operations will use CPU)")

        # Try to import Numba CUDA
        try:
            from numba import cuda
            self.numba_cuda_available = True
            self.cuda = cuda
            if verbose:
                print("Numba CUDA available for GPU kernels")
        except ImportError:
            if verbose:
                print("ERROR: Numba CUDA not available (kernels will use CPU)")

        self._initialized = True

    def get_array_module(self, prefer_gpu: bool = True) -> ModuleType:
        """Get the best available array module (CuPy or NumPy)."""
        if not self._initialized:
            self.initialize(verbose=False)

        if prefer_gpu and self.cupy_available:
            # Test if CuPy can actually create arrays (has GPU device)
            try:
                import cupy as cp
                # Check if CUDA devices are available
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count == 0:
                    print("INFO: CuPy available but no CUDA devices found. Using CPU processing.")
                    import numpy
                    return numpy

                # Try to create a small test array to ensure GPU is actually usable
                with cp.cuda.Device(0):
                    cp.array([1.0], dtype=cp.float32)
                print("INFO: Using GPU processing (GPU libraries available)")
                return self.cupy
            except Exception as e:
                # GPU not available or other CuPy error - fall back to NumPy
                error_str = str(e)
                if any(err in error_str for err in ["cudaErrorNoDevice", "CUDA", "cuda", "device"]):
                    print("INFO: Using CPU processing (GPU libraries not available)")
                else:
                    print(f"WARNING: CuPy error: {e}. Falling back to CPU (NumPy).")
                import numpy
                return numpy
        else:
            import numpy
            return numpy

    def try_import_cupy(self, fallback_to_numpy: bool = True) -> Tuple[ModuleType, bool]:
        """
        Try to import CuPy, with optional fallback to NumPy.

        Returns:
            tuple: (module, is_gpu) where module is CuPy or NumPy,
                   and is_gpu indicates if GPU acceleration is available
        """
        if not self._initialized:
            self.initialize(verbose=False)

        if self.cupy_available:
            return self.cupy, True
        elif fallback_to_numpy:
            import numpy
            return numpy, False
        else:
            raise ImportError("CuPy not available and fallback disabled")

    def try_import_numba_cuda(self, raise_on_fail: bool = False) -> Tuple[Optional[ModuleType], bool]:
        """
        Try to import Numba CUDA.

        Returns:
            tuple: (cuda_module, is_available) where cuda_module is the CUDA module
                   or None if not available, and is_available indicates availability
        """
        if not self._initialized:
            self.initialize(verbose=False)

        if self.numba_cuda_available:
            return self.cuda, True
        elif raise_on_fail:
            raise ImportError("Numba CUDA not available")
        else:
            return None, False


# Global GPU support instance
_gpu_support: Optional[GPUSupport] = None

def get_gpu_support() -> GPUSupport:
    """Get the global GPU support instance."""
    global _gpu_support
    if _gpu_support is None:
        _gpu_support = GPUSupport()
    return _gpu_support

def get_array_module(prefer_gpu: bool = True) -> ModuleType:
    """Get the best available array module (CuPy or NumPy)."""
    return get_gpu_support().get_array_module(prefer_gpu)

def try_gpu_import() -> Dict[str, Any]:
    """
    Try to import GPU libraries with graceful fallback.

    Returns:
        dict: ``xp`` (CuPy or NumPy), ``cuda`` (Numba CUDA module or None),
        ``cupy_available``, ``numba_available``, ``gpu_support``.
    """
    gpu_support = get_gpu_support()

    xp, cupy_available = gpu_support.try_import_cupy()
    cuda, numba_available = gpu_support.try_import_numba_cuda()

    return {
        'xp': xp,
        'cuda': cuda,
        'cupy_available': cupy_available,
        'numba_available': numba_available,
        'gpu_support': gpu_support
    }
