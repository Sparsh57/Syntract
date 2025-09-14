"""
GPU utilities for graceful GPU/CPU fallback in the synthesis package.

This module provides centralized GPU library detection and management,
allowing functions to use whatever GPU acceleration is available without
requiring all GPU dependencies to be installed.
"""

import warnings

class GPUSupport:
    """Manages GPU library availability and provides graceful fallbacks."""
    
    def __init__(self):
        self.cupy_available = False
        self.numba_cuda_available = False
        self.cupy = None
        self.cuda = None
        self._initialized = False
        
    def initialize(self, verbose=True):
        """Initialize GPU support detection."""
        if self._initialized:
            return
            
        # Try to import CuPy
        try:
            import cupy
            self.cupy_available = True
            self.cupy = cupy
            if verbose:
                print("✓ CuPy available for GPU array operations")
        except ImportError:
            if verbose:
                print("✗ CuPy not available (array operations will use CPU)")
        
        # Try to import Numba CUDA
        try:
            from numba import cuda
            self.numba_cuda_available = True 
            self.cuda = cuda
            if verbose:
                print("✓ Numba CUDA available for GPU kernels")
        except ImportError:
            if verbose:
                print("✗ Numba CUDA not available (kernels will use CPU)")
        
        self._initialized = True
    
    def get_array_module(self, prefer_gpu=True):
        """Get the best available array module (CuPy or NumPy)."""
        if not self._initialized:
            self.initialize(verbose=False)
            
        if prefer_gpu and self.cupy_available:
            return self.cupy
        else:
            import numpy
            return numpy
    
    def has_full_gpu_support(self):
        """Check if full GPU support (both CuPy and Numba CUDA) is available."""
        if not self._initialized:
            self.initialize(verbose=False)
        return self.cupy_available and self.numba_cuda_available
    
    def has_partial_gpu_support(self):
        """Check if any GPU support is available."""
        if not self._initialized:
            self.initialize(verbose=False)
        return self.cupy_available or self.numba_cuda_available
    
    def has_cupy(self):
        """Check if CuPy is available."""
        if not self._initialized:
            self.initialize(verbose=False)
        return self.cupy_available
    
    def has_numba_cuda(self):
        """Check if Numba CUDA is available."""
        if not self._initialized:
            self.initialize(verbose=False)
        return self.numba_cuda_available
    
    def get_gpu_status_string(self):
        """Get a human-readable string describing GPU support."""
        if not self._initialized:
            self.initialize(verbose=False)
            
        if self.has_full_gpu_support():
            return "Full GPU support (CuPy + Numba CUDA)"
        elif self.cupy_available:
            return "Partial GPU support (CuPy only - array operations accelerated)"
        elif self.numba_cuda_available:
            return "Partial GPU support (Numba CUDA only - custom kernels accelerated)"
        else:
            return "CPU only"
    
    def try_import_cupy(self, fallback_to_numpy=True):
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
    
    def try_import_numba_cuda(self, raise_on_fail=False):
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
    
    def convert_to_numpy(self, array):
        """
        Convert a GPU array to NumPy array if needed.
        
        Handles both CuPy arrays and other GPU array types gracefully.
        """
        if hasattr(array, 'get'):  # CuPy array or other GPU arrays with .get() method
            return array.get()
        elif hasattr(array, 'cpu'):  # PyTorch tensor
            return array.cpu().numpy()
        else:
            # Already a CPU array
            return array


# Global GPU support instance
_gpu_support = None

def get_gpu_support():
    """Get the global GPU support instance."""
    global _gpu_support
    if _gpu_support is None:
        _gpu_support = GPUSupport()
    return _gpu_support

def initialize_gpu_support(verbose=True):
    """Initialize GPU support and return status."""
    gpu_support = get_gpu_support()
    gpu_support.initialize(verbose=verbose)
    return gpu_support

# Convenience functions
def get_array_module(prefer_gpu=True):
    """Get the best available array module (CuPy or NumPy)."""
    return get_gpu_support().get_array_module(prefer_gpu)

def has_gpu_support():
    """Check if any GPU support is available."""
    return get_gpu_support().has_partial_gpu_support()

def has_full_gpu_support():
    """Check if full GPU support is available."""
    return get_gpu_support().has_full_gpu_support()

def try_gpu_import():
    """
    Try to import GPU libraries with graceful fallback.
    
    Returns:
        dict: Dictionary containing import results and availability flags
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