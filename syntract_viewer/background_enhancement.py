"""
Background enhancement functions for improving slice appearance before fiber overlay.

This module provides various techniques to make thin NIfTI slices appear smoother and more realistic,
addressing pixelation issues that can occur with dimensions like (500, 1, 500).
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union
import random

# Import OpenCV with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Install with: pip install opencv-python")

# Import PIL with fallback
try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available. Install with: pip install Pillow")

# Import scipy and skimage
try:
    from scipy import ndimage, interpolate
    from skimage import filters, restoration, morphology, transform
    SCIPY_SKIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_SKIMAGE_AVAILABLE = False
    warnings.warn("Scipy/scikit-image not available. Limited functionality.")


def enhance_background_smoothness(slice_data: np.ndarray, 
                                method: str = 'multi_scale_bicubic',
                                enhancement_params: Optional[dict] = None,
                                random_state: Optional[int] = None,
                                **kwargs) -> np.ndarray:
    """
    Enhance background smoothness to reduce pixelation artifacts.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data (2D array)
    method : str
        Enhancement method to use
    enhancement_params : dict, optional
        Parameters for the enhancement method
    random_state : int, optional
        Random seed for reproducible results
    **kwargs : dict
        Additional parameters (including 'methods' for combined enhancement)
    
    Returns
    -------
    np.ndarray
        Enhanced slice data
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    if enhancement_params is None:
        enhancement_params = {}
    
    # Merge kwargs into enhancement_params
    all_params = enhancement_params.copy()
    all_params.update(kwargs)
    
    # Choose enhancement method
    if method == 'multi_scale_bicubic':
        return _apply_multi_scale_bicubic_enhancement(slice_data, **all_params)
    elif method == 'opencv_super_resolution':
        return _apply_opencv_super_resolution(slice_data, **all_params)
    elif method == 'adaptive_smoothing':
        return _apply_adaptive_smoothing(slice_data, **all_params)
    elif method == 'edge_preserving_filter':
        return _apply_edge_preserving_filter(slice_data, **all_params)
    elif method == 'anisotropic_diffusion':
        return _apply_anisotropic_diffusion(slice_data, **all_params)
    elif method == 'texture_synthesis':
        return _apply_texture_synthesis(slice_data, **all_params)
    elif method == 'combined_enhancement':
        return _apply_combined_enhancement(slice_data, **all_params)
    else:
        # Fallback to basic method
        return _apply_basic_smoothing(slice_data, **all_params)


def _apply_multi_scale_bicubic_enhancement(slice_data: np.ndarray, 
                                         scale_factor: float = 2.0,
                                         iterations: int = 2,
                                         gaussian_sigma: float = 0.5,
                                         **kwargs) -> np.ndarray:
    """
    Apply multi-scale bicubic interpolation for smooth upsampling and downsampling.
    """
    enhanced = slice_data.copy()
    
    if not SCIPY_SKIMAGE_AVAILABLE:
        return _apply_basic_smoothing(enhanced)
    
    # Multi-scale enhancement
    for i in range(iterations):
        # Upscale using bicubic interpolation
        upscaled_shape = (int(enhanced.shape[0] * scale_factor), 
                         int(enhanced.shape[1] * scale_factor))
        
        upscaled = transform.resize(enhanced, upscaled_shape, order=3, 
                                  anti_aliasing=True, preserve_range=True)
        
        # Apply gentle smoothing
        upscaled = filters.gaussian(upscaled, sigma=gaussian_sigma * (i + 1))
        
        # Downscale back to original size
        enhanced = transform.resize(upscaled, slice_data.shape, order=3,
                                  anti_aliasing=True, preserve_range=True)
    
    return enhanced.astype(slice_data.dtype)


def _apply_opencv_super_resolution(slice_data: np.ndarray,
                                 sr_scale: int = 2,
                                 method: str = 'bilateral',
                                 **kwargs) -> np.ndarray:
    """
    Apply OpenCV-based super-resolution techniques.
    """
    if not CV2_AVAILABLE:
        return _apply_basic_smoothing(slice_data)
    
    try:
        # Normalize to 0-255 for OpenCV
        normalized = ((slice_data - np.min(slice_data)) / 
                     (np.ptp(slice_data) + 1e-8) * 255).astype(np.uint8)
        
        # Apply different OpenCV enhancement methods
        if method == 'bilateral':
            # Bilateral filtering for edge-preserving smoothing
            enhanced = cv2.bilateralFilter(normalized, 9, 75, 75)
        elif method == 'guided':
            # Guided filter (if available)
            enhanced = cv2.ximgproc.guidedFilter(normalized, normalized, 8, 0.01) if hasattr(cv2, 'ximgproc') else normalized
        elif method == 'non_local_means':
            # Non-local means denoising
            enhanced = cv2.fastNlMeansDenoising(normalized, None, 10, 7, 21)
        else:
            # Default: morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            enhanced = cv2.morphologyEx(normalized, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        
        # Convert back to original range
        enhanced = enhanced.astype(np.float32) / 255.0
        original_range = np.ptp(slice_data)
        original_min = np.min(slice_data)
        enhanced = enhanced * original_range + original_min
        
        return enhanced.astype(slice_data.dtype)
        
    except Exception as e:
        warnings.warn(f"OpenCV enhancement failed: {e}")
        return _apply_basic_smoothing(slice_data)


def _apply_adaptive_smoothing(slice_data: np.ndarray,
                            edge_threshold: float = 0.1,
                            smooth_sigma: float = 1.0,
                            edge_sigma: float = 0.3,
                            **kwargs) -> np.ndarray:
    """
    Apply adaptive smoothing that preserves edges while smoothing flat regions.
    """
    if not SCIPY_SKIMAGE_AVAILABLE:
        return _apply_basic_smoothing(slice_data)
    
    # Detect edges
    edges = filters.sobel(slice_data)
    edge_mask = edges > edge_threshold
    
    # Apply different smoothing to edges vs flat regions
    smooth_flat = filters.gaussian(slice_data, sigma=smooth_sigma)
    smooth_edges = filters.gaussian(slice_data, sigma=edge_sigma)
    
    # Combine based on edge mask
    enhanced = np.where(edge_mask, smooth_edges, smooth_flat)
    
    return enhanced.astype(slice_data.dtype)


def _apply_edge_preserving_filter(slice_data: np.ndarray,
                                sigma_color: float = 0.1,
                                sigma_spatial: float = 1.5,
                                **kwargs) -> np.ndarray:
    """
    Apply edge-preserving filters to smooth while maintaining structure.
    """
    if not CV2_AVAILABLE:
        return _apply_adaptive_smoothing(slice_data)
    
    try:
        # Normalize for OpenCV
        normalized = ((slice_data - np.min(slice_data)) / 
                     (np.ptp(slice_data) + 1e-8) * 255).astype(np.uint8)
        
        # Apply edge-preserving filter
        enhanced = cv2.edgePreservingFilter(normalized, flags=1, 
                                          sigma_s=sigma_spatial, sigma_r=sigma_color)
        
        # Convert back
        enhanced = enhanced.astype(np.float32) / 255.0
        original_range = np.ptp(slice_data)
        original_min = np.min(slice_data)
        enhanced = enhanced * original_range + original_min
        
        return enhanced.astype(slice_data.dtype)
        
    except Exception as e:
        warnings.warn(f"Edge-preserving filter failed: {e}")
        return _apply_adaptive_smoothing(slice_data)


def _apply_anisotropic_diffusion(slice_data: np.ndarray,
                               num_iterations: int = 20,
                               kappa: float = 20,
                               gamma: float = 0.1,
                               **kwargs) -> np.ndarray:
    """
    Apply anisotropic diffusion for selective smoothing.
    """
    if not SCIPY_SKIMAGE_AVAILABLE:
        return _apply_basic_smoothing(slice_data)
    
    try:
        # Use restoration module if available
        enhanced = restoration.denoise_tv_chambolle(slice_data, weight=0.1)
        return enhanced.astype(slice_data.dtype)
    except:
        # Fallback to manual anisotropic diffusion
        return _manual_anisotropic_diffusion(slice_data, num_iterations, kappa, gamma)


def _manual_anisotropic_diffusion(slice_data: np.ndarray,
                                num_iterations: int = 20,
                                kappa: float = 20,
                                gamma: float = 0.1) -> np.ndarray:
    """
    Manual implementation of anisotropic diffusion.
    """
    enhanced = slice_data.astype(np.float32)
    
    for _ in range(num_iterations):
        # Calculate gradients
        grad_n = np.roll(enhanced, -1, axis=0) - enhanced
        grad_s = np.roll(enhanced, 1, axis=0) - enhanced
        grad_e = np.roll(enhanced, -1, axis=1) - enhanced
        grad_w = np.roll(enhanced, 1, axis=1) - enhanced
        
        # Calculate conductance coefficients
        c_n = np.exp(-(grad_n / kappa) ** 2)
        c_s = np.exp(-(grad_s / kappa) ** 2)
        c_e = np.exp(-(grad_e / kappa) ** 2)
        c_w = np.exp(-(grad_w / kappa) ** 2)
        
        # Update
        enhanced += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
    
    return enhanced.astype(slice_data.dtype)


def _apply_texture_synthesis(slice_data: np.ndarray,
                           texture_scale: float = 0.02,
                           blend_factor: float = 0.3,
                           **kwargs) -> np.ndarray:
    """
    Add subtle texture to make the background more realistic.
    """
    enhanced = slice_data.copy()
    
    # Create subtle texture pattern
    h, w = slice_data.shape
    x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h), indexing='xy')
    
    # Multi-frequency texture
    texture = (np.sin(x * 2.1) * np.cos(y * 1.7) * 0.3 +
              np.sin(x * 5.3) * np.cos(y * 4.2) * 0.2 +
              np.random.normal(0, 0.1, (h, w)))
    
    texture *= texture_scale
    
    # Only add texture to non-zero regions
    brain_mask = enhanced > np.percentile(enhanced[enhanced > 0], 5) if np.any(enhanced > 0) else enhanced > 0
    
    if np.any(brain_mask):
        enhanced[brain_mask] += texture[brain_mask] * blend_factor
        enhanced = np.clip(enhanced, 0, np.max(slice_data))
    
    return enhanced


def _apply_combined_enhancement(slice_data: np.ndarray,
                              methods: Optional[list] = None,
                              **kwargs) -> np.ndarray:
    """
    Apply a combination of enhancement methods for optimal results.
    """
    if methods is None:
        methods = ['multi_scale_bicubic', 'adaptive_smoothing', 'texture_synthesis']
    
    enhanced = slice_data.copy()
    
    # Apply methods sequentially with decreasing weights
    weights = np.linspace(1.0, 0.3, len(methods))
    
    for i, method in enumerate(methods):
        try:
            if method == 'multi_scale_bicubic':
                method_result = _apply_multi_scale_bicubic_enhancement(enhanced, **kwargs)
            elif method == 'adaptive_smoothing':
                method_result = _apply_adaptive_smoothing(enhanced, **kwargs)
            elif method == 'edge_preserving_filter':
                method_result = _apply_edge_preserving_filter(enhanced, **kwargs)
            elif method == 'texture_synthesis':
                method_result = _apply_texture_synthesis(enhanced, **kwargs)
            else:
                continue
            
            # Blend with original
            enhanced = enhanced * (1 - weights[i] * 0.5) + method_result * (weights[i] * 0.5)
            
        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")
            continue
    
    return enhanced.astype(slice_data.dtype)


def _apply_basic_smoothing(slice_data: np.ndarray,
                         sigma: float = 1.0,
                         **kwargs) -> np.ndarray:
    """
    Basic smoothing fallback when other methods are not available.
    """
    if SCIPY_SKIMAGE_AVAILABLE:
        enhanced = filters.gaussian(slice_data, sigma=sigma)
    else:
        # Very basic smoothing using numpy
        kernel_size = int(sigma * 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Simple box filter
        enhanced = slice_data.copy()
        for i in range(kernel_size//2, slice_data.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, slice_data.shape[1] - kernel_size//2):
                enhanced[i, j] = np.mean(
                    slice_data[i-kernel_size//2:i+kernel_size//2+1, 
                              j-kernel_size//2:j+kernel_size//2+1]
                )
    
    return enhanced.astype(slice_data.dtype)


def apply_smart_sharpening(slice_data: np.ndarray,
                          method: str = 'unsharp_mask',
                          sharpening_params: Optional[dict] = None) -> np.ndarray:
    """
    Apply intelligent sharpening to enhance details without amplifying noise.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data
    method : str
        Sharpening method ('unsharp_mask', 'laplacian', 'high_pass')
    sharpening_params : dict, optional
        Parameters for sharpening
    
    Returns
    -------
    np.ndarray
        Sharpened slice data
    """
    if sharpening_params is None:
        sharpening_params = {}
    
    if method == 'unsharp_mask':
        return _apply_unsharp_mask(slice_data, **sharpening_params)
    elif method == 'laplacian':
        return _apply_laplacian_sharpening(slice_data, **sharpening_params)
    elif method == 'high_pass':
        return _apply_high_pass_sharpening(slice_data, **sharpening_params)
    else:
        return slice_data


def _apply_unsharp_mask(slice_data: np.ndarray,
                       radius: float = 1.0,
                       amount: float = 0.5,
                       threshold: float = 0.0,
                       **kwargs) -> np.ndarray:
    """
    Apply unsharp masking for controlled sharpening.
    """
    if SCIPY_SKIMAGE_AVAILABLE:
        # Create blurred version
        blurred = filters.gaussian(slice_data, sigma=radius)
        
        # Calculate high-frequency component
        high_freq = slice_data - blurred
        
        # Apply threshold
        if threshold > 0:
            high_freq = np.where(np.abs(high_freq) > threshold, high_freq, 0)
        
        # Add back with amount control
        sharpened = slice_data + amount * high_freq
        
        return np.clip(sharpened, np.min(slice_data), np.max(slice_data))
    else:
        return slice_data


def _apply_laplacian_sharpening(slice_data: np.ndarray,
                              strength: float = 0.3,
                              **kwargs) -> np.ndarray:
    """
    Apply Laplacian-based sharpening.
    """
    if SCIPY_SKIMAGE_AVAILABLE:
        # Laplacian kernel
        laplacian = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        
        # Apply convolution
        high_freq = ndimage.convolve(slice_data, laplacian, mode='reflect')
        
        # Add to original
        sharpened = slice_data + strength * high_freq
        
        return np.clip(sharpened, np.min(slice_data), np.max(slice_data))
    else:
        return slice_data


def _apply_high_pass_sharpening(slice_data: np.ndarray,
                              cutoff_frequency: float = 0.1,
                              strength: float = 0.4,
                              **kwargs) -> np.ndarray:
    """
    Apply frequency-domain high-pass sharpening.
    """
    try:
        # FFT-based high-pass filter
        f_transform = np.fft.fft2(slice_data)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter
        rows, cols = slice_data.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        y, x = np.ogrid[:rows, :cols]
        mask = np.sqrt((x - ccol)**2 + (y - crow)**2) > cutoff_frequency * min(rows, cols)
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        high_freq = np.real(np.fft.ifft2(f_ishift))
        
        # Combine with original
        sharpened = slice_data + strength * high_freq
        
        return np.clip(sharpened, np.min(slice_data), np.max(slice_data))
        
    except Exception as e:
        warnings.warn(f"High-pass sharpening failed: {e}")
        return slice_data


def create_enhancement_presets() -> dict:
    """
    Create predefined enhancement preset configurations.
    
    Returns
    -------
    dict
        Dictionary of enhancement preset configurations
    """
    return {
        'smooth_realistic': {
            'method': 'combined_enhancement',
            'methods': ['multi_scale_bicubic', 'adaptive_smoothing'],
            'scale_factor': 1.5,
            'smooth_sigma': 0.8,
            'edge_sigma': 0.3
        },
        
        'high_quality': {
            'method': 'combined_enhancement', 
            'methods': ['multi_scale_bicubic', 'edge_preserving_filter', 'texture_synthesis'],
            'scale_factor': 2.0,
            'iterations': 3,
            'sigma_color': 0.1,
            'sigma_spatial': 2.0,
            'texture_scale': 0.015
        },
        
        'clinical_appearance': {
            'method': 'opencv_super_resolution',
            'sr_scale': 2,
            'method': 'bilateral'
        },
        
        'preserve_edges': {
            'method': 'edge_preserving_filter',
            'sigma_color': 0.05,
            'sigma_spatial': 1.0
        },
        
        'subtle_enhancement': {
            'method': 'adaptive_smoothing',
            'edge_threshold': 0.05,
            'smooth_sigma': 0.5,
            'edge_sigma': 0.2
        }
    }


def enhance_slice_background(slice_data: np.ndarray,
                           preset: str = 'smooth_realistic',
                           custom_params: Optional[dict] = None,
                           apply_sharpening: bool = False,
                           sharpening_method: str = 'unsharp_mask',
                           sharpening_params: Optional[dict] = None,
                           random_state: Optional[int] = None) -> np.ndarray:
    """
    Main function to enhance slice background appearance.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data (2D array)
    preset : str
        Enhancement preset name
    custom_params : dict, optional
        Custom parameters to override preset
    apply_sharpening : bool
        Whether to apply sharpening after enhancement
    sharpening_method : str
        Sharpening method to use
    sharpening_params : dict, optional
        Parameters for sharpening
    random_state : int, optional
        Random seed for reproducible results
    
    Returns
    -------
    np.ndarray
        Enhanced slice data
    """
    # Get preset parameters
    presets = create_enhancement_presets()
    if preset in presets:
        enhancement_params = presets[preset].copy()
    else:
        enhancement_params = presets['smooth_realistic'].copy()
        warnings.warn(f"Unknown preset '{preset}', using 'smooth_realistic'")
    
    # Override with custom parameters
    if custom_params:
        enhancement_params.update(custom_params)
    
    # Apply enhancement
    enhanced = enhance_background_smoothness(
        slice_data, 
        **enhancement_params,
        random_state=random_state
    )
    
    # Apply sharpening if requested
    if apply_sharpening:
        if sharpening_params is None:
            sharpening_params = {'radius': 0.8, 'amount': 0.3}

        enhanced = apply_smart_sharpening(
            enhanced,
            method=sharpening_method,
            sharpening_params=sharpening_params
        )
    
    return enhanced 