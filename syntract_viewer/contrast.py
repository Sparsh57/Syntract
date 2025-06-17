"""
Contrast enhancement functions for NIfTI data visualization.
"""

import numpy as np
import warnings
from skimage import exposure
from scipy import ndimage

# Import Cornucopia integration with fallback
try:
    from .improved_cornucopia import augment_fiber_slice
    CORNUCOPIA_INTEGRATION_AVAILABLE = True
except ImportError:
    try:
        from improved_cornucopia import augment_fiber_slice
        CORNUCOPIA_INTEGRATION_AVAILABLE = True
    except ImportError:
        CORNUCOPIA_INTEGRATION_AVAILABLE = False

# Import background enhancement with fallback
try:
    from .background_enhancement import enhance_slice_background
    BACKGROUND_ENHANCEMENT_AVAILABLE = True
except ImportError:
    try:
        from background_enhancement import enhance_slice_background
        BACKGROUND_ENHANCEMENT_AVAILABLE = True
    except ImportError:
        BACKGROUND_ENHANCEMENT_AVAILABLE = False


def apply_contrast_enhancement(slice_data, clip_limit=0.01, tile_grid_size=(8, 8), gentle_mode=False):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a slice.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data
    clip_limit : float
        CLAHE clip limit
    tile_grid_size : tuple
        CLAHE tile grid size
    gentle_mode : bool
        If True, apply gentler processing to avoid over-processing already smoothed data
    """
    slice_norm = (slice_data - np.min(slice_data)) / (np.ptp(slice_data) + 1e-8)
    
    if gentle_mode:
        # For already-processed quantized data, use gentler CLAHE
        gentle_clip_limit = min(clip_limit * 2, 0.03)  # Slightly higher clip limit
        gentle_tile_size = tuple(max(s * 2, 64) for s in tile_grid_size)  # Larger tiles
        enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=gentle_clip_limit, kernel_size=gentle_tile_size)
    else:
        enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit, kernel_size=tile_grid_size)
    
    return enhanced


def apply_enhanced_contrast_and_augmentation(slice_data, 
                                           contrast_method='clahe',
                                           contrast_params=None,
                                           cornucopia_augmentation=None,
                                           background_enhancement=None,
                                           enable_sharpening=True,
                                           sharpening_strength=0.5,
                                           random_state=None):
    """
    Apply background enhancement FIRST, then Cornucopia augmentation, then contrast enhancement.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data
    contrast_method : str
        Contrast enhancement method
    contrast_params : dict, optional
        Parameters for contrast enhancement
    cornucopia_augmentation : str or dict, optional
        Cornucopia augmentation configuration
    background_enhancement : str or dict, optional
        Background enhancement configuration
    random_state : int, optional
        Random seed for reproducible results
    
    Returns
    -------
    np.ndarray
        Enhanced and augmented slice data
    """
    # Start with the original slice
    processed_slice = slice_data.copy()
    
    # Check if this is quantized data before preprocessing
    original_unique_count = len(np.unique(slice_data[slice_data > 0]))
    was_quantized = original_unique_count < 200
    
    # Preprocess quantized data to reduce tiling artifacts
    processed_slice = preprocess_quantized_data(processed_slice)
    
    # Step 1: Apply background enhancement to reduce pixelation
    if background_enhancement is not None and BACKGROUND_ENHANCEMENT_AVAILABLE:
        if np.all(slice_data == 0):
            print("   ⚠️  Input slice is all zeros, skipping background enhancement")
        elif np.std(slice_data) < 1e-6:
            print("   ⚠️  Input slice has no variance, skipping background enhancement")
        else:
            try:
                if isinstance(background_enhancement, str):
                    # Use preset with optional sharpening
                    processed_slice = enhance_slice_background(
                        processed_slice,
                        preset=background_enhancement,
                        apply_sharpening=enable_sharpening,
                        sharpening_params={'radius': 0.8, 'amount': sharpening_strength} if enable_sharpening else None,
                        random_state=random_state
                    )
                elif isinstance(background_enhancement, dict):
                    # Use custom configuration with optional sharpening
                    processed_slice = enhance_slice_background(
                        processed_slice,
                        apply_sharpening=enable_sharpening,
                        sharpening_params={'radius': 0.8, 'amount': sharpening_strength} if enable_sharpening else None,
                        random_state=random_state,
                        **background_enhancement
                    )
                
                # Safety checks
                if np.any(np.isnan(processed_slice)) or np.any(np.isinf(processed_slice)):
                    print(f"      ⚠️  Background enhancement produced invalid values, using original")
                    processed_slice = slice_data.copy()
                elif np.std(processed_slice) < 1e-6:
                    print(f"      ⚠️  Background enhancement removed all variance, using original")
                    processed_slice = slice_data.copy()
                else:
                    print(f"   ✅ Applied background enhancement '{background_enhancement}' to reduce pixelation")
            except Exception as e:
                print(f"   ⚠️  Background enhancement failed: {e}, using original slice")
                processed_slice = slice_data.copy()
    
    # Step 2: Apply Cornucopia augmentation
    if cornucopia_augmentation is not None and CORNUCOPIA_INTEGRATION_AVAILABLE:
        if np.all(processed_slice == 0):
            print("   ⚠️  Processed slice is all zeros, skipping Cornucopia augmentation")
        elif np.std(processed_slice) < 1e-6:
            print("   ⚠️  Processed slice has no variance, skipping Cornucopia augmentation")
        else:
            try:
                if isinstance(cornucopia_augmentation, str):
                    cornucopia_result = augment_fiber_slice(
                        processed_slice,
                        preset=cornucopia_augmentation,
                        random_state=random_state
                    )
                elif isinstance(cornucopia_augmentation, dict):
                    cornucopia_result = augment_fiber_slice(
                        processed_slice,
                        custom_config=cornucopia_augmentation,
                        random_state=random_state
                    )
                
                # Safety checks
                if np.all(cornucopia_result == 0):
                    print(f"      ⚠️  Cornucopia augmentation made slice all zeros, using background-enhanced slice")
                elif np.std(cornucopia_result) < 1e-6:
                    print(f"      ⚠️  Cornucopia augmentation removed all variance, using background-enhanced slice")
                elif np.max(cornucopia_result) < 0.01:
                    print(f"      ⚠️  Cornucopia augmentation made slice too dark, using background-enhanced slice")
                else:
                    processed_slice = cornucopia_result
                    print(f"   ✅ Applied Cornucopia '{cornucopia_augmentation}' to background-enhanced slice")
            except Exception as e:
                print(f"   ⚠️  Cornucopia augmentation failed: {e}, using background-enhanced slice")
    
    # Step 3: Apply contrast enhancement with gentle mode for pre-processed quantized data
    if contrast_params is None:
        contrast_params = {}
    
    # Use gentle mode if this was originally quantized data that has been aggressively preprocessed
    gentle_mode = was_quantized and len(np.unique(processed_slice[processed_slice > 0])) > original_unique_count * 10
    
    enhanced_slice = apply_contrast_enhancement(
        processed_slice, 
        clip_limit=contrast_params.get('clip_limit', 0.01),
        tile_grid_size=contrast_params.get('tile_grid_size', (8, 8)),
        gentle_mode=gentle_mode
    )
    
    if gentle_mode:
        print(f"   ✅ Applied gentle CLAHE processing for ultra-aggressive preprocessed data")
    
    return enhanced_slice




def preprocess_quantized_data(slice_data, smooth_sigma=2.0, upscale_factor=2.0, aggressive=True):
    """
    Preprocess heavily quantized data to reduce tiling artifacts using ultra-aggressive approach.
    
    This function addresses the issue where data has very few unique values
    causing visible "blocks" or "tiling" patterns in the visualization.
    
    Based on the ultra-aggressive preprocessing that eliminates tiling while preserving quality.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data
    smooth_sigma : float
        Initial Gaussian smoothing sigma
    upscale_factor : float
        Initial upscale factor for bicubic resampling
    aggressive : bool
        Whether to apply ultra-aggressive multi-step processing
    """
    unique_count = len(np.unique(slice_data[slice_data > 0]))

    if unique_count < 200:  # Detect quantized data
        print(f"   ⚠️  Quantized data detected ({unique_count} unique values), applying tear-free smoothing")

        # Step 1: Convert to float for processing
        processed = slice_data.astype(np.float64)

        if aggressive:
            # OPTIMIZED TILING-ELIMINATION PROCESSING (TEAR-FREE VERSION)
            # Reduced aggressive operations to prevent tears while maintaining smoothness

            # Step 2: Gentle progressive Gaussian smoothing (reduced passes)
            for i in range(2):  # Reduced from 3 to 2 passes to minimize artifacts
                sigma = smooth_sigma + i * 0.2  # Smaller increments to reduce discontinuities
                processed = ndimage.gaussian_filter(processed, sigma=sigma)

            # Step 3: TEAR-FREE upscale/downscale cycles (modified approach)
            # Use smaller factors and better interpolation to prevent tears
            for i in range(1):  # Reduced to 1 cycle to minimize interpolation artifacts
                factor = 1.5 + i * 0.1  # Much smaller factors (1.5 instead of 2.0+)
                
                # Use order=2 (quadratic) instead of order=3 (cubic) for less overshoot
                # This reduces the "ringing" artifacts that can appear as tears
                upscaled = ndimage.zoom(processed, factor, order=2, prefilter=True)
                processed = ndimage.zoom(upscaled, 1.0/factor, order=2, prefilter=True)
                
                # Apply gentle smoothing after each cycle to eliminate any remaining artifacts
                processed = ndimage.gaussian_filter(processed, sigma=0.3)

            # Step 4: Additional tear-prevention smoothing
            # Apply very gentle edge-preserving smoothing
            smoothed = ndimage.gaussian_filter(processed, sigma=0.5)  # Reduced from 0.8
            blend_factor = 0.3  # Reduced from 0.5 for less blur
            processed = (1 - blend_factor) * processed + blend_factor * smoothed

        else:
            # GENTLE PROCESSING (original)
            processed = ndimage.gaussian_filter(processed, sigma=smooth_sigma)

            if upscale_factor > 1.0:
                # Use gentler interpolation for non-aggressive mode too
                upscaled = ndimage.zoom(processed, upscale_factor, order=2, prefilter=True)
                processed = ndimage.zoom(upscaled, 1.0/upscale_factor, order=2, prefilter=True)

        # Step 5: Preserve original value range
        if np.max(slice_data) > 0:
            processed = processed * (np.max(slice_data) / np.max(processed))

        final_unique = len(np.unique(processed[processed > 0]))
        print(f"   ✅ Tear-free quantization processing complete: {unique_count} → {final_unique} unique values")
        
        return processed.astype(slice_data.dtype)
    else:
        # Data is not heavily quantized, return as-is
        return slice_data

def apply_comprehensive_slice_processing(slice_data,
                                       background_preset=None,
                                       cornucopia_preset=None,
                                       contrast_method='clahe',
                                       background_params=None,
                                       cornucopia_params=None,
                                       contrast_params=None,
                                       enable_sharpening=True,
                                       sharpening_strength=0.5,
                                       random_state=None,
                                       preprocess_quantized=True):
    """
    Apply comprehensive slice processing with all available enhancement methods.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data
    background_preset : str
        Background enhancement preset
    cornucopia_preset : str
        Cornucopia augmentation preset
    contrast_method : str
        Contrast enhancement method
    background_params : dict, optional
        Custom background enhancement parameters
    cornucopia_params : dict, optional
        Custom Cornucopia parameters
    contrast_params : dict, optional
        Custom contrast parameters
    random_state : int, optional
        Random seed for reproducible results
    
    Returns
    -------
    np.ndarray
        Fully processed slice data
    """
    # Prepare background enhancement configuration
    if background_params is not None:
        background_config = background_params
    else:
        background_config = background_preset if BACKGROUND_ENHANCEMENT_AVAILABLE else None
    
    # Prepare Cornucopia configuration
    if cornucopia_params is not None:
        cornucopia_config = cornucopia_params
    else:
        cornucopia_config = cornucopia_preset if CORNUCOPIA_INTEGRATION_AVAILABLE else None
    
    # Apply comprehensive processing
    return apply_enhanced_contrast_and_augmentation(
        slice_data,
        contrast_method=contrast_method,
        contrast_params=contrast_params,
        cornucopia_augmentation=cornucopia_config,
        background_enhancement=background_config,
        enable_sharpening=enable_sharpening,
        sharpening_strength=sharpening_strength,
        random_state=random_state
    ) 