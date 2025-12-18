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
        If True, apply gentler processing to avoid over-processing
    """
    slice_norm = (slice_data - np.min(slice_data)) / (np.ptp(slice_data) + 1e-8)
    
    if gentle_mode:
        gentle_clip_limit = min(clip_limit * 2, 0.03)
        gentle_tile_size = tuple(max(s * 2, 64) for s in tile_grid_size)
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
                                           random_state=None,
                                           patch_size=None,
                                           apply_background_noise=True,
                                           background_noise_intensity=0.4):
    """
    Apply background enhancement, then Cornucopia augmentation, then contrast enhancement.
    Now with option to apply noise ONLY to background, keeping fibers clean.
    
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
    apply_background_noise : bool
        If True, apply heavy noise to background only (like dark-field microscopy)
    background_noise_intensity : float
        Intensity of background-only noise (0-1)
    
    Returns
    -------
    np.ndarray
        Enhanced and augmented slice data
    """
    # SIMPLIFIED: Keep original texture, only normalize to 0-1
    processed_slice = slice_data.copy()
    
    # Normalize to 0-1 range
    if np.ptp(processed_slice) > 0:
        processed_slice = (processed_slice - np.min(processed_slice)) / np.ptp(processed_slice)
    
    # Create mask for actual tissue vs empty/padding regions
    zero_threshold = 0.01
    tissue_mask = processed_slice > zero_threshold
    
    # SKIP aggressive preprocessing - it destroys texture
    # processed_slice = preprocess_quantized_data(processed_slice)  # DISABLED
    
    # SKIP background enhancement - it destroys texture  
    # background_enhancement disabled
    
    # Apply Cornucopia augmentation if configured (for noise effects)
    if cornucopia_augmentation is not None and CORNUCOPIA_INTEGRATION_AVAILABLE:
        if np.all(processed_slice == 0) or np.std(processed_slice) < 1e-6:
            pass
        else:
            try:
                if isinstance(cornucopia_augmentation, str):
                    cornucopia_result = augment_fiber_slice(
                        processed_slice,
                        preset=cornucopia_augmentation,
                        random_state=random_state,
                        patch_size=patch_size
                    )
                elif isinstance(cornucopia_augmentation, dict):
                    cornucopia_result = augment_fiber_slice(
                        processed_slice,
                        custom_config=cornucopia_augmentation,
                        random_state=random_state,
                        patch_size=patch_size
                    )
                
                if not (np.all(cornucopia_result == 0) or np.std(cornucopia_result) < 1e-6 or np.max(cornucopia_result) < 0.01):
                    processed_slice = cornucopia_result
            except Exception:
                pass
    
    # SIMPLIFIED: Light contrast enhancement only, preserve texture
    if contrast_params is None:
        contrast_params = {}
    
    # Use very gentle CLAHE to preserve texture
    enhanced_slice = apply_contrast_enhancement(
        processed_slice, 
        clip_limit=contrast_params.get('clip_limit', 0.005),  # Very low clip limit
        tile_grid_size=contrast_params.get('tile_grid_size', (64, 64)),  # Large tiles
        gentle_mode=True  # Always gentle
    )
    
    # Apply tissue mask to ensure empty regions stay black
    enhanced_slice = enhanced_slice * tissue_mask
    
    return enhanced_slice


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
                                       preprocess_quantized=True,
                                       patch_size=None,
                                       apply_background_noise=True,
                                       background_noise_intensity=0.5):
    """
    Apply comprehensive slice processing with all available enhancement methods.
    Now with background-only noise option for dark-field microscopy appearance.
    
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
    apply_background_noise : bool
        Apply heavy noise to background only (keeps fibers clean)
    background_noise_intensity : float
        Intensity of background noise (0-1), default 0.5 for heavy speckle
    
    Returns
    -------
    np.ndarray
        Fully processed slice data
    """
    if background_params is not None:
        background_config = background_params
    else:
        background_config = background_preset if BACKGROUND_ENHANCEMENT_AVAILABLE else None
    
    if cornucopia_params is not None:
        cornucopia_config = cornucopia_params
    else:
        cornucopia_config = cornucopia_preset if CORNUCOPIA_INTEGRATION_AVAILABLE else None
    
    return apply_enhanced_contrast_and_augmentation(
        slice_data,
        contrast_method=contrast_method,
        contrast_params=contrast_params,
        cornucopia_augmentation=cornucopia_config,
        background_enhancement=background_config,
        enable_sharpening=enable_sharpening,
        sharpening_strength=sharpening_strength,
        random_state=random_state,
        patch_size=patch_size,
        apply_background_noise=apply_background_noise,
        background_noise_intensity=background_noise_intensity
    ) 