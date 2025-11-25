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


def apply_background_only_noise(slice_data, noise_intensity=0.3, fiber_threshold_percentile=70, random_state=None, noise_probability=0.7):
    """
    Apply heavy noise ONLY to background regions, keeping bright fibers clean.
    This mimics dark-field microscopy where fibers are bright and background is noisy.
    Noise application is now randomized - some images get heavy noise, some light, some none.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data (normalized to 0-1)
    noise_intensity : float
        Intensity of noise to apply to background (0-1)
    fiber_threshold_percentile : float
        Percentile threshold to identify bright fiber regions (e.g., 70 = top 30% are fibers)
    random_state : int, optional
        Random seed for reproducibility
    noise_probability : float
        Probability of applying noise (0-1). Default 0.7 means 70% of images get noise.
    
    Returns
    -------
    np.ndarray
        Slice with noisy background but clean fibers (or clean background if noise skipped)
    """
    import random
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    result = slice_data.copy()
    
    # Randomly decide whether to apply noise at all
    apply_noise = random.random() < noise_probability
    
    if not apply_noise:
        # Return clean image (no noise)
        return result
    
    # Randomly vary the noise intensity for this image (100% get noise, but varying levels)
    # Options: very light (10-30%), light (30-50%), moderate (50-70%), heavy (70-120%)
    # Good distribution: mostly subtle, some moderate, occasional heavy like the slide
    noise_level = random.choices(
        ['very_light', 'light', 'moderate', 'heavy'],
        weights=[0.30, 0.35, 0.25, 0.10]  # 30% very light, 35% light, 25% moderate, 10% heavy
    )[0]
    
    if noise_level == 'very_light':
        actual_intensity = noise_intensity * random.uniform(0.1, 0.3)
    elif noise_level == 'light':
        actual_intensity = noise_intensity * random.uniform(0.3, 0.5)
    elif noise_level == 'moderate':
        actual_intensity = noise_intensity * random.uniform(0.5, 0.8)
    else:  # heavy
        actual_intensity = noise_intensity * random.uniform(0.8, 1.2)
    
    # Identify fiber regions (bright areas)
    if np.any(slice_data > 0):
        fiber_threshold = np.percentile(slice_data[slice_data > 0], fiber_threshold_percentile)
    else:
        fiber_threshold = 0.5
    
    # Create fiber mask (bright areas should stay clean)
    fiber_mask = slice_data > fiber_threshold
    
    # Create background mask (dark areas should get noise)
    background_mask = ~fiber_mask
    
    # Apply noise to background only
    if np.any(background_mask):
        # Generate multiple types of noise for realistic speckle
        h, w = slice_data.shape
        
        # Randomly choose which noise types to apply
        noise_types = []
        if random.random() < 0.8:  # 80% chance
            noise_types.append('gaussian')
        if random.random() < 0.85:  # 85% chance - INCREASED for gamma speckle you like!
            noise_types.append('gamma')
        if random.random() < 0.3 and noise_level in ['moderate', 'heavy']:  # 30% chance for moderate/heavy only
            noise_types.append('salt_pepper')
        
        # Apply selected noise types
        if 'gaussian' in noise_types:
            # 1. Gaussian mixture noise (most visible)
            gaussian_noise = np.random.normal(0, actual_intensity * 0.5, (h, w))
            result[background_mask] = result[background_mask] + gaussian_noise[background_mask]
        
        if 'gamma' in noise_types:
            # 2. Gamma multiplicative noise (speckle pattern)
            gamma_noise = np.random.gamma(1.0, actual_intensity * 0.3, (h, w))
            combined_noise = (gamma_noise - 1.0) * result
            result[background_mask] = result[background_mask] + combined_noise[background_mask]
        
        if 'salt_pepper' in noise_types:
            # 3. Salt and pepper noise (bright spots) - only for heavier noise
            salt_pepper = np.random.random((h, w))
            salt_mask = salt_pepper > (1.0 - actual_intensity * 0.05)
            pepper_mask = salt_pepper < (actual_intensity * 0.05)
            result[background_mask & salt_mask] = np.maximum(result[background_mask & salt_mask], 0.9)
            result[background_mask & pepper_mask] = np.minimum(result[background_mask & pepper_mask], 0.1)
        
        # Clip to valid range
        result = np.clip(result, 0, 1)
    
    return result


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


def preprocess_quantized_data(slice_data, smooth_sigma=2.0, upscale_factor=2.0, aggressive=True):
    """
    Preprocess heavily quantized data to reduce tiling artifacts.
    
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

    if unique_count < 200:
        processed = slice_data.astype(np.float64)

        if aggressive:
            for i in range(2):
                sigma = smooth_sigma + i * 0.2
                processed = ndimage.gaussian_filter(processed, sigma=sigma)

            for i in range(1):
                factor = 1.5 + i * 0.1
                upscaled = ndimage.zoom(processed, factor, order=2, prefilter=True)
                processed = ndimage.zoom(upscaled, 1.0/factor, order=2, prefilter=True)
                processed = ndimage.gaussian_filter(processed, sigma=0.3)

            smoothed = ndimage.gaussian_filter(processed, sigma=0.5)
            blend_factor = 0.3
            processed = (1 - blend_factor) * processed + blend_factor * smoothed

        else:
            processed = ndimage.gaussian_filter(processed, sigma=smooth_sigma)

            if upscale_factor > 1.0:
                upscaled = ndimage.zoom(processed, upscale_factor, order=2, prefilter=True)
                processed = ndimage.zoom(upscaled, 1.0/upscale_factor, order=2, prefilter=True)

        if np.max(slice_data) > 0:
            processed = processed * (np.max(slice_data) / np.max(processed))
        
        return processed.astype(slice_data.dtype)
    else:
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