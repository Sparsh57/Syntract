"""
Contrast enhancement functions for NIfTI data visualization.
"""

import numpy as np
import warnings
from skimage import exposure

# Import Cornucopia integration with fallback
try:
    from .cornucopia_augmentation import augment_fiber_slice
    CORNUCOPIA_INTEGRATION_AVAILABLE = True
except ImportError:
    CORNUCOPIA_INTEGRATION_AVAILABLE = False
    warnings.warn(
        "Cornucopia integration module not found. "
        "Some advanced augmentation features will be unavailable."
    )


def apply_contrast_enhancement(slice_data, clip_limit=0.01, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a slice.
    """
    slice_norm = (slice_data - np.min(slice_data)) / (np.ptp(slice_data) + 1e-8)
    enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit, kernel_size=tile_grid_size)
    return enhanced


def apply_enhanced_contrast_and_augmentation(slice_data, 
                                           contrast_method='clahe',
                                           contrast_params=None,
                                           cornucopia_augmentation=None,
                                           random_state=None):
    """
    Apply Cornucopia augmentation FIRST, then contrast enhancement.
    """
    augmented_slice = slice_data.copy()
    
    if cornucopia_augmentation is not None and CORNUCOPIA_INTEGRATION_AVAILABLE:
        if np.all(slice_data == 0):
            print("   ⚠️  Input slice is all zeros, skipping Cornucopia augmentation")
        elif np.std(slice_data) < 1e-6:
            print("   ⚠️  Input slice has no variance, skipping Cornucopia augmentation")
        else:
            try:
                if isinstance(cornucopia_augmentation, str):
                    augmented_slice = augment_fiber_slice(
                        slice_data,
                        preset=cornucopia_augmentation,
                        random_state=random_state
                    )
                elif isinstance(cornucopia_augmentation, dict):
                    augmented_slice = augment_fiber_slice(
                        slice_data,
                        custom_config=cornucopia_augmentation,
                        random_state=random_state
                    )
                
                # Safety checks
                if np.all(augmented_slice == 0):
                    print(f"      ⚠️  Cornucopia augmentation made slice all zeros, using original")
                    augmented_slice = slice_data.copy()
                elif np.std(augmented_slice) < 1e-6:
                    print(f"      ⚠️  Cornucopia augmentation removed all variance, using original")
                    augmented_slice = slice_data.copy()
                elif np.max(augmented_slice) < 0.01:
                    print(f"      ⚠️  Cornucopia augmentation made slice too dark, using original")
                    augmented_slice = slice_data.copy()
                else:
                    print(f"   ✅ Applied Cornucopia '{cornucopia_augmentation}' to NIfTI slice")
            except Exception as e:
                print(f"   ⚠️  Cornucopia augmentation failed: {e}, using original slice")
                augmented_slice = slice_data.copy()
    
    # Apply contrast enhancement
    if contrast_params is None:
        contrast_params = {}
    
    enhanced_slice = apply_contrast_enhancement(
        augmented_slice, 
        clip_limit=contrast_params.get('clip_limit', 0.01),
        tile_grid_size=contrast_params.get('tile_grid_size', (8, 8))
    )
    
    return enhanced_slice 