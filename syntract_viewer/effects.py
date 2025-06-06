"""
Visual effects for NIfTI tractography visualization.
"""

import numpy as np
from skimage import filters, exposure
import random

from masking import create_smart_brain_mask, adaptive_ventricle_preservation


def apply_dark_field_effect(slice_clahe, intensity_params=None, random_state=None):
    """
    Apply a dark field microscopy effect with controllable parameters.
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Default parameters with randomization
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create simple brain mask
    brain_mask = slice_clahe > 0.01
    
    # Invert colors for dark field effect
    inverted = 1 - slice_clahe
    
    # Apply gamma correction
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Apply threshold
    threshold = intensity_params['threshold']
    dark_field_stretched[dark_field_stretched < threshold] = 0
    
    # Force background areas to black
    original_background = slice_clahe <= 0.001
    dark_field_stretched[original_background] = 0
    
    # Add subtle noise
    noise_level = random.uniform(0.005, 0.02)
    noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
    tissue_areas = dark_field_stretched > 0
    dark_field_stretched[tissue_areas] = np.clip(dark_field_stretched[tissue_areas] + noise[tissue_areas], 0, 1)
    
    return dark_field_stretched


def apply_smart_dark_field_effect(slice_clahe, intensity_params=None, mask_method='adaptive_morphology',
                                 preserve_ventricles=True, random_state=None, 
                                 mask_threshold=0.01, keep_all_brain_parts=True):
    """
    Enhanced dark field effect with smart brain masking.
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Default parameters with randomization
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create smart brain mask
    brain_mask = create_smart_brain_mask(
        slice_clahe, 
        method=mask_method,
        initial_threshold=mask_threshold,
        keep_all_components=keep_all_brain_parts,
        min_object_size=300,
        closing_disk_size=20,
        opening_disk_size=2
    )
    
    # Preserve ventricles if requested
    if preserve_ventricles:
        brain_mask = adaptive_ventricle_preservation(slice_clahe, brain_mask, ventricle_threshold_percentile=10)
    
    # Apply dark field effect only to brain regions
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Smart thresholding: different treatment for brain vs background
    threshold = intensity_params['threshold']
    
    # For brain regions: gentle thresholding
    brain_region = dark_field_stretched * brain_mask
    brain_region[brain_region < threshold/3] = 0
    
    # For background: complete suppression
    background_region = dark_field_stretched * (~brain_mask.astype(bool))
    background_region[:] = 0
    
    # Combine brain and background
    result = brain_region + background_region
    
    # Add subtle noise to brain regions only
    if np.any(brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, result.shape)
        brain_areas = brain_mask > 0
        result[brain_areas] = np.clip(result[brain_areas] + noise[brain_areas], 0, 1)
    
    return result 