"""
Visual effects for NIfTI tractography visualization.
"""

import numpy as np
from skimage import filters, exposure
import random

try:
    from .masking import create_smart_brain_mask, adaptive_ventricle_preservation
except ImportError:
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
                                 preserve_ventricles=False, random_state=None, 
                                 mask_threshold=0.01, keep_all_brain_parts=True):
    """
    Enhanced dark field effect with smart brain masking.
    
    Parameters
    ----------
    preserve_ventricles : bool, default False
        Whether to attempt to preserve ventricle-like regions. 
        Set to True only if you specifically need ventricle preservation 
        and are experiencing ventricle suppression artifacts.
        False by default to prevent fake ventricle creation.
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
    
    # Create smart brain mask (ultra-conservative settings to eliminate fake ventricles)
    brain_mask = create_smart_brain_mask(
        slice_clahe, 
        method='adaptive_morphology',  # Use most reliable method
        initial_threshold=max(mask_threshold, 0.02),  # Much higher threshold to avoid noise
        keep_all_components=True,
        min_object_size=1000,  # Much larger minimum size
        closing_disk_size=5,   # Very small closing to avoid over-filling
        opening_disk_size=1    # Minimal opening
    )
    
    # Preserve ventricles if requested (now much more conservative)
    if preserve_ventricles:
        brain_mask = adaptive_ventricle_preservation(slice_clahe, brain_mask, ventricle_threshold_percentile=2)
    
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
    
    # AGGRESSIVE background suppression - force ALL background to pure black
    background_areas = ~brain_mask.astype(bool)
    result[background_areas] = 0.0
    
    # COMPREHENSIVE ARTIFACT REMOVAL - eliminate small isolated bright spots AND black dots
    if np.any(result > 0):
        from skimage import measure, morphology
        
        # Remove small isolated bright spots
        bright_threshold = np.percentile(result[result > 0], 85) if np.any(result > 0) else 0
        bright_regions = result > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Calculate brain area for size thresholding
        brain_area = np.sum(brain_mask)
        min_region_size = max(50, brain_area * 0.001)  # At least 0.1% of brain area
        
        for region in regions:
            if region.area < min_region_size:
                # This is likely an artifact - remove it
                artifact_mask = labeled_bright == region.label
                result[artifact_mask] = 0.0
        
        # Remove small black holes/dots within brain tissue
        # First, create a mask of brain tissue areas
        brain_tissue_mask = (result > np.percentile(result[result > 0], 10)) & brain_mask
        
        # Fill small holes in brain tissue
        filled_brain = morphology.remove_small_holes(brain_tissue_mask.astype(bool), area_threshold=100)
        
        # Find holes that were filled
        holes_to_fill = filled_brain & (~brain_tissue_mask.astype(bool))
        
        # Fill these holes with interpolated values from surrounding tissue
        if np.any(holes_to_fill):
            # Get mean intensity of surrounding brain tissue
            surrounding_intensity = np.mean(result[brain_tissue_mask]) if np.any(brain_tissue_mask) else 0
            
            # Fill holes with slightly randomized values around the mean
            try:
                # Ensure we have the right shapes
                holes_indices = np.where(holes_to_fill)
                num_holes = len(holes_indices[0])
                
                if num_holes > 0:
                    hole_values = surrounding_intensity * np.random.uniform(0.8, 1.2, num_holes)
                    result[holes_to_fill] = hole_values
            except Exception as e:
                # If there's any shape mismatch, skip hole filling
                print(f"      ⚠️  Hole filling failed: {e}, skipping hole interpolation")
                pass
        
        # Remove small isolated dark spots within brain regions
        # Identify unusually dark spots within brain tissue
        if np.any(brain_tissue_mask):
            brain_mean = np.mean(result[brain_tissue_mask])
            brain_std = np.std(result[brain_tissue_mask])
            
            # Find abnormally dark spots (more than 2 std below mean)
            dark_threshold = max(0, brain_mean - 2 * brain_std)
            dark_spots = (result < dark_threshold) & brain_mask & (result > 0)
            
            # Label and remove small dark spots
            labeled_dark = measure.label(dark_spots)
            dark_regions = measure.regionprops(labeled_dark)
            
            for region in dark_regions:
                if region.area < min_region_size:
                    # Remove small dark spots by replacing with local average
                    dark_spot_mask = labeled_dark == region.label
                    
                    # Get surrounding area for interpolation
                    y_min, x_min, y_max, x_max = region.bbox
                    y_pad = max(5, (y_max - y_min) // 2)
                    x_pad = max(5, (x_max - x_min) // 2)
                    
                    y_start = max(0, y_min - y_pad)
                    y_end = min(result.shape[0], y_max + y_pad)
                    x_start = max(0, x_min - x_pad)
                    x_end = min(result.shape[1], x_max + x_pad)
                    
                    surrounding_region = result[y_start:y_end, x_start:x_end]
                    surrounding_mask = brain_tissue_mask[y_start:y_end, x_start:x_end]
                    
                    if np.any(surrounding_mask):
                        replacement_value = np.mean(surrounding_region[surrounding_mask])
                        result[dark_spot_mask] = replacement_value * np.random.uniform(0.9, 1.1)
    
    # Add subtle noise to brain regions only (but keep background pure black)
    if np.any(brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, result.shape)
        brain_areas = brain_mask > 0
        result[brain_areas] = np.clip(result[brain_areas] + noise[brain_areas], 0, 1)
        
        # Final background cleanup - ensure no noise leaked to background
        background_areas = ~brain_mask.astype(bool)
        result[background_areas] = 0.0
    
    return result


def apply_conservative_dark_field_effect(slice_clahe, intensity_params=None, random_state=None):
    """
    Apply dark field effect with no ventricle preservation to avoid fake artifacts.
    
    This is a simplified version that focuses purely on the dark field effect
    without any ventricle preservation logic that might create fake structures.
    
    Parameters
    ----------
    slice_clahe : np.ndarray
        Input slice data after contrast enhancement
    intensity_params : dict, optional
        Parameters for intensity adjustments
    random_state : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    np.ndarray
        Dark field processed slice with no fake ventricles
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
    
    # Simple brain mask - higher threshold, no ventricle logic
    brain_mask = slice_clahe > 0.02
    
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
    
    # Simple thresholding
    threshold = intensity_params['threshold']
    dark_field_stretched[dark_field_stretched < threshold] = 0
    
    # Strict background suppression - anything outside brain mask is zero
    background_areas = ~brain_mask.astype(bool)
    dark_field_stretched[background_areas] = 0.0
    
    # COMPREHENSIVE ARTIFACT REMOVAL - eliminate small isolated bright spots AND black dots
    if np.any(dark_field_stretched > 0):
        from skimage import measure, morphology
        
        # Remove small isolated bright spots
        bright_threshold = np.percentile(dark_field_stretched[dark_field_stretched > 0], 85)
        bright_regions = dark_field_stretched > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Calculate brain area for size thresholding
        brain_area = np.sum(brain_mask)
        min_region_size = max(50, brain_area * 0.001)  # At least 0.1% of brain area
        
        for region in regions:
            if region.area < min_region_size:
                # This is likely an artifact - remove it
                artifact_mask = labeled_bright == region.label
                dark_field_stretched[artifact_mask] = 0.0
        
        # Remove small black holes/dots within brain tissue
        brain_tissue_mask = (dark_field_stretched > np.percentile(dark_field_stretched[dark_field_stretched > 0], 10)) & brain_mask
        
        # Fill small holes in brain tissue
        filled_brain = morphology.remove_small_holes(brain_tissue_mask.astype(bool), area_threshold=100)
        holes_to_fill = filled_brain & (~brain_tissue_mask.astype(bool))
        
        if np.any(holes_to_fill):
            surrounding_intensity = np.mean(dark_field_stretched[brain_tissue_mask]) if np.any(brain_tissue_mask) else 0
            try:
                # Ensure we have the right shapes
                holes_indices = np.where(holes_to_fill)
                num_holes = len(holes_indices[0])
                
                if num_holes > 0:
                    hole_values = surrounding_intensity * np.random.uniform(0.8, 1.2, num_holes)
                    dark_field_stretched[holes_to_fill] = hole_values
            except Exception as e:
                # If there's any shape mismatch, skip hole filling
                print(f"      ⚠️  Hole filling failed: {e}, skipping hole interpolation")
                pass
        
        # Remove small isolated dark spots within brain regions
        if np.any(brain_tissue_mask):
            brain_mean = np.mean(dark_field_stretched[brain_tissue_mask])
            brain_std = np.std(dark_field_stretched[brain_tissue_mask])
            
            dark_threshold = max(0, brain_mean - 2 * brain_std)
            dark_spots = (dark_field_stretched < dark_threshold) & brain_mask & (dark_field_stretched > 0)
            
            labeled_dark = measure.label(dark_spots)
            dark_regions = measure.regionprops(labeled_dark)
            
            for region in dark_regions:
                if region.area < min_region_size:
                    dark_spot_mask = labeled_dark == region.label
                    
                    y_min, x_min, y_max, x_max = region.bbox
                    y_pad = max(5, (y_max - y_min) // 2)
                    x_pad = max(5, (x_max - x_min) // 2)
                    
                    y_start = max(0, y_min - y_pad)
                    y_end = min(dark_field_stretched.shape[0], y_max + y_pad)
                    x_start = max(0, x_min - x_pad)
                    x_end = min(dark_field_stretched.shape[1], x_max + x_pad)
                    
                    surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                    surrounding_mask = brain_tissue_mask[y_start:y_end, x_start:x_end]
                    
                    if np.any(surrounding_mask):
                        replacement_value = np.mean(surrounding_region[surrounding_mask])
                        dark_field_stretched[dark_spot_mask] = replacement_value * np.random.uniform(0.9, 1.1)
    
    # Add subtle noise to brain regions only
    if np.any(brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
        brain_areas = brain_mask > 0
        dark_field_stretched[brain_areas] = np.clip(dark_field_stretched[brain_areas] + noise[brain_areas], 0, 1)
        
        # Final background cleanup - ensure pure black background
        background_areas = ~brain_mask.astype(bool)
        dark_field_stretched[background_areas] = 0.0
    
    return dark_field_stretched 