"""
Visual effects for NIfTI tractography visualization.
"""

import numpy as np
from skimage import filters, exposure, morphology
from scipy import ndimage
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
                                 mask_threshold=0.01, keep_all_brain_parts=True,
                                 force_background_black=True):
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
    
    # Create conservative brain mask that preserves bright blockface areas
    # Use multiple criteria to ensure bright areas are always included
    basic_threshold = max(mask_threshold, 0.015)  # Lower threshold than before
    brain_mask_basic = slice_clahe > basic_threshold
    
    # Always include very bright areas (potential blockface)
    bright_threshold = 0.3  # Areas that are clearly bright tissue
    bright_areas = slice_clahe > bright_threshold
    
    # Combine basic mask with bright areas
    brain_mask = brain_mask_basic | bright_areas
    
    # Very minimal morphological processing to avoid losing bright areas
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
    
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
    
    # Background suppression - configurable to preserve bright blockface areas
    if force_background_black:
        background_areas = ~brain_mask.astype(bool)
        result[background_areas] = 0.0
    else:
        # Dim background instead of forcing to pure black
        background_areas = ~brain_mask.astype(bool)
        result[background_areas] *= 0.3
    
    # COMPREHENSIVE ARTIFACT REMOVAL - eliminate small isolated bright spots AND black dots
    if np.any(result > 0):
        from skimage import measure
        
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
        
        # Final background cleanup - configurable
        if force_background_black:
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
    
    # Conservative brain mask that preserves bright blockface areas
    # Use multiple criteria to ensure bright areas are always included
    basic_threshold = 0.015  # Lower threshold
    brain_mask_basic = slice_clahe > basic_threshold
    
    # Always include very bright areas (potential blockface)
    bright_threshold = 0.3  # Areas that are clearly bright tissue
    bright_areas = slice_clahe > bright_threshold
    
    # Combine basic mask with bright areas
    brain_mask = brain_mask_basic | bright_areas
    
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


def apply_gentle_dark_field_effect(slice_clahe, intensity_params=None, random_state=None):
    """
    Apply a gentle dark field effect with minimal artifact removal to prevent sharp black spots.
    
    This version reduces aggressive morphological operations that can create unnatural 
    sharp boundaries and black spots in the visualization.
    
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
        Dark field processed slice with smooth transitions
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
    
    # Create gentle brain mask that preserves bright blockface areas
    # Use multiple criteria to ensure bright areas are always included
    basic_threshold = 0.02  # Gentle but not too high
    brain_mask_basic = slice_clahe > basic_threshold
    
    # Always include very bright areas (potential blockface)
    bright_threshold = 0.3  # Areas that are clearly bright tissue
    bright_areas = slice_clahe > bright_threshold
    
    # Combine basic mask with bright areas
    brain_mask = brain_mask_basic | bright_areas
    
    # Minimal morphological processing to avoid sharp boundaries
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
    
    # Apply dark field effect
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply gentle smoothing
    dark_field = filters.gaussian(dark_field, sigma=0.8)  # Slightly more smoothing
    
    # Enhance contrast with gentler stretching
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Gentle thresholding without sharp cutoffs
    threshold = intensity_params['threshold']
    
    # Apply soft thresholding to avoid sharp boundaries
    soft_threshold_range = threshold * 2  # Create a soft transition zone
    transition_mask = (dark_field_stretched >= threshold) & (dark_field_stretched <= soft_threshold_range)
    below_threshold = dark_field_stretched < threshold
    
    # Smooth transition for areas near threshold
    if np.any(transition_mask):
        transition_factor = (dark_field_stretched[transition_mask] - threshold) / (soft_threshold_range - threshold)
        dark_field_stretched[transition_mask] *= transition_factor
    
    # Set areas clearly below threshold to zero
    dark_field_stretched[below_threshold] = 0
    
    # Apply brain mask with soft edges
    background_areas = ~brain_mask.astype(bool)
    
    # Create soft edge transition for brain boundary
    from scipy import ndimage
    brain_distance = ndimage.distance_transform_edt(brain_mask)
    edge_distance = ndimage.distance_transform_edt(~brain_mask) 
    
    # Create smooth transition at brain boundary (within 3 pixels)
    transition_zone = (edge_distance <= 3) & (brain_distance <= 3)
    if np.any(transition_zone):
        edge_factor = np.minimum(brain_distance[transition_zone] / 3.0, 1.0)
        dark_field_stretched[transition_zone] *= edge_factor
    
    # Set clear background to zero
    clear_background = edge_distance > 3
    dark_field_stretched[clear_background] = 0.0
    
    # GENTLE artifact removal - only remove very obvious artifacts
    if np.any(dark_field_stretched > 0):
        from skimage import measure
        
        # Only remove extremely small isolated bright spots (much more conservative)
        bright_threshold = np.percentile(dark_field_stretched[dark_field_stretched > 0], 95)
        bright_regions = dark_field_stretched > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Only remove tiny artifacts (< 10 pixels)
        for region in regions:
            if region.area < 10:  # Much smaller threshold
                artifact_mask = labeled_bright == region.label
                # Instead of setting to zero, blend with surrounding
                y_min, x_min, y_max, x_max = region.bbox
                
                # Get larger surrounding area for smooth interpolation
                pad = 5
                y_start = max(0, y_min - pad)
                y_end = min(dark_field_stretched.shape[0], y_max + pad)
                x_start = max(0, x_min - pad)
                x_end = min(dark_field_stretched.shape[1], x_max + pad)
                
                surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                surrounding_mask = brain_mask[y_start:y_end, x_start:x_end]
                
                if np.any(surrounding_mask):
                    # Use median instead of mean for more robust estimation
                    replacement_value = np.median(surrounding_region[surrounding_mask])
                    # Add some noise to avoid uniform patches
                    noise_factor = np.random.uniform(0.95, 1.05)
                    dark_field_stretched[artifact_mask] = replacement_value * noise_factor
    
    # Add very subtle noise only to avoid uniform regions
    if np.any(brain_mask):
        noise_level = random.uniform(0.002, 0.008)  # Much less noise
        noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
        brain_areas = brain_mask > 0
        dark_field_stretched[brain_areas] = np.clip(
            dark_field_stretched[brain_areas] + noise[brain_areas], 0, 1
        )
    
    return dark_field_stretched


def apply_balanced_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                    force_background_black=True):
    """
    Apply a balanced dark field effect that provides moderate artifact removal 
    while preventing sharp black spots.
    
    This is a middle ground between aggressive (may create black spots) and 
    gentle (minimal cleanup) approaches, providing better artifact removal 
    with smooth transitions.
    
    Parameters
    ----------
    slice_clahe : np.ndarray
        Input slice data after contrast enhancement
    intensity_params : dict, optional
        Parameters for intensity adjustments
    random_state : int, optional
        Random seed for reproducible results
    force_background_black : bool, optional
        Whether to force background areas to black
        
    Returns
    -------
    np.ndarray
        Dark field processed slice with balanced artifact removal
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
    
    # Create conservative brain mask that preserves bright blockface areas
    # Use multiple criteria to ensure bright areas are always included
    basic_threshold = 0.015  # Lower threshold than before
    brain_mask_basic = slice_clahe > basic_threshold
    
    # Always include very bright areas (potential blockface)
    bright_threshold = 0.3  # Areas that are clearly bright tissue
    bright_areas = slice_clahe > bright_threshold
    
    # Combine basic mask with bright areas
    brain_mask = brain_mask_basic | bright_areas
    
    # Very minimal morphological processing to avoid losing bright areas
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))  # Smaller disk
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)  # Smaller minimum
    
    # Apply dark field effect
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply moderate smoothing
    dark_field = filters.gaussian(dark_field, sigma=0.6)
    
    # Enhance contrast
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Balanced thresholding with soft transitions
    threshold = intensity_params['threshold']
    
    # Create a soft transition zone (smaller than gentle but still present)
    soft_threshold_range = threshold * 1.5
    transition_mask = (dark_field_stretched >= threshold) & (dark_field_stretched <= soft_threshold_range)
    below_threshold = dark_field_stretched < threshold
    
    # Apply soft transition
    if np.any(transition_mask):
        transition_factor = (dark_field_stretched[transition_mask] - threshold) / (soft_threshold_range - threshold)
        # Use a smooth curve instead of linear
        transition_factor = np.power(transition_factor, 0.7)  # Slightly curved transition
        dark_field_stretched[transition_mask] *= transition_factor
    
    # Set areas clearly below threshold to zero
    dark_field_stretched[below_threshold] = 0
    
    # Apply brain mask with moderate edge softening
    background_areas = ~brain_mask.astype(bool)
    
    # Create moderate soft edge transition (2 pixel transition instead of 3)
    from scipy import ndimage
    brain_distance = ndimage.distance_transform_edt(brain_mask)
    edge_distance = ndimage.distance_transform_edt(~brain_mask) 
    
    # Moderate transition zone
    transition_zone = (edge_distance <= 2) & (brain_distance <= 2)
    if np.any(transition_zone):
        edge_factor = np.minimum(brain_distance[transition_zone] / 2.0, 1.0)
        # Apply smooth transition curve
        edge_factor = np.power(edge_factor, 0.8)
        dark_field_stretched[transition_zone] *= edge_factor
    
    # Set clear background to zero (configurable)
    if force_background_black:
        clear_background = edge_distance > 2
        dark_field_stretched[clear_background] = 0.0
    else:
        # Dim background instead of forcing to black
        clear_background = edge_distance > 2
        dark_field_stretched[clear_background] *= 0.2
    
    # BALANCED artifact removal - moderate cleanup
    if np.any(dark_field_stretched > 0):
        from skimage import measure
        
        # Remove moderately small isolated bright spots
        bright_threshold = np.percentile(dark_field_stretched[dark_field_stretched > 0], 90)  # 90th instead of 95th
        bright_regions = dark_field_stretched > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Calculate brain area for size thresholding
        brain_area = np.sum(brain_mask)
        min_region_size = max(25, brain_area * 0.0005)  # Moderate threshold (25 pixels or 0.05% of brain)
        
        # Remove artifacts with smooth replacement
        for region in regions:
            if region.area < min_region_size:
                artifact_mask = labeled_bright == region.label
                # Get surrounding area for smooth interpolation
                y_min, x_min, y_max, x_max = region.bbox
                
                # Moderate padding for interpolation
                pad = 4
                y_start = max(0, y_min - pad)
                y_end = min(dark_field_stretched.shape[0], y_max + pad)
                x_start = max(0, x_min - pad)
                x_end = min(dark_field_stretched.shape[1], x_max + pad)
                
                surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                surrounding_mask = brain_mask[y_start:y_end, x_start:x_end]
                
                if np.any(surrounding_mask):
                    # Use weighted average of median and mean for balanced replacement
                    median_val = np.median(surrounding_region[surrounding_mask])
                    mean_val = np.mean(surrounding_region[surrounding_mask])
                    replacement_value = 0.7 * median_val + 0.3 * mean_val
                    
                    # Add moderate noise to avoid uniform patches
                    noise_factor = np.random.uniform(0.92, 1.08)
                    dark_field_stretched[artifact_mask] = replacement_value * noise_factor
        
        # Moderate hole filling within brain tissue
        brain_tissue_mask = (dark_field_stretched > np.percentile(dark_field_stretched[dark_field_stretched > 0], 15)) & brain_mask
        
        # Fill moderate holes
        filled_brain = morphology.remove_small_holes(brain_tissue_mask.astype(bool), area_threshold=75)
        holes_to_fill = filled_brain & (~brain_tissue_mask.astype(bool))
        
        if np.any(holes_to_fill):
            surrounding_intensity = np.mean(dark_field_stretched[brain_tissue_mask]) if np.any(brain_tissue_mask) else 0
            try:
                holes_indices = np.where(holes_to_fill)
                num_holes = len(holes_indices[0])
                
                if num_holes > 0:
                    # Moderate randomization
                    hole_values = surrounding_intensity * np.random.uniform(0.85, 1.15, num_holes)
                    dark_field_stretched[holes_to_fill] = hole_values
            except Exception as e:
                pass  # Silently skip if there are issues
        
        # Remove moderate dark spots within brain regions
        if np.any(brain_tissue_mask):
            brain_mean = np.mean(dark_field_stretched[brain_tissue_mask])
            brain_std = np.std(dark_field_stretched[brain_tissue_mask])
            
            # Moderate threshold for dark spots (1.5 std instead of 2)
            dark_threshold = max(0, brain_mean - 1.5 * brain_std)
            dark_spots = (dark_field_stretched < dark_threshold) & brain_mask & (dark_field_stretched > 0)
            
            labeled_dark = measure.label(dark_spots)
            dark_regions = measure.regionprops(labeled_dark)
            
            for region in dark_regions:
                if region.area < min_region_size:
                    dark_spot_mask = labeled_dark == region.label
                    
                    # Get surrounding area for interpolation
                    y_min, x_min, y_max, x_max = region.bbox
                    y_pad = max(4, (y_max - y_min))
                    x_pad = max(4, (x_max - x_min))
                    
                    y_start = max(0, y_min - y_pad)
                    y_end = min(dark_field_stretched.shape[0], y_max + y_pad)
                    x_start = max(0, x_min - x_pad)
                    x_end = min(dark_field_stretched.shape[1], x_max + x_pad)
                    
                    surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                    surrounding_mask = brain_tissue_mask[y_start:y_end, x_start:x_end]
                    
                    if np.any(surrounding_mask):
                        # Balanced replacement with weighted average
                        median_val = np.median(surrounding_region[surrounding_mask])
                        mean_val = np.mean(surrounding_region[surrounding_mask])
                        replacement_value = 0.6 * median_val + 0.4 * mean_val
                        dark_field_stretched[dark_spot_mask] = replacement_value * np.random.uniform(0.9, 1.1)
    
    # Add moderate noise to brain regions
    if np.any(brain_mask):
        noise_level = random.uniform(0.008, 0.015)  # Moderate noise level
        noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
        brain_areas = brain_mask > 0
        dark_field_stretched[brain_areas] = np.clip(
            dark_field_stretched[brain_areas] + noise[brain_areas], 0, 1
        )
    
    return dark_field_stretched


def apply_blockface_preserving_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                                force_background_black=True):
    """
    Apply a dark field effect that specifically preserves bright blockface areas.
    
    This function applies the dark field inversion only to areas that are NOT bright blockface,
    preserving bright tissue areas in their original form while still providing the dark field
    aesthetic for brain tissue.
    
    Parameters
    ----------
    slice_clahe : np.ndarray
        Input slice data after contrast enhancement
    intensity_params : dict, optional
        Parameters for intensity adjustments
    random_state : int, optional
        Random seed for reproducible results
    force_background_black : bool, optional
        Whether to force background areas to black
        
    Returns
    -------
    np.ndarray
        Dark field processed slice with preserved bright blockface areas
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
    
    # STEP 1: Apply FULL aggressive dark field processing to entire slice first
    # (same as smart dark field effect for maximum quality)
    
    # Create aggressive brain mask that includes bright areas
    basic_threshold = 0.015
    brain_mask_basic = slice_clahe > basic_threshold
    
    # Always include very bright areas initially
    bright_threshold = 0.3
    bright_areas = slice_clahe > bright_threshold
    
    # Combine basic mask with bright areas for processing
    full_brain_mask = brain_mask_basic | bright_areas
    # Apply aggressive morphological processing for clarity
    full_brain_mask = morphology.binary_closing(full_brain_mask, morphology.disk(3))
    full_brain_mask = morphology.remove_small_objects(full_brain_mask, min_size=100)
    full_brain_mask = ndimage.binary_fill_holes(full_brain_mask)
    full_brain_mask = morphology.binary_dilation(full_brain_mask, morphology.disk(1))
    
    # Apply dark field effect to ENTIRE slice (aggressive processing)
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast globally
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost globally
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Smart thresholding: different treatment for brain vs background
    threshold = intensity_params['threshold']
    
    # For brain regions: gentle thresholding
    brain_region = dark_field_stretched * full_brain_mask
    brain_region[brain_region < threshold/3] = 0
    
    # For background: complete suppression
    background_region = dark_field_stretched * (~full_brain_mask.astype(bool))
    background_region[:] = 0
    
    # Combine brain and background
    result = brain_region + background_region
    
    # STEP 2: Identify tissue density areas for enhanced contrast
    # Create multiple density levels for realistic tissue contrast enhancement
    medium_tissue_threshold = 0.15  # Medium density tissue
    high_tissue_threshold = 0.25    # High density tissue  
    very_high_tissue_threshold = bright_threshold  # Very high density (blockface)
    
    # Identify different tissue density areas
    medium_tissue_areas = (slice_clahe > medium_tissue_threshold) & (slice_clahe <= high_tissue_threshold)
    high_tissue_areas = (slice_clahe > high_tissue_threshold) & (slice_clahe <= very_high_tissue_threshold)
    blockface_areas = slice_clahe > very_high_tissue_threshold
    
    # Store the original values for selective restoration
    original_medium_values = slice_clahe[medium_tissue_areas] if np.any(medium_tissue_areas) else None
    original_high_values = slice_clahe[high_tissue_areas] if np.any(high_tissue_areas) else None
    original_blockface_values = slice_clahe[blockface_areas] if np.any(blockface_areas) else None
    
        # STEP 3: Apply COMPREHENSIVE ARTIFACT REMOVAL (same as smart dark field)
    if np.any(result > 0):
        from skimage import measure
        
        # Remove small isolated bright spots
        bright_threshold_removal = np.percentile(result[result > 0], 85) if np.any(result > 0) else 0
        bright_regions = result > bright_threshold_removal
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Calculate brain area for size thresholding
        brain_area = np.sum(full_brain_mask)
        min_region_size = max(50, brain_area * 0.001)  # At least 0.1% of brain area
        
        for region in regions:
            if region.area < min_region_size:
                # This is likely an artifact - remove it
                artifact_mask = labeled_bright == region.label
                result[artifact_mask] = 0.0
        
        # Remove small black holes/dots within brain tissue
        brain_tissue_mask = (result > np.percentile(result[result > 0], 10)) & full_brain_mask
        
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
                holes_indices = np.where(holes_to_fill)
                num_holes = len(holes_indices[0])
                
                if num_holes > 0:
                    hole_values = surrounding_intensity * np.random.uniform(0.8, 1.2, num_holes)
                    result[holes_to_fill] = hole_values
            except Exception:
                # If there's any shape mismatch, skip hole filling
                pass
        
        # Remove small isolated dark spots within brain regions
        if np.any(brain_tissue_mask):
            brain_mean = np.mean(result[brain_tissue_mask])
            brain_std = np.std(result[brain_tissue_mask])
            
            # Find abnormally dark spots (more than 2 std below mean)
            dark_threshold_removal = max(0, brain_mean - 2 * brain_std)
            dark_spots = (result < dark_threshold_removal) & full_brain_mask & (result > 0)
            
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

    # STEP 4: Process different tissue density areas with SUBTLE darkfield-appropriate enhancement
    
    # Process medium density tissue areas - very dark for authentic darkfield
    if np.any(medium_tissue_areas) and original_medium_values is not None:
        medium_processed = original_medium_values.copy()
        
        # Apply moderate smoothing like balanced version
        medium_smoothed = filters.gaussian(medium_processed, sigma=0.6)
        
        # More aggressive contrast stretching like balanced version
        p_low, p_high = (2.0, 95.0)  # Tighter range for better contrast
        p1, p99 = np.percentile(medium_smoothed, (p_low, p_high))
        if p99 > p1:
            medium_stretched = exposure.rescale_intensity(medium_smoothed, in_range=(p1, p99))
        else:
            medium_stretched = medium_smoothed
        
        # Much darker enhancement for authentic darkfield appearance
        medium_enhanced = medium_stretched * 1.05  # Very minimal boost for dark look
        medium_final = np.clip(medium_enhanced * 0.1, 0, 0.25)  # Very dark values for authentic darkfield
        
        result[medium_tissue_areas] = medium_final
    
    # Process high density tissue areas - very dark for authentic darkfield  
    if np.any(high_tissue_areas) and original_high_values is not None:
        high_processed = original_high_values.copy()
        
        # Apply moderate contrast enhancement like balanced version
        high_smoothed = filters.gaussian(high_processed, sigma=0.6)
        
        # More aggressive contrast stretching like balanced version
        p_low, p_high = (1.5, 97.0)  # Tighter range for better contrast
        p1, p99 = np.percentile(high_smoothed, (p_low, p_high))
        if p99 > p1:
            high_stretched = exposure.rescale_intensity(high_smoothed, in_range=(p1, p99))
        else:
            high_stretched = high_smoothed
        
        # Much darker enhancement for authentic darkfield appearance
        high_enhanced = high_stretched * 1.08  # Very minimal boost for dark look
        high_final = np.clip(high_enhanced * 0.2, 0, 0.35)  # Very dark values for authentic darkfield
        
        result[high_tissue_areas] = high_final
    
    # Process blockface areas (very high density) - selective brightness for authentic darkfield
    if np.any(blockface_areas) and original_blockface_values is not None:
        blockface_processed = original_blockface_values.copy()
        
        # Apply enhanced contrast for blockface visibility but keep darkfield character
        blockface_smoothed = filters.gaussian(blockface_processed, sigma=0.5)
        
        # Moderate contrast stretching for darkfield authenticity
        p_low, p_high = (1.0, 98.0)  # Less aggressive for darkfield character
        p1, p99 = np.percentile(blockface_smoothed, (p_low, p_high))
        if p99 > p1:
            blockface_stretched = exposure.rescale_intensity(blockface_smoothed, in_range=(p1, p99))
        else:
            blockface_stretched = blockface_smoothed
        
        # Selective enhancement - only brighten the brightest blockface areas
        blockface_enhanced = blockface_stretched * 1.2  # Reduced enhancement
        blockface_final = np.clip(blockface_enhanced * 0.35, 0, 0.55)  # Much darker range for authentic darkfield
        
        result[blockface_areas] = blockface_final

    # STEP 5: Handle background areas - make them PURE BLACK for maximum contrast
    background_areas = ~full_brain_mask.astype(bool)

    # Always force background to pure black for authentic darkfield
    result[background_areas] = 0.0
    
    # STEP 5.5: Apply subtle global contrast enhancement for darkfield realism
    # Only enhance brain tissue areas to maintain realism
    brain_tissue_areas = full_brain_mask.astype(bool)
    if np.any(brain_tissue_areas):
        # Get current brain tissue values
        brain_values = result[brain_tissue_areas]
        
        # Apply minimal global contrast boost for authentic darkfield
        non_enhanced_areas = brain_tissue_areas & (~medium_tissue_areas) & (~high_tissue_areas) & (~blockface_areas)
        if np.any(non_enhanced_areas):
            # Apply minimal contrast enhancement to maintain authentic darkfield character
            regular_values = result[non_enhanced_areas]
            if len(regular_values) > 0 and np.std(regular_values) > 0:
                # Minimal enhancement for very dark authentic darkfield
                enhanced_regular = regular_values * 1.02  # Barely any boost for authentic darkfield
                result[non_enhanced_areas] = np.clip(enhanced_regular, 0, 0.3)  # Very low cap for dark appearance

    # STEP 6: Add subtle noise to brain regions only (same as smart dark field)
    if np.any(full_brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, result.shape)
        brain_areas_for_noise = full_brain_mask > 0
        result[brain_areas_for_noise] = np.clip(result[brain_areas_for_noise] + noise[brain_areas_for_noise], 0, 1)
        
        # Final background cleanup - always force to pure black for darkfield
        background_areas = ~full_brain_mask.astype(bool)
        result[background_areas] = 0.0
    
    # STEP 7: Final balanced contrast enhancement 
    # Apply moderate enhancement like balanced version for good contrast
    tissue_areas = medium_tissue_areas | high_tissue_areas | blockface_areas
    if np.any(tissue_areas):
        # Apply moderate processing for good contrast like balanced version
        tissue_values = result[tissue_areas]
        if len(tissue_values) > 0 and np.std(tissue_values) > 0:
            # Normalize tissue values for processing
            tissue_min, tissue_max = np.min(tissue_values), np.max(tissue_values)
            if tissue_max > tissue_min:
                tissue_normalized = (tissue_values - tissue_min) / (tissue_max - tissue_min)
                
                # Apply ultra-subtle adaptive histogram equalization for authentic darkfield
                try:
                    tissue_enhanced = exposure.equalize_adapthist(
                        tissue_normalized.reshape(np.sum(tissue_areas), 1),
                        clip_limit=0.002,  # Ultra-subtle for authentic darkfield character
                        kernel_size=None
                    ).flatten()
                    
                    # Scale back to original range with barely any boost
                    tissue_final = tissue_enhanced * (tissue_max - tissue_min) + tissue_min
                    tissue_final *= 1.01  # Barely any final boost for darkfield
                    
                    result[tissue_areas] = np.clip(tissue_final, 0, 0.55)  # Much lower cap for very dark appearance
                except:
                    # If adaptive enhancement fails, apply barely any boost
                    result[tissue_areas] = np.clip(tissue_values * 1.01, 0, 0.55)

    return result 