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
    from skimage import morphology
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
    
    # STEP 2: Now identify and selectively preserve blockface areas
    blockface_areas = slice_clahe > bright_threshold
    
    # Store the original blockface values for selective restoration
    original_blockface_values = slice_clahe[blockface_areas] if np.any(blockface_areas) else None
    
        # STEP 3: Apply COMPREHENSIVE ARTIFACT REMOVAL (same as smart dark field)
    if np.any(result > 0):
        from skimage import measure, morphology
        
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

    # STEP 4: Process blockface areas with contrast enhancement but NO inversion
    if np.any(blockface_areas) and original_blockface_values is not None:
        # Apply the SAME contrast enhancement as the rest, but without inversion
        blockface_processed = original_blockface_values.copy()
        
        # Apply same gaussian smoothing
        blockface_temp = original_blockface_values.copy()
        blockface_smoothed = filters.gaussian(blockface_temp, sigma=0.5)
        
        # Apply same contrast stretching
        p_low, p_high = intensity_params['contrast_stretch']
        p1, p99 = np.percentile(blockface_smoothed, (p_low, p_high))
        if p99 > p1:
            blockface_stretched = exposure.rescale_intensity(blockface_smoothed, in_range=(p1, p99))
        else:
            blockface_stretched = blockface_smoothed
        
        # Apply same background boost
        blockface_enhanced = blockface_stretched * intensity_params['background_boost']
        
        # Apply dulling factor to prevent being too bright
        dulling_factor = 0.15  # Slightly less dulling to maintain contrast
        blockface_final = blockface_enhanced * dulling_factor
        blockface_final = np.clip(blockface_final, 0, 1)
        
        result[blockface_areas] = blockface_final

    # STEP 5: Handle background areas
    background_areas = ~full_brain_mask.astype(bool)

    if force_background_black and np.any(background_areas):
        result[background_areas] = 0.0
    elif np.any(background_areas):
        # Dim background instead of forcing to black
        result[background_areas] *= 0.3

    # STEP 6: Add subtle noise to brain regions only (same as smart dark field)
    if np.any(full_brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, result.shape)
        brain_areas_for_noise = full_brain_mask > 0
        result[brain_areas_for_noise] = np.clip(result[brain_areas_for_noise] + noise[brain_areas_for_noise], 0, 1)
        
        # Final background cleanup - configurable
        if force_background_black:
            background_areas = ~full_brain_mask.astype(bool)
            result[background_areas] = 0.0

    

    return result


def apply_natural_anatomical_enhancement(slice_clahe, intensity_params=None, random_state=None, 
                                       force_background_black=True):
    """
    Apply gentle, natural anatomical enhancement without over-processing.
    
    This creates a more realistic appearance by:
    - Gently enhancing white matter brightness
    - Preserving natural tissue contrast
    - Avoiding aggressive dark field inversion
    - Maintaining anatomical authenticity
    
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
        Naturally enhanced slice with realistic appearance
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Use gentle default parameters
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.9, 1.1),
            'threshold': random.uniform(0.005, 0.015),
            'contrast_stretch': (random.uniform(1.0, 3.0), random.uniform(97.0, 99.0)),
            'background_boost': random.uniform(0.95, 1.05),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.3)
        }
    
    # Start with the original slice
    result = slice_clahe.copy().astype(np.float64)
    
    # Create a gentle brain mask to separate brain from background
    brain_threshold = np.percentile(result[result > 0], 5) if np.any(result > 0) else 0.01
    brain_mask = result > brain_threshold
    
    if force_background_black:
        result[~brain_mask] = 0.0
    
    if not np.any(brain_mask):
        return result.astype(slice_clahe.dtype)
    
    # Gentle white matter enhancement
    brain_values = result[brain_mask]
    if len(brain_values) > 0:
        # Identify potential white matter (brighter regions)
        white_matter_threshold = np.percentile(brain_values, 75)
        white_matter_mask = brain_mask & (result > white_matter_threshold)
        
        # Very gentle enhancement of white matter
        if np.any(white_matter_mask):
            white_values = result[white_matter_mask]
            # Subtle brightness boost using gentle gamma correction
            enhanced_white = np.power(white_values, 0.85)  # Very mild brightening
            # Small multiplicative boost
            enhanced_white *= 1.15  # Just 15% brighter
            result[white_matter_mask] = enhanced_white
    
    # Apply very gentle contrast enhancement to brain regions only
    if np.any(brain_mask):
        brain_region = result[brain_mask]
        if len(brain_region) > 0:
            # Gentle contrast stretch
            p_low, p_high = intensity_params['contrast_stretch']
            p1, p99 = np.percentile(brain_region, (p_low, p_high))
            if p99 > p1:
                enhanced_brain = exposure.rescale_intensity(brain_region, in_range=(p1, p99))
                result[brain_mask] = enhanced_brain * intensity_params['background_boost']
    
    # Light smoothing to reduce noise while preserving structure
    if np.any(brain_mask):
        # Apply very gentle Gaussian smoothing
        smoothed = filters.gaussian(result, sigma=0.3)
        # Blend original with smoothed (70% original, 30% smoothed)
        result = 0.7 * result + 0.3 * smoothed
    
    # Apply gentle gamma correction for overall brightness balance
    gamma = intensity_params['gamma']
    if gamma != 1.0:
        # Only apply to brain regions
        brain_region = result[brain_mask]
        if len(brain_region) > 0:
            gamma_corrected = np.power(brain_region, gamma)
            result[brain_mask] = gamma_corrected
    
    # Final gentle thresholding to clean up very dark areas
    threshold = intensity_params['threshold'] 
    result[brain_mask & (result < threshold)] = 0
    
    # Ensure output is in reasonable range
    result = np.clip(result, 0, 1)
    
    return result.astype(slice_clahe.dtype)


def apply_anatomically_aware_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                             force_background_black=True):
    """
    Apply a dark field effect with anatomically-aware white matter enhancement.
    
    This function detects white matter regions and enhances their brightness to appear
    more white and anatomically realistic, while applying appropriate dark field processing
    to other tissue types. Reduces dullness and improves contrast differentiation.
    
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
        Anatomically-aware processed slice with enhanced white matter
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Default parameters optimized for anatomical realism
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # STEP 1: Identify different tissue types based on intensity
    
    # Calculate intensity thresholds for tissue classification
    non_zero_intensities = slice_clahe[slice_clahe > 0]
    if len(non_zero_intensities) == 0:
        return slice_clahe
    
    # Use percentiles to identify tissue types
    background_threshold = 0.015
    grey_matter_threshold = np.percentile(non_zero_intensities, 25)  # Lower 25% = grey matter
    white_matter_threshold = np.percentile(non_zero_intensities, 70)  # Upper 30% = white matter
    very_bright_threshold = np.percentile(non_zero_intensities, 90)   # Top 10% = very bright areas
    
    # Create tissue masks
    background_mask = slice_clahe <= background_threshold
    grey_matter_mask = (slice_clahe > background_threshold) & (slice_clahe <= grey_matter_threshold)
    intermediate_mask = (slice_clahe > grey_matter_threshold) & (slice_clahe <= white_matter_threshold)
    white_matter_mask = (slice_clahe > white_matter_threshold) & (slice_clahe <= very_bright_threshold)
    very_bright_mask = slice_clahe > very_bright_threshold
    
    # STEP 2: Apply different processing to each tissue type
    
    # Initialize result with zeros
    result = np.zeros_like(slice_clahe)
    
    # Background: Pure black
    if force_background_black:
        result[background_mask] = 0.0
    
    # Grey matter: Traditional dark field effect (inverted, dimmed)
    if np.any(grey_matter_mask):
        grey_values = slice_clahe[grey_matter_mask]
        inverted_grey = 1 - grey_values
        processed_grey = np.power(inverted_grey, intensity_params['gamma'])
        
        # Apply gentle contrast enhancement
        if len(processed_grey) > 0:
            p_low, p_high = intensity_params['contrast_stretch']
            p1, p99 = np.percentile(processed_grey, (p_low, p_high))
            if p99 > p1:
                processed_grey = exposure.rescale_intensity(processed_grey, in_range=(p1, p99))
        
        # Dim grey matter to be darker
        processed_grey *= 0.4 * intensity_params['background_boost']
        result[grey_matter_mask] = processed_grey
    
    # Intermediate tissue: Moderate processing
    if np.any(intermediate_mask):
        intermediate_values = slice_clahe[intermediate_mask]
        # Light inversion with enhancement
        light_inverted = 0.8 - (intermediate_values * 0.6)
        processed_intermediate = np.power(light_inverted, intensity_params['gamma'] * 0.8)
        processed_intermediate *= 0.6 * intensity_params['background_boost']
        result[intermediate_mask] = processed_intermediate
    
    # White matter: Enhanced brightness, minimal inversion
    if np.any(white_matter_mask):
        white_values = slice_clahe[white_matter_mask]
        
        # Instead of inversion, enhance brightness while maintaining contrast
        # Apply gamma correction for brightness enhancement
        enhanced_white = np.power(white_values, 0.7)  # Brighten with gamma < 1
        
        # Apply contrast enhancement specifically for white matter
        if len(enhanced_white) > 0:
            # Use tighter contrast stretch for white matter preservation
            p1, p99 = np.percentile(enhanced_white, (5, 98))
            if p99 > p1:
                enhanced_white = exposure.rescale_intensity(enhanced_white, in_range=(p1, p99))
        
        # Boost white matter brightness significantly - reduce dullness
        white_matter_boost = 0.85 * intensity_params['background_boost']  # Much brighter
        enhanced_white *= white_matter_boost
        
        # Apply additional sharpening to white matter for clarity
        enhanced_white = filters.gaussian(enhanced_white, sigma=0.3)  # Light smoothing
        
        result[white_matter_mask] = enhanced_white
    
    # Very bright areas (potential blockface): Preserve with enhancement
    if np.any(very_bright_mask):
        very_bright_values = slice_clahe[very_bright_mask]
        
        # Minimal processing - just enhance and preserve
        processed_bright = np.power(very_bright_values, 0.6)  # Further brighten
        
        # Apply light contrast enhancement
        if len(processed_bright) > 0:
            p1, p99 = np.percentile(processed_bright, (2, 99))
            if p99 > p1:
                processed_bright = exposure.rescale_intensity(processed_bright, in_range=(p1, p99))
        
        # Maximum brightness preservation
        processed_bright *= 0.9 * intensity_params['background_boost']
        result[very_bright_mask] = processed_bright
    
    # STEP 3: Apply edge enhancement to improve tissue boundaries
    from skimage import feature
    
    # Detect edges for boundary enhancement
    edges = feature.canny(slice_clahe, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
    edges_dilated = morphology.binary_dilation(edges, morphology.disk(1))
    
    # Enhance boundaries between white matter and other tissues
    if np.any(edges_dilated):
        # Boost edge areas slightly for better definition
        edge_boost = 1.15
        result[edges_dilated] *= edge_boost
    
    # STEP 4: Apply tissue-specific smoothing
    
    # Light smoothing for grey matter regions (reduce noise)
    if np.any(grey_matter_mask):
        grey_region = result * grey_matter_mask.astype(float)
        smoothed_grey = filters.gaussian(grey_region, sigma=0.8)
        result[grey_matter_mask] = smoothed_grey[grey_matter_mask]
    
    # Minimal smoothing for white matter (preserve detail)
    if np.any(white_matter_mask):
        white_region = result * white_matter_mask.astype(float)
        smoothed_white = filters.gaussian(white_region, sigma=0.4)
        result[white_matter_mask] = smoothed_white[white_matter_mask]
    
    # STEP 5: Final contrast adjustment and artifact removal
    
    # Apply smart thresholding
    threshold = intensity_params['threshold']
    brain_region = ~background_mask
    
    # Gentle thresholding that preserves white matter
    result[brain_region & (result < threshold/4)] = 0  # More lenient threshold
    
    # Remove small artifacts while preserving anatomical structures
    if np.any(result > 0):
        # Remove very small isolated bright spots (likely artifacts)
        bright_threshold_removal = np.percentile(result[result > 0], 95)
        bright_regions = result > bright_threshold_removal
        from skimage import measure
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        brain_area = np.sum(brain_region)
        min_region_size = max(20, brain_area * 0.0005)  # Smaller threshold to preserve anatomy
        
        for region in regions:
            if region.area < min_region_size:
                # Check if this is in white matter before removing
                region_mask = labeled_bright == region.label
                if np.mean(white_matter_mask[region_mask]) < 0.5:  # Not primarily white matter
                    result[region_mask] = 0.0
    
    # STEP 6: Final brightness and contrast optimization
    
    # Ensure good contrast between tissue types
    if np.any(result > 0):
        # Apply histogram equalization only to brain regions for better contrast
        brain_values = result[brain_region & (result > 0)]
        if len(brain_values) > 100:
            # Gentle histogram equalization
            brain_eq = exposure.equalize_hist(brain_values)
            
            # Blend with original (50% each for natural appearance)
            blending_factor = 0.3  # Less aggressive blending
            result[brain_region & (result > 0)] = (
                (1 - blending_factor) * brain_values + 
                blending_factor * brain_eq
            )
    
    # Final cleanup and normalization
    result = np.clip(result, 0, 1)
    
    return result