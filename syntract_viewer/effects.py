"""
Visual effects for NIfTI tractography visualization.
"""

import numpy as np
from skimage import filters, exposure, morphology, measure
from scipy import ndimage
import random

try:
    from .masking import create_smart_brain_mask, adaptive_ventricle_preservation
except ImportError:
    from masking import create_smart_brain_mask, adaptive_ventricle_preservation


def apply_balanced_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                 force_background_black=True):
    """
    Apply a balanced dark field effect with moderate artifact removal.
    """
    if random_state is not None:
        random.seed(random_state)
    
    if intensity_params is None:
        intensity_params = {
            'gamma': 1.0,  # Fixed value for consistency
            'threshold': 0.02,  # Fixed threshold for consistency
            'contrast_stretch': (0.5, 99.5),  # Fixed balanced stretch
            'background_boost': 1.0,  # Fixed value for consistency
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create brain mask
    basic_threshold = 0.015
    brain_mask_basic = slice_clahe > basic_threshold
    bright_areas = slice_clahe > 0.3
    brain_mask = brain_mask_basic | bright_areas
    
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
    
    # Apply dark field effect
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    dark_field = filters.gaussian(dark_field, sigma=0.6)
    
    # Enhance contrast
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Balanced thresholding
    threshold = intensity_params['threshold']
    soft_threshold_range = threshold * 1.5
    transition_mask = (dark_field_stretched >= threshold) & (dark_field_stretched <= soft_threshold_range)
    below_threshold = dark_field_stretched < threshold
    
    if np.any(transition_mask):
        transition_factor = (dark_field_stretched[transition_mask] - threshold) / (soft_threshold_range - threshold)
        transition_factor = np.power(transition_factor, 0.7)
        dark_field_stretched[transition_mask] *= transition_factor
    
    dark_field_stretched[below_threshold] = 0
    
    # Apply brain mask with edge softening
    background_areas = ~brain_mask.astype(bool)
    brain_distance = ndimage.distance_transform_edt(brain_mask)
    edge_distance = ndimage.distance_transform_edt(~brain_mask) 
    
    transition_zone = (edge_distance <= 2) & (brain_distance <= 2)
    if np.any(transition_zone):
        edge_factor = np.minimum(brain_distance[transition_zone] / 2.0, 1.0)
        edge_factor = np.power(edge_factor, 0.8)
        dark_field_stretched[transition_zone] *= edge_factor
    
    if force_background_black:
        clear_background = edge_distance > 2
        dark_field_stretched[clear_background] = 0.0
    else:
        clear_background = edge_distance > 2
        dark_field_stretched[clear_background] *= 0.2
    
    # Artifact removal
    if np.any(dark_field_stretched > 0):
        bright_threshold = np.percentile(dark_field_stretched[dark_field_stretched > 0], 90)
        bright_regions = dark_field_stretched > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        brain_area = np.sum(brain_mask)
        min_region_size = max(25, brain_area * 0.0005)
        
        for region in regions:
            if region.area < min_region_size:
                artifact_mask = labeled_bright == region.label
                y_min, x_min, y_max, x_max = region.bbox
                
                pad = 4
                y_start = max(0, y_min - pad)
                y_end = min(dark_field_stretched.shape[0], y_max + pad)
                x_start = max(0, x_min - pad)
                x_end = min(dark_field_stretched.shape[1], x_max + pad)
                
                surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                surrounding_mask = brain_mask[y_start:y_end, x_start:x_end]
                
                if np.any(surrounding_mask):
                    median_val = np.median(surrounding_region[surrounding_mask])
                    mean_val = np.mean(surrounding_region[surrounding_mask])
                    replacement_value = 0.7 * median_val + 0.3 * mean_val
                    
                    noise_factor = np.random.uniform(0.97, 1.03)
                    dark_field_stretched[artifact_mask] = replacement_value * noise_factor
        
        # Fill holes
        brain_tissue_mask = (dark_field_stretched > np.percentile(dark_field_stretched[dark_field_stretched > 0], 15)) & brain_mask
        filled_brain = morphology.remove_small_holes(brain_tissue_mask.astype(bool), area_threshold=75)
        holes_to_fill = filled_brain & (~brain_tissue_mask.astype(bool))
        
        if np.any(holes_to_fill):
            surrounding_intensity = np.mean(dark_field_stretched[brain_tissue_mask]) if np.any(brain_tissue_mask) else 0
            try:
                holes_indices = np.where(holes_to_fill)
                num_holes = len(holes_indices[0])
                
                if num_holes > 0:
                    hole_values = surrounding_intensity * np.random.uniform(0.95, 1.05, num_holes)
                    dark_field_stretched[holes_to_fill] = hole_values
            except Exception:
                pass
        
        # Remove dark spots
        if np.any(brain_tissue_mask):
            brain_mean = np.mean(dark_field_stretched[brain_tissue_mask])
            brain_std = np.std(dark_field_stretched[brain_tissue_mask])
            
            dark_threshold = max(0, brain_mean - 1.5 * brain_std)
            dark_spots = (dark_field_stretched < dark_threshold) & brain_mask & (dark_field_stretched > 0)
            
            labeled_dark = measure.label(dark_spots)
            dark_regions = measure.regionprops(labeled_dark)
            
            for region in dark_regions:
                if region.area < min_region_size:
                    dark_spot_mask = labeled_dark == region.label
                    y_min, x_min, y_max, x_max = region.bbox
                    y_pad = max(4, (y_max - y_min))
                    x_pad = max(4, (x_max - x_min))
                    
                    y_start = max(0, y_min - y_pad)
                    y_end = min(dark_field_stretched.shape[0], y_max + y_pad)
                    x_start = max(0, x_min - x_pad)
                    x_end = min(dark_field_stretched.shape[1], x_max + x_pad)
                    
                    surrounding_region = dark_field_stretched[y_start:y_end, x_start:x_end]
                    surrounding_mask = brain_mask[y_start:y_end, x_start:x_end]
                    
                    if np.any(surrounding_mask):
                        replacement_value = np.median(surrounding_region[surrounding_mask])
                        noise_factor = np.random.uniform(0.96, 1.04)
                        dark_field_stretched[dark_spot_mask] = replacement_value * noise_factor
    
    # Final smoothing
    dark_field_stretched = filters.gaussian(dark_field_stretched, sigma=0.3)
    
    return dark_field_stretched


def apply_blockface_preserving_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                                force_background_black=True):
    """
    Apply black background masking like balanced effect but keep entire original image as-is.
    No inversion - just preserve original anatomy with black background.
    """
    if random_state is not None:
        random.seed(random_state)
    
    if intensity_params is None:
        intensity_params = {
            'gamma': 1.0,  # Fixed value for consistency
            'threshold': 0.08,  # MUCH HIGHER threshold - only very bright areas survive
            'contrast_stretch': (0.5, 99.5),  # Fixed balanced stretch
            'background_boost': 1.0,  # Fixed value for consistency
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # PRESERVE ORIGINAL IMAGE - no inversion anywhere, but create dark field appearance
    preserved_image = slice_clahe.copy()

    # Simple background removal - make background black for dark field look
    background_threshold = 0.05
    background_mask = slice_clahe < background_threshold
    preserved_image[background_mask] = 0.0
    
    # Create dark field appearance: make everything very close to black
    # This preserves blockface intensity relationships but creates dark field look
    anatomy_areas = preserved_image >= background_threshold
    if np.any(anatomy_areas):
        # Work directly on the image array to avoid broadcasting issues
        # Very aggressive dark field transformation
        dark_field_gamma = 5.0  # Much higher gamma for extreme darkening
        darkened_image = np.power(preserved_image, dark_field_gamma)
        
        # Only the brightest features remain visible
        visibility_threshold = 0.02  # Very low threshold - only brightest areas survive
        bright_features = darkened_image > visibility_threshold
        
        # Make everything extremely dark - near black
        preserved_image[anatomy_areas] = 0.001  # Almost black base
        
        # Even the "bright" features are kept quite dark
        if np.any(bright_features):
            bright_values = darkened_image[bright_features]
            # Scale down bright features to stay dark but still visible
            preserved_image[bright_features] = bright_values * 0.3  # Scale down to 30%
    
    # Ensure the brightest blockface areas stay prominent but still dark
    very_bright = slice_clahe > 0.8  # Higher threshold - only the absolute brightest areas
    if np.any(very_bright):
        # Keep bright areas visible but still quite dark
        bright_values = slice_clahe[very_bright]
        darkened_bright = np.power(bright_values, 3.0)  # Darken even the brightest areas
        preserved_image[very_bright] = darkened_bright * 0.4  # Scale down to 40%
    
    # Light smoothing to clean up edges
    preserved_image = filters.gaussian(preserved_image, sigma=0.2)
    
    print(f"   NO INVERSION: Original blockface preserved with three-level thresholding")
    
    return preserved_image 