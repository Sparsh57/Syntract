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
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
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
                    
                    noise_factor = np.random.uniform(0.92, 1.08)
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
                    hole_values = surrounding_intensity * np.random.uniform(0.85, 1.15, num_holes)
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
                        noise_factor = np.random.uniform(0.9, 1.1)
                        dark_field_stretched[dark_spot_mask] = replacement_value * noise_factor
    
    # Final smoothing
    dark_field_stretched = filters.gaussian(dark_field_stretched, sigma=0.3)
    
    return dark_field_stretched


def apply_blockface_preserving_dark_field_effect(slice_clahe, intensity_params=None, random_state=None, 
                                                force_background_black=True):
    """
    Apply dark field effect optimized for preserving bright blockface areas.
    """
    if random_state is not None:
        random.seed(random_state)
    
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.7, 1.0),
            'threshold': random.uniform(0.005, 0.015),
            'contrast_stretch': (random.uniform(0.05, 0.5), random.uniform(99.5, 99.95)),
            'background_boost': random.uniform(1.0, 1.2),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create ultra-conservative brain mask that preserves ALL bright areas
    very_bright_threshold = 0.15
    bright_areas = slice_clahe > very_bright_threshold
    
    # Also include moderate brightness areas
    medium_threshold = 0.05
    medium_areas = slice_clahe > medium_threshold
    
    # Very minimal morphological processing
    brain_mask = medium_areas | bright_areas
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=50)
    
    # Apply gentler dark field effect
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    dark_field = filters.gaussian(dark_field, sigma=0.4)
    
    # More conservative contrast enhancement
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Very gentle thresholding to preserve bright areas
    threshold = intensity_params['threshold']
    dark_field_stretched[dark_field_stretched < threshold] = 0
    
    # Preserve bright blockface areas by applying mask very conservatively
    if force_background_black:
        # Only suppress areas that are definitely background
        clear_background = ~brain_mask.astype(bool)
        dark_field_stretched[clear_background] = 0.0
    
    # Minimal artifact removal
    if np.any(dark_field_stretched > 0):
        # Only remove very small isolated spots
        bright_threshold = np.percentile(dark_field_stretched[dark_field_stretched > 0], 95)
        bright_regions = dark_field_stretched > bright_threshold
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        # Very conservative size threshold
        min_region_size = 10
        
        for region in regions:
            if region.area < min_region_size:
                artifact_mask = labeled_bright == region.label
                # Simply set small artifacts to zero instead of interpolation
                dark_field_stretched[artifact_mask] = 0
    
    # Very light final smoothing
    dark_field_stretched = filters.gaussian(dark_field_stretched, sigma=0.2)
    
    return dark_field_stretched 