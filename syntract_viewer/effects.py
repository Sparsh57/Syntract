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
    Apply dark field effect like balanced version but preserve bright blockface areas.
    
    This function uses the same dark processing as balanced but selectively preserves
    very bright areas that represent blockface tissue.
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
    
    # Identify very bright blockface areas to preserve
    blockface_threshold = 0.4  # Only very bright areas count as blockface
    blockface_areas = slice_clahe > blockface_threshold
    
    # Create brain mask like balanced version
    basic_threshold = 0.015
    brain_mask_basic = slice_clahe > basic_threshold
    bright_areas = slice_clahe > 0.3
    brain_mask = brain_mask_basic | bright_areas
    
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
    
    print(f"   🔍 Blockface preservation:")
    print(f"      Blockface areas: {np.sum(blockface_areas):,} pixels ({100*np.sum(blockface_areas)/slice_clahe.size:.1f}%)")
    print(f"      Brain areas: {np.sum(brain_mask):,} pixels ({100*np.sum(brain_mask)/slice_clahe.size:.1f}%)")
    
    # Start with full balanced dark field processing for the dark aesthetic
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    dark_field = filters.gaussian(dark_field, sigma=0.6)
    
    # Enhance contrast like balanced version
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Apply much more aggressive thresholding to make soft tissues blacker
    threshold = intensity_params['threshold'] * 3.0  # Much higher threshold
    soft_threshold_range = threshold * 1.2  # Smaller transition zone
    transition_mask = (dark_field_stretched >= threshold) & (dark_field_stretched <= soft_threshold_range)
    below_threshold = dark_field_stretched < threshold
    
    if np.any(transition_mask):
        transition_factor = (dark_field_stretched[transition_mask] - threshold) / (soft_threshold_range - threshold)
        transition_factor = np.power(transition_factor, 0.5)  # More aggressive transition
        dark_field_stretched[transition_mask] *= transition_factor
    
    dark_field_stretched[below_threshold] = 0
    
    # Make non-blockface areas much darker
    non_blockface_mask = ~blockface_areas
    dark_field_stretched[non_blockface_mask] *= 0.3  # Make soft tissues 70% darker
    
    # Apply brain mask with edge softening like balanced
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
    
    # NOW preserve blockface areas by selectively replacing them with brighter values
    if np.any(blockface_areas):
        # Take original blockface values and enhance them moderately
        blockface_values = slice_clahe[blockface_areas]
        preserved_blockface = np.power(blockface_values, 0.7)  # More enhancement for contrast
        
        # Scale to dark gray range (0.15-0.35) for dark field aesthetic with contrast
        blockface_min, blockface_max = np.min(preserved_blockface), np.max(preserved_blockface)
        if blockface_max > blockface_min:
            preserved_blockface = 0.15 + 0.2 * (preserved_blockface - blockface_min) / (blockface_max - blockface_min)
        else:
            preserved_blockface = np.full_like(preserved_blockface, 0.25)
            
        # Replace the dark field values with preserved blockface values
        dark_field_stretched[blockface_areas] = preserved_blockface
    
    # Apply the same artifact removal as balanced version
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
    
    # Push more soft tissue areas to pure black for maximum contrast
    # Aggressive black threshold for non-blockface areas (lower for darker image)
    soft_tissue_threshold = 0.08
    soft_tissue_areas = (~blockface_areas) & (dark_field_stretched < soft_tissue_threshold)
    dark_field_stretched[soft_tissue_areas] = 0.0
    
    # Final smoothing like balanced
    dark_field_stretched = filters.gaussian(dark_field_stretched, sigma=0.3)
    
    # Apply global darkening to prevent "radiating light" effect
    # Reduce overall exposure while preserving local contrast
    dark_field_stretched = dark_field_stretched * 0.6  # Global 40% darkening
    
    # NOW selectively brighten only the blockface areas for local contrast
    if np.any(blockface_areas):
        # Take original blockface values and enhance them to proper white levels
        blockface_values = slice_clahe[blockface_areas]
        enhanced_blockface = np.power(blockface_values, 0.5)  # More enhancement
        
        # Scale blockface to bright range (0.5-0.9) for proper white appearance
        blockface_min, blockface_max = np.min(enhanced_blockface), np.max(enhanced_blockface)
        if blockface_max > blockface_min:
            bright_blockface = 0.5 + 0.4 * (enhanced_blockface - blockface_min) / (blockface_max - blockface_min)
        else:
            bright_blockface = np.full_like(enhanced_blockface, 0.7)
            
        # Replace blockface areas with bright values
        dark_field_stretched[blockface_areas] = bright_blockface
    
    return dark_field_stretched 