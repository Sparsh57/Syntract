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
    Apply dark field effect while PRESERVING original intensity relationships.
    
    Creates TRUE dark field look (like balanced) but preserves relative intensities:
    - Balanced: inverts (white matter becomes dark, gray matter becomes bright)
    - Preserving: scales down intensities (white matter stays brighter RELATIVE to gray)
    
    Both produce dark field appearance (dim tissue on black background).
    The difference is in relative tissue contrast, not overall brightness.
    """
    if random_state is not None:
        random.seed(random_state)
    
    if intensity_params is None:
        intensity_params = {
            'gamma': 1.2,
            'threshold': 0.02,
            'contrast_stretch': (0.5, 99.5),
            'background_boost': 1.0,
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # CRITICAL: First check if CLAHE made everything too bright
    # If mean intensity is very high, CLAHE over-enhanced - force dark output
    if np.mean(slice_clahe) > 0.4:
        print(f"Warning: CLAHE over-enhanced (mean={np.mean(slice_clahe):.3f}) - forcing very dark output")
        dark_field = np.zeros_like(slice_clahe)
        # Create minimal structure visibility
        dark_field = np.clip(slice_clahe * 0.02, 0, 0.02)
        return dark_field
    
    # Create brain mask with intelligent validation
    basic_threshold = 0.015
    brain_mask_basic = slice_clahe > basic_threshold
    bright_areas = slice_clahe > 0.3
    brain_mask = brain_mask_basic | bright_areas
    
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(2))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
    
    # Start with black canvas
    dark_field = np.zeros_like(slice_clahe)
    
    # Only process within brain mask
    if not np.any(brain_mask):
        return dark_field  # Return black if no brain detected
    
    tissue_values = slice_clahe[brain_mask]
    
    # CRITICAL: Check if this is a low-contrast slice (noise/artifact)
    # Real brain tissue has meaningful intensity variation
    tissue_std = np.std(tissue_values)
    tissue_mean = np.mean(tissue_values)
    
    if tissue_std < 0.02 or tissue_mean < 0.05:
        # Very low contrast or very dark - likely noise or bad slice
        # Return minimal dark field
        dark_field[brain_mask] = 0.01
        return np.clip(dark_field, 0, 0.02)
    
    if tissue_values.max() <= tissue_values.min():
        dark_field[brain_mask] = 0.01
        return np.clip(dark_field, 0, 0.02)
    
    # TRUE DARK FIELD: Scale down all intensities dramatically
    # Normalize to 0-1 within tissue
    normalized = (slice_clahe - tissue_values.min()) / (tissue_values.max() - tissue_values.min())
    
    # Create CONTINUOUS mapping - keep it VERY dark
    dark_field[brain_mask] = normalized[brain_mask] * 0.05
    
    # Apply gamma for contrast
    dark_field = np.power(dark_field, intensity_params['gamma'])
    
    # Heavy smoothing to eliminate blockiness
    dark_field = filters.gaussian(dark_field, sigma=1.5)
    
    # Light contrast stretch ONLY if there's real tissue variation
    if np.any(dark_field[brain_mask] > 0):
        p_low, p_high = intensity_params['contrast_stretch']
        tissue_vals = dark_field[brain_mask]
        
        # Skip contrast stretch if tissue has very low variation (artifact/noise)
        if np.std(tissue_vals) < 0.001:
            # Too uniform - skip stretch to avoid amplifying noise
            pass
        else:
            p1, p99 = np.percentile(tissue_vals, (p_low, p_high))
            
            if p99 > p1 and (p99 - p1) > 0.005:  # Require meaningful range
                # Gentle rescale maintaining the dark field look
                stretched = (dark_field - p1) / (p99 - p1)
                stretched = np.clip(stretched, 0, 1)
                # Keep max VERY low with hard cap
                dark_field = np.minimum(stretched * 0.08, 0.08)
                # Keep non-brain areas black
                dark_field[~brain_mask] = 0.0
    
    # Additional smoothing after contrast stretch
    dark_field = filters.gaussian(dark_field, sigma=0.8)
    
    # CRITICAL: Final safety clamp to prevent white backgrounds
    # Ensure max value never exceeds 0.10 (very dark gray at most)
    dark_field = np.clip(dark_field, 0, 0.10)
    
    # Edge softening (same as balanced)
    brain_distance = ndimage.distance_transform_edt(brain_mask)
    edge_distance = ndimage.distance_transform_edt(~brain_mask)
    
    transition_zone = (edge_distance <= 3) & (brain_distance <= 3)
    if np.any(transition_zone):
        edge_factor = np.minimum(brain_distance[transition_zone] / 3.0, 1.0)
        edge_factor = np.power(edge_factor, 0.7)
        dark_field[transition_zone] *= edge_factor
    
    # Force background black
    clear_background = edge_distance > 3
    dark_field[clear_background] = 0.0
    
    # MINIMAL artifact removal - only remove very bright outliers
    if np.any(dark_field > 0):
        bright_threshold_artifact = np.percentile(dark_field[dark_field > 0], 95)  # 95th instead of 90th
        bright_regions = dark_field > bright_threshold_artifact
        
        labeled_bright = measure.label(bright_regions)
        regions = measure.regionprops(labeled_bright)
        
        brain_area = np.sum(brain_mask)
        min_region_size = max(25, brain_area * 0.0005)
        
        for region in regions:
            # Only remove TINY bright artifacts
            if region.area < max(5, min_region_size * 0.3):
                artifact_mask = labeled_bright == region.label
                y_min, x_min, y_max, x_max = region.bbox
                
                pad = 2  # Smaller padding
                y_start = max(0, y_min - pad)
                y_end = min(dark_field.shape[0], y_max + pad)
                x_start = max(0, x_min - pad)
                x_end = min(dark_field.shape[1], x_max + pad)
                
                surrounding_region = dark_field[y_start:y_end, x_start:x_end]
                surrounding_mask = brain_mask[y_start:y_end, x_start:x_end] & (surrounding_region > 0)
                
                if np.any(surrounding_mask):
                    # Use median only
                    median_val = np.median(surrounding_region[surrounding_mask])
                    dark_field[artifact_mask] = median_val
    
    # Final smoothing
    dark_field = filters.gaussian(dark_field, sigma=0.3)
    
    return np.clip(dark_field, 0, 1) 