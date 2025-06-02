#!/usr/bin/env python
"""
NIfTI Tractography Slice Viewer

This module provides comprehensive visualization capabilities for NIfTI images with overlaid
tractography data. It supports multiple viewing orientations, advanced contrast enhancement,
and synthetic dataset generation for machine learning applications.

Key Features:
- Dark field microscopy-style visualization
- Multi-view support (axial, coronal, sagittal)
- Ground truth mask generation for ML training
- Adaptive contrast enhancement with multiple methods
- Smart brain masking to preserve internal structures
- Spatial ROI selection and fiber density variation

Authors: Sparsh Makharia, LINC Team
License: MIT
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from matplotlib.colors import LinearSegmentedColormap
from skimage import exposure, filters, util, morphology, draw, feature, segmentation
from dipy.tracking.streamline import transform_streamlines
from matplotlib.collections import LineCollection
import random
from scipy import ndimage


def select_random_streamlines(streamlines, percentage=10.0, random_state=None):
    """
    Randomly sample a subset of streamlines for visualization or analysis.
    
    This function is particularly useful for:
    - Reducing computational load for large datasets
    - Creating varied visualizations for data augmentation
    - Testing algorithms on representative subsets
    
    Parameters
    ----------
    streamlines : list
        Collection of streamlines to sample from
    percentage : float
        Percentage of streamlines to select (1-100)
    random_state : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    list
        Randomly selected subset of streamlines
    """
    if percentage >= 100.0:
        return streamlines
    
    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    n_select = max(1, int(len(streamlines) * percentage / 100.0))
    indices = np.random.choice(len(streamlines), n_select, replace=False)
    return [streamlines[i] for i in indices]


def select_streamlines_by_sphere(streamlines, center, radius):
    """
    Filter streamlines that pass through a spherical region of interest (ROI).
    
    This is useful for focusing analysis on specific anatomical regions
    or studying connectivity patterns in targeted brain areas.
    
    Parameters
    ----------
    streamlines : list
        Collection of streamlines to filter
    center : tuple or list
        (x, y, z) coordinates of sphere center in voxel space
    radius : float
        Radius of sphere in voxel units
        
    Returns
    -------
    list
        Streamlines that intersect with the spherical ROI
    """
    selected_streamlines = []
    center_coords = np.array(center)
    
    for streamline in streamlines:
        # Calculate minimum distance from any point in streamline to sphere center
        distances = np.sqrt(np.sum((streamline - center_coords)**2, axis=1))
        if np.min(distances) <= radius:
            selected_streamlines.append(streamline)
            
    return selected_streamlines


def create_fiber_mask(streamlines_voxel, slice_idx, orientation='axial', dims=(256, 256, 256), 
                     thickness=1, dilate=True, density_threshold=0.15, gaussian_sigma=2.0,
                     close_gaps=False, closing_footprint_size=5, label_bundles=False,
                     min_bundle_size=20):
    """
    Create a binary mask of fiber bundle outlines in a specific slice.
    Uses a density-based approach to identify and outline fiber bundles rather than 
    individual streamlines.
    
    Parameters
    ----------
    streamlines_voxel : list
        List of streamlines in voxel coordinates
    slice_idx : int
        Slice index
    orientation : str
        'axial', 'coronal', or 'sagittal'
    dims : tuple
        Dimensions of the volume
    thickness : int
        Thickness of the mask lines in pixels
    dilate : bool
        Whether to apply dilation to the mask to create thicker fiber outlines
    density_threshold : float
        Threshold for the density map (0.0-1.0). Higher values create tighter outlines
    gaussian_sigma : float
        Sigma for Gaussian smoothing of density map. Higher values create smoother outlines
    close_gaps : bool
        Whether to apply morphological closing to create contiguous regions
    closing_footprint_size : int
        Size of the footprint for morphological closing operations
    label_bundles : bool
        Whether to perform connected components analysis to label individual bundles
    min_bundle_size : int
        Minimum size (in pixels) for a region to be considered a bundle
    
    Returns
    -------
    ndarray or tuple of ndarrays
        If label_bundles=False: Binary mask showing fiber bundle outlines
        If label_bundles=True: Tuple of (binary_mask, labeled_mask) where labeled_mask
        contains integer IDs for each separate bundle
    """
    # DEBUG: Print the min_bundle_size value received by create_fiber_mask
    print(f"DEBUG: create_fiber_mask received min_bundle_size = {min_bundle_size}")
    
    # Create empty mask
    if orientation == 'axial':
        mask_shape = (dims[0], dims[1])
        axis = 2
    elif orientation == 'coronal':
        mask_shape = (dims[0], dims[2])
        axis = 1
    elif orientation == 'sagittal':
        mask_shape = (dims[1], dims[2])
        axis = 0
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    
    # Create a density map for streamlines
    density_map = np.zeros(mask_shape, dtype=np.float32)
    
    # Distance threshold for including streamlines
    distance_threshold = thickness * 2
    
    # Project streamlines onto the slice
    for sl in streamlines_voxel:
        # Skip if streamline is too short
        if len(sl) < 2:
            continue
            
        # Get coordinates
        coords = sl.astype(np.float32)
        
        # Calculate distance to slice
        distance_to_slice = np.abs(coords[:, axis] - slice_idx)
        min_distance = np.min(distance_to_slice)
        
        # Only include streamlines near this slice
        if min_distance > distance_threshold:
            continue
            
        # Get points that are close to the slice
        mask_points = []
        for i in range(len(coords)):
            if distance_to_slice[i] <= distance_threshold:
                if orientation == 'axial':
                    x, y = int(coords[i, 0]), int(coords[i, 1])
                    if 0 <= x < dims[0] and 0 <= y < dims[1]:
                        mask_points.append((x, y))
                elif orientation == 'coronal':
                    x, z = int(coords[i, 0]), int(coords[i, 2])
                    if 0 <= x < dims[0] and 0 <= z < dims[2]:
                        mask_points.append((x, z))
                elif orientation == 'sagittal':
                    y, z = int(coords[i, 1]), int(coords[i, 2])
                    if 0 <= y < dims[1] and 0 <= z < dims[2]:
                        mask_points.append((y, z))
        
        # Increment density map at each point and along lines between consecutive points
        for j in range(len(mask_points) - 1):
            p1 = mask_points[j]
            p2 = mask_points[j + 1]
            
            # Draw line on density map
            rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
            valid_indices = (
                (rr >= 0) & (rr < mask_shape[0]) & 
                (cc >= 0) & (cc < mask_shape[1])
            )
            if np.any(valid_indices):
                # Increment density along the line
                density_map[rr[valid_indices], cc[valid_indices]] += 1
    
    # If no streamlines found, return empty mask
    if np.max(density_map) == 0:
        if label_bundles:
            return np.zeros(mask_shape, dtype=np.uint8), np.zeros(mask_shape, dtype=np.uint8)
        else:
            return np.zeros(mask_shape, dtype=np.uint8)
    
    # Apply Gaussian smoothing to create a continuous density field
    density_map = filters.gaussian(density_map, sigma=gaussian_sigma)
    
    # Normalize density map to [0,1]
    density_map = density_map / np.max(density_map)
    
    # Threshold the density map to create the fiber bundle mask
    mask = (density_map > density_threshold).astype(np.uint8)
    
    # Apply morphological operations to create nicer outlines
    if dilate and np.any(mask):
        # First apply closing to fill small holes
        mask = morphology.binary_closing(mask, morphology.disk(thickness))
        
        # Then dilate to expand slightly
        mask = morphology.binary_dilation(mask, morphology.disk(thickness))
        
        # Optional: remove small isolated regions
        if np.sum(mask) > 0:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=10).astype(np.uint8)
    
    # Apply additional morphological closing to fill gaps between fibers in the same bundle
    if close_gaps and np.any(mask):
        # Use a larger disk for closing to bridge wider gaps
        closing_footprint = morphology.disk(closing_footprint_size)
        
        # Apply closing operation (dilation followed by erosion)
        mask = morphology.binary_closing(mask, closing_footprint)
        
        # Fill any holes that might be completely enclosed
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=500).astype(np.uint8)
        
        # Remove any small disconnected regions that might be artifacts
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50).astype(np.uint8)
    
    # Apply min_bundle_size filtering to the binary mask regardless of labeling
    if not label_bundles and min_bundle_size > 0 and np.any(mask):
        from skimage import measure
        
        # Label connected components to identify separate bundles
        temp_labeled = measure.label(mask, connectivity=2)
        
        # Get region properties
        regions = measure.regionprops(temp_labeled)
        
        # Create a filtered mask
        filtered_mask = np.zeros_like(mask)
        filtered_regions = 0
        
        # Keep only regions that meet the size threshold
        for region in regions:
            if region.area >= min_bundle_size:
                filtered_mask[temp_labeled == region.label] = 1
                filtered_regions += 1
        
        # Update the mask with filtered regions
        if filtered_regions > 0:
            mask = filtered_mask
        else: # If all regions are filtered out, make mask empty
            mask = np.zeros_like(mask)
    
    # If bundle labeling is requested, perform connected components analysis
    if label_bundles and np.any(mask):
        from skimage import measure
        
        # Use connected components to label different bundles
        labeled_mask = measure.label(mask, connectivity=2)
        
        # Count number of pixels in each labeled region
        regions = measure.regionprops(labeled_mask)
        
        # Filter out small regions that are likely noise or artifacts
        filtered_mask = np.zeros_like(labeled_mask)
        valid_regions = 0
        
        for region in regions:
            if region.area >= min_bundle_size:
                # Keep only regions that meet the size threshold
                filtered_mask[labeled_mask == region.label] = 1
                valid_regions += 1
        
        # Relabel the filtered mask to ensure consecutive labels
        if valid_regions > 0:
            labeled_mask = measure.label(filtered_mask, connectivity=2)
            
            # Create a colored version for visualization (each bundle gets a different color)
            num_labels = np.max(labeled_mask)
            print(f"Found {num_labels} distinct fiber bundles in slice {slice_idx}")
            
            return mask, labeled_mask
        else:
            print(f"No significant fiber bundles found in slice {slice_idx}")
            return mask, np.zeros_like(mask)
    
    return mask


def apply_contrast_enhancement(slice_data, method='clahe', **kwargs):
    """
    Apply contrast enhancement to a slice using various methods.
    
    Parameters
    ----------
    slice_data : ndarray
        2D slice data to enhance
    method : str
        Enhancement method: 'clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'none'
    **kwargs : dict
        Method-specific parameters:
        - For 'clahe': clip_limit, tile_grid_size
        - For 'gamma': gamma_value
        - For 'adaptive_eq': clip_limit
        
    Returns
    -------
    ndarray
        Enhanced slice data normalized to [0, 1]
    """
    # Normalize input to [0, 1] range
    slice_norm = (slice_data - np.min(slice_data)) / (np.ptp(slice_data) + 1e-8)
    
    if method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clip_limit = kwargs.get('clip_limit', 0.01)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit, kernel_size=tile_grid_size)
        
    elif method == 'histogram_eq':
        # Global histogram equalization
        enhanced = exposure.equalize_hist(slice_norm)
        
    elif method == 'adaptive_eq':
        # Adaptive histogram equalization (similar to CLAHE but different implementation)
        clip_limit = kwargs.get('clip_limit', 0.03)
        enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit)
        
    elif method == 'gamma':
        # Gamma correction
        gamma_value = kwargs.get('gamma_value', 1.2)
        enhanced = exposure.adjust_gamma(slice_norm, gamma=gamma_value)
        
    elif method == 'rescale_intensity':
        # Simple intensity rescaling using percentiles
        p_low = kwargs.get('p_low', 2)
        p_high = kwargs.get('p_high', 98)
        p1, p99 = np.percentile(slice_norm, (p_low, p_high))
        enhanced = exposure.rescale_intensity(slice_norm, in_range=(p1, p99))
        
    elif method == 'none':
        # No enhancement, just normalization
        enhanced = slice_norm
        
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")
    
    return enhanced


def apply_clahe_to_slice(slice_data, clip_limit=0.01, tile_grid_size=(8, 8)):
    """
    Legacy function for backward compatibility.
    Apply CLAHE to a slice - use apply_contrast_enhancement instead.
    """
    return apply_contrast_enhancement(slice_data, method='clahe', 
                                    clip_limit=clip_limit, 
                                    tile_grid_size=tile_grid_size)


def _adaptive_morphology_mask(image, initial_threshold=0.03, min_object_size=500, 
                             closing_disk_size=15, opening_disk_size=3, keep_all_components=True):
    """
    Create brain mask using adaptive morphological operations.
    
    This method is excellent for clean separation of brain from background
    while preserving internal structures.
    """
    # Step 1: More conservative initial threshold to capture all brain tissue
    brain_rough = image > initial_threshold
    
    # Step 2: Remove only very small noise objects
    brain_rough = morphology.remove_small_objects(brain_rough, min_size=25)
    
    # Step 3: Fill holes and connect nearby regions (handles CSF spaces)
    brain_filled = morphology.binary_closing(brain_rough, morphology.disk(closing_disk_size))
    
    # Step 4: Gentle smoothing to preserve details
    brain_smooth = morphology.binary_opening(brain_filled, morphology.disk(opening_disk_size))
    
    # Step 5: Handle connected components more carefully
    from skimage import measure
    brain_labels = measure.label(brain_smooth)
    
    if brain_labels.max() > 0:
        if keep_all_components:
            # Keep multiple large components (handles separated brain regions)
            regions = measure.regionprops(brain_labels)
            brain_mask = np.zeros_like(brain_labels, dtype=bool)
            
            # Keep all regions above a certain size threshold
            total_brain_area = np.sum(brain_smooth)
            size_threshold = max(min_object_size, total_brain_area * 0.05)  # At least 5% of total
            
            for region in regions:
                if region.area >= size_threshold:
                    brain_mask[brain_labels == region.label] = True
        else:
            # Original behavior - keep only largest
            largest_region = np.argmax(np.bincount(brain_labels.flat)[1:]) + 1
            brain_mask = (brain_labels == largest_region)
    else:
        brain_mask = brain_smooth
    
    # Step 6: Final cleanup - remove only very small objects
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size//4)
    
    return brain_mask.astype(np.uint8)


def _edge_based_mask(image, sigma=1.5, low_threshold=0.01, high_threshold=0.08, 
                    dilation_size=8, min_object_size=500, keep_all_components=True):
    """
    Create brain mask using edge detection and region growing.
    
    Excellent for preserving fine details and handling gradual intensity changes.
    """
    # Step 1: Smooth image to reduce noise (less aggressive)
    smoothed = filters.gaussian(image, sigma=sigma)
    
    # Step 2: Detect edges using Canny (more sensitive)
    edges = feature.canny(smoothed, sigma=sigma/2, low_threshold=low_threshold, 
                         high_threshold=high_threshold)
    
    # Step 3: Create initial mask from non-zero intensities (more inclusive)
    initial_mask = smoothed > np.percentile(smoothed[smoothed > 0], 5)  # Lower percentile
    
    # Step 4: Use edges to refine the mask boundaries (less aggressive)
    # Dilate edges to create boundary zones
    edge_zones = morphology.binary_dilation(edges, morphology.disk(1))  # Smaller dilation
    
    # Step 5: Combine initial mask with edge information (more conservative removal)
    brain_mask = initial_mask.copy()
    
    # Remove edge zones that are likely external boundaries (more careful)
    external_edges = edge_zones & (~morphology.binary_erosion(initial_mask, morphology.disk(5)))
    brain_mask[external_edges] = False
    
    # Step 6: Fill internal holes and smooth
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(dilation_size))
    
    # Step 7: Handle multiple components
    from skimage import measure
    brain_labels = measure.label(brain_mask)
    
    if brain_labels.max() > 0 and keep_all_components:
        # Keep multiple large components
        regions = measure.regionprops(brain_labels)
        final_mask = np.zeros_like(brain_labels, dtype=bool)
        
        # Keep all regions above size threshold
        total_area = np.sum(brain_mask)
        size_threshold = max(min_object_size, total_area * 0.03)  # At least 3% of total
        
        for region in regions:
            if region.area >= size_threshold:
                final_mask[brain_labels == region.label] = True
        
        brain_mask = final_mask
    elif brain_labels.max() > 0:
        # Keep only largest component
        largest_region = np.argmax(np.bincount(brain_labels.flat)[1:]) + 1
        brain_mask = (brain_labels == largest_region)
    
    # Step 8: Remove only very small objects
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size//4)
    
    return brain_mask.astype(np.uint8)


def _watershed_mask(image, min_distance=20, threshold_rel=0.7, min_object_size=1000):
    """
    Create brain mask using watershed segmentation.
    
    Great for handling complex shapes and multiple tissue types.
    """
    # Step 1: Create distance transform from thresholded image
    threshold = filters.threshold_otsu(image[image > 0]) * threshold_rel
    binary = image > threshold
    
    # Step 2: Distance transform
    distance = ndimage.distance_transform_edt(binary)
    
    # Step 3: Find local maxima (seeds)
    local_maxima = feature.peak_local_maxima(distance, min_distance=min_distance, 
                                           threshold_abs=min_distance/2)
    
    if len(local_maxima[0]) == 0:
        # Fallback to simple thresholding
        return _adaptive_morphology_mask(image, min_object_size=min_object_size)
    
    # Step 4: Create markers
    markers = np.zeros_like(image, dtype=np.int32)
    markers[local_maxima] = np.arange(1, len(local_maxima[0]) + 1)
    
    # Step 5: Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    # Step 6: Combine all regions into single brain mask
    brain_mask = labels > 0
    
    # Step 7: Post-processing
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    
    return brain_mask.astype(np.uint8)


def _combined_mask(image, **kwargs):
    """
    Combine multiple masking approaches for robust results.
    """
    # Get masks from different methods
    morph_mask = _adaptive_morphology_mask(image, **kwargs)
    edge_mask = _edge_based_mask(image, **kwargs)
    
    # Combine using intersection for conservative approach
    combined = morph_mask & edge_mask
    
    # If intersection is too small, use union
    if np.sum(combined) < 0.1 * np.sum(morph_mask):
        combined = morph_mask | edge_mask
        
        # Clean up the union result
        combined = morphology.remove_small_objects(combined, min_size=1000)
        combined = ndimage.binary_fill_holes(combined)
    
    return combined.astype(np.uint8)


def adaptive_ventricle_preservation(image, brain_mask, ventricle_threshold_percentile=5):
    """
    Specifically preserve ventricle regions that might be suppressed.
    
    Parameters
    ----------
    image : ndarray
        Original brain image
    brain_mask : ndarray
        Brain mask
    ventricle_threshold_percentile : float
        Percentile threshold for detecting ventricles within brain
        
    Returns
    -------
    ndarray
        Modified brain mask that preserves ventricle regions
    """
    # Extract brain region intensities
    brain_pixels = image[brain_mask > 0]
    
    if len(brain_pixels) == 0:
        return brain_mask
    
    # Find low-intensity regions within brain (potential ventricles)
    ventricle_threshold = np.percentile(brain_pixels, ventricle_threshold_percentile)
    
    # Identify potential ventricle regions
    potential_ventricles = (image <= ventricle_threshold) & (brain_mask > 0)
    
    # Clean up ventricle regions - remove very small spots
    ventricle_regions = morphology.remove_small_objects(potential_ventricles, min_size=50)
    
    # Create enhanced mask that includes ventricles as valid brain regions
    enhanced_mask = brain_mask.copy()
    enhanced_mask[ventricle_regions] = 1
    
    return enhanced_mask


def apply_smart_dark_field_effect(slice_clahe, intensity_params=None, mask_method='adaptive_morphology',
                                 preserve_ventricles=True, random_state=None, 
                                 mask_threshold=0.01, keep_all_brain_parts=True):
    """
    Enhanced version of apply_dark_field_effect with smart brain masking.
    
    This function sets external background to pure black while preserving
    internal brain structure contrast including ventricles.
    
    Parameters
    ----------
    slice_clahe : ndarray
        CLAHE-processed slice
    intensity_params : dict, optional
        Parameters to control the dark field effect (same as original)
    mask_method : str
        Brain masking method: 'adaptive_morphology', 'edge_based', 'watershed', 'combined'
    preserve_ventricles : bool
        Whether to specifically preserve ventricle regions
    random_state : int, optional
        Random seed for reproducible results
    mask_threshold : float
        Threshold for brain mask creation (lower = more inclusive)
    keep_all_brain_parts : bool
        Whether to keep all brain components (recommended: True)
        
    Returns
    -------
    ndarray
        Dark field processed image with smart background removal
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Default parameters with randomization if not specified
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create smart brain mask with more inclusive parameters
    brain_mask = create_smart_brain_mask(
        slice_clahe, 
        method=mask_method,
        initial_threshold=mask_threshold,
        keep_all_components=keep_all_brain_parts,
        min_object_size=300,  # Smaller minimum size
        closing_disk_size=20, # Larger closing for better connectivity
        opening_disk_size=2   # Smaller opening to preserve details
    )
    
    # Preserve ventricles if requested
    if preserve_ventricles:
        brain_mask = adaptive_ventricle_preservation(slice_clahe, brain_mask, ventricle_threshold_percentile=10)
    
    # Apply dark field effect only to brain regions
    inverted = 1 - slice_clahe
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter to enhance edges while preserving smoothness
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast using percentile-based stretching
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Smart thresholding: different treatment for brain vs background
    threshold = intensity_params['threshold']
    
    # For brain regions: gentle thresholding that preserves structure
    brain_region = dark_field_stretched * brain_mask
    brain_region[brain_region < threshold/3] = 0  # Even more gentle threshold for brain
    
    # For background: complete suppression
    background_region = dark_field_stretched * (~brain_mask.astype(bool))
    background_region[:] = 0  # Pure black background
    
    # Combine brain and background
    result = brain_region + background_region
    
    # Add subtle noise to brain regions only
    if np.any(brain_mask):
        noise_level = random.uniform(0.005, 0.02)
        noise = noise_level * np.random.normal(0, 1, result.shape)
        brain_areas = brain_mask > 0
        result[brain_areas] = np.clip(result[brain_areas] + noise[brain_areas], 0, 1)
    
    return result


def apply_dark_field_effect(slice_clahe, intensity_params=None, random_state=None):
    """
    Apply a dark field microscopy effect with controllable parameters.
    
    ORIGINAL VERSION - for compatibility. Use apply_smart_dark_field_effect for better results.
    
    Parameters
    ----------
    slice_clahe : ndarray
        CLAHE-processed slice
    intensity_params : dict, optional
        Parameters to control the dark field effect:
        - gamma: float, gamma correction value (default: random between 0.8-1.2)
        - threshold: float, threshold for deep black (default: random between 0.01-0.03)
        - contrast_stretch: tuple, percentiles for contrast stretching (default: (0.5, 99.5))
        - background_boost: float, factor to enhance background (default: random between 0.9-1.1)
        - color_scheme: str, 'bw' for black and white, 'blue' for bluish tint (default: random)
        - blue_tint: float, amount of blue tint to apply (default: random between 0.1-0.4)
    random_state : int, optional
        Random seed for reproducible results
    
    Returns
    -------
    ndarray
        Dark field processed image
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Default parameters with randomization if not specified
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.01, 0.03),  # More conservative threshold
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Create a simple brain mask - just based on intensity
    brain_mask = slice_clahe > 0.01  # Conservative brain tissue detection
    
    # Invert colors - dark field has dark background with bright structures
    inverted = 1 - slice_clahe
    
    # Apply gamma correction with parameter
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter to enhance edges while preserving smoothness
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast using percentile-based stretching
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost to control intensity of tissue
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Apply threshold smoothly without creating borders
    threshold = intensity_params['threshold']
    
    # Apply threshold to all areas equally - no special treatment for brain vs background
    dark_field_stretched[dark_field_stretched < threshold] = 0
    
    # Force only true background areas (where original data was zero) to be black
    original_background = slice_clahe <= 0.001
    dark_field_stretched[original_background] = 0
    
    # Add subtle noise to simulate microscopy grain (only to areas with some intensity)
    noise_level = random.uniform(0.005, 0.02)
    noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
    tissue_areas = dark_field_stretched > 0
    dark_field_stretched[tissue_areas] = np.clip(dark_field_stretched[tissue_areas] + noise[tissue_areas], 0, 1)
    
    return dark_field_stretched


def densify_streamline(streamline, step=0.2):
    # Linear interpolation to densify streamline for smoothness
    if len(streamline) < 2:
        return streamline
    diffs = np.cumsum(np.r_[0, np.linalg.norm(np.diff(streamline, axis=0), axis=1)])
    n_points = max(int(diffs[-1] / step), 2)
    new_dists = np.linspace(0, diffs[-1], n_points)
    new_points = np.empty((n_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(new_dists, diffs, streamline[:, i])
    return new_points


def get_intensity_along_streamline(slice_data, x, y):
    # x, y are in voxel coordinates; slice_data is 2D
    x = np.clip(np.round(x).astype(int), 0, slice_data.shape[1] - 1)
    y = np.clip(np.round(y).astype(int), 0, slice_data.shape[0] - 1)
    return slice_data[y, x]


def generate_tract_color_variation(base_color=(1.0, 1.0, 0.0), variation=0.2, random_state=None):
    """
    Generate a variation of the base tract color.
    
    Parameters
    ----------
    base_color : tuple
        Base RGB color (default: yellow)
    variation : float
        Amount of variation to apply (default: 0.2)
    random_state : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    tuple
        Varied RGB color with alpha=1.0
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        
    r, g, b = base_color
    # Apply random variation to each channel
    r_var = np.clip(r + random.uniform(-variation, variation), 0.7, 1.0)
    g_var = np.clip(g + random.uniform(-variation, variation), 0.7, 1.0)
    b_var = np.clip(b + random.uniform(-variation, variation), 0.0, 0.3)  # Keep blue low for yellowish colors
    
    return (r_var, g_var, b_var)


def get_colormap(color_scheme='bw', blue_tint=0.3):
    """
    Get appropriate colormap for dark field visualization
    
    Parameters
    ----------
    color_scheme : str
        'bw' for black and white, 'blue' for bluish-gray tint
    blue_tint : float
        Amount of blue tint to apply (0.0-1.0), higher values increase blue intensity
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Colormap for visualization
    """
    if color_scheme == 'blue':
        # Bluish-gray colormap for dark field effect
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_blue', [
            (0, 0, 0),                                      # Pure black
            (0.35, 0.35, 0.35 + blue_tint),                 # Dark gray with blue tint
            (0.55, 0.55, 0.55 + min(0.45, blue_tint * 1.5)) # Light gray with blue tint
        ], N=256)
    else:  # 'bw' or any other value defaults to black and white
        # Pure black and white colormap
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_bw', [
            (0, 0, 0),       # Pure black
            (0.4, 0.4, 0.4), # Dark gray
            (0.7, 0.7, 0.7)  # Light gray
        ], N=256)
    
    return dark_field_cmap


def visualize_nifti_with_trk(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None):
    """
    Visualize multiple axial slices of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    n_slices : int, optional
        Number of slices to display (default: 1)
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01) - deprecated, use contrast_params
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8) - deprecated, use contrast_params
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    slice_idx : int, optional
        Specific slice index to visualize. If None, slices will be evenly spaced.
    streamline_percentage : float, optional
        Percentage of streamlines to randomly select (1-100, default: 100 = all streamlines)
    roi_sphere : tuple, optional
        (center_x, center_y, center_z, radius) for spherical ROI selection in voxel coordinates
    tract_linewidth : float, optional
        Width of the tract lines (default: 1.0). Values less than 1.0 create thinner lines,
        values greater than 1.0 create thicker lines.
    save_masks : bool, optional
        Whether to save ground truth masks for segmentation (default: False)
    mask_thickness : int, optional
        Thickness of the mask lines in pixels (default: 1)
    density_threshold : float, optional
        Threshold for fiber density map (0.0-1.0) (default: 0.15)
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of density map (default: 2.0)
    random_state : int, optional
        Random seed for reproducible results
    close_gaps : bool, optional
        Whether to apply morphological closing to create contiguous regions (default: False)
    closing_footprint_size : int, optional
        Size of the footprint for morphological closing operations (default: 5)
    label_bundles : bool, optional
        Whether to label distinct fiber bundles (default: False)
    min_bundle_size : int, optional
        Minimum size (in pixels) for a region to be considered a bundle (default: 20)
    contrast_method : str, optional
        Contrast enhancement method: 'clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'rescale_intensity', 'none' (default: 'clahe')
    contrast_params : dict, optional
        Parameters for contrast enhancement method. If None, uses clahe_clip_limit and clahe_tile_grid_size for backward compatibility
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Set up contrast enhancement parameters
    if contrast_params is None:
        # Use legacy parameters for backward compatibility
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (clahe_tile_grid_size, clahe_tile_grid_size)
        }
    
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        # Apply streamline selection if requested
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")
            
        if roi_sphere is not None:
            center_x, center_y, center_z, radius = roi_sphere
            print(f"Selecting streamlines passing through sphere at ({center_x}, {center_y}, {center_z}) with radius {radius}")
            streamlines_voxel = select_streamlines_by_sphere(
                streamlines_voxel, (center_x, center_y, center_z), radius
            )
            print(f"Selected {len(streamlines_voxel)} streamlines through ROI")

    # Calculate slice positions or use specific slice
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        # Calculate slice positions for axial view (z-axis)
        slice_positions = np.linspace(dims[2] // 4, 3 * dims[2] // 4, n_slices).astype(int)

    # Create figure for axial view
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Initialize list to store masks if needed
    fiber_masks = []
    labeled_masks = []

    # Plot each slice with tractography
    for i, slice_idx in enumerate(slice_positions):
        # Use a copy of the intensity parameters for this slice
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        slice_data = nii_data[:, :, slice_idx]
        slice_enhanced = apply_contrast_enhancement(slice_data, method=contrast_method, **contrast_params)
        
        # Apply dark field effect to mimic dark field microscopy
        dark_field_slice = apply_smart_dark_field_effect(slice_enhanced, slice_intensity_params, random_state=random_state)
        
        # Display with colormap for dark field effect
        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bicubic', vmin=0)
        axes[i].set_facecolor('black')  # Set axes background to black
        
        # Create mask for this slice if requested
        if save_masks and has_streamlines:
            if label_bundles:
                # Get both binary mask and labeled mask
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=min_bundle_size
                )
                # Rotate masks to match visualization
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=min_bundle_size
                )
                # Rotate mask to match visualization
                mask = np.rot90(mask)
                fiber_masks.append(mask)
        
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Project all points onto the axial plane
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                # Only include streamlines near this slice
                distance_to_slice = np.abs(z - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:  # Only show streamlines within 2 voxels of the slice
                    continue
                
                # For np.rot90, swap x/y and flip y
                y_plot = dims[1] - y - 1
                points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate varied color for this streamline
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Apply the user-specified linewidth (with a small random variation for realism)
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and fiber_masks:
            mask_dir = os.path.dirname(output_file)
            if not mask_dir:
                mask_dir = "../synthesis"
            mask_basename = os.path.splitext(os.path.basename(output_file))[0]
            
            for i, mask in enumerate(fiber_masks):
                slice_id = slice_positions[i]
                mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_id}.png"
                plt.imsave(mask_filename, mask, cmap='gray')
                print(f"Saved mask for slice {slice_id} to {mask_filename}")
                
                # Save labeled bundles if requested
                if label_bundles and labeled_masks:
                    labeled_mask = labeled_masks[i]
                    # Create a visualization of the labeled bundles
                    labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_id}.png"
                    visualize_labeled_bundles(labeled_mask, labeled_filename)
    else:
        plt.show()

    # Return labeled masks if they were created
    if label_bundles and labeled_masks:
        return fig, axes, fiber_masks, labeled_masks
    else:
        return fig, axes, fiber_masks if save_masks else None


def visualize_nifti_with_trk_coronal(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None):
    """
    Visualize multiple coronal slices of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    n_slices : int, optional
        Number of slices to display (default: 1)
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01) - deprecated, use contrast_params
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8) - deprecated, use contrast_params
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    slice_idx : int, optional
        Specific slice index to visualize. If None, slices will be evenly spaced.
    streamline_percentage : float, optional
        Percentage of streamlines to randomly select (1-100, default: 100 = all streamlines)
    roi_sphere : tuple, optional
        (center_x, center_y, center_z, radius) for spherical ROI selection in voxel coordinates
    tract_linewidth : float, optional
        Width of the tract lines (default: 1.0). Values less than 1.0 create thinner lines,
        values greater than 1.0 create thicker lines.
    save_masks : bool, optional
        Whether to save ground truth masks for segmentation (default: False)
    mask_thickness : int, optional
        Thickness of the mask lines in pixels (default: 1)
    density_threshold : float, optional
        Threshold for fiber density map (0.0-1.0) (default: 0.15)
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of density map (default: 2.0)
    random_state : int, optional
        Random seed for reproducible results
    close_gaps : bool, optional
        Whether to apply morphological closing to create contiguous regions (default: False)
    closing_footprint_size : int, optional
        Size of the footprint for morphological closing operations (default: 5)
    label_bundles : bool, optional
        Whether to label distinct fiber bundles (default: False)
    min_bundle_size : int, optional
        Minimum size (in pixels) for a region to be considered a bundle (default: 20)
    contrast_method : str, optional
        Contrast enhancement method: 'clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'rescale_intensity', 'none' (default: 'clahe')
    contrast_params : dict, optional
        Parameters for contrast enhancement method. If None, uses clahe_clip_limit and clahe_tile_grid_size for backward compatibility
        
    Returns
    -------
    tuple
        If label_bundles=False: (fig, axes, masks)
        If label_bundles=True: (fig, axes, masks, labeled_masks)
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Set up contrast enhancement parameters
    if contrast_params is None:
        # Use legacy parameters for backward compatibility
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (clahe_tile_grid_size, clahe_tile_grid_size)
        }
    
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        # Apply streamline selection if requested
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")
            
        if roi_sphere is not None:
            center_x, center_y, center_z, radius = roi_sphere
            print(f"Selecting streamlines passing through sphere at ({center_x}, {center_y}, {center_z}) with radius {radius}")
            streamlines_voxel = select_streamlines_by_sphere(
                streamlines_voxel, (center_x, center_y, center_z), radius
            )
            print(f"Selected {len(streamlines_voxel)} streamlines through ROI")

    # Calculate slice positions or use specific slice
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        # Calculate slice positions for coronal view (y-axis)
        slice_positions = np.linspace(dims[1] // 4, 3 * dims[1] // 4, n_slices).astype(int)

    # Create figure for coronal view
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Initialize list to store masks if needed
    fiber_masks = []
    labeled_masks = []

    # Plot each slice with tractography
    for i, slice_idx in enumerate(slice_positions):
        # Use a copy of the intensity parameters for this slice
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        # Get coronal slice (x-z plane)
        slice_data = nii_data[:, slice_idx, :]
        slice_enhanced = apply_contrast_enhancement(slice_data, method=contrast_method, **contrast_params)
        
        # Apply dark field effect to mimic dark field microscopy
        dark_field_slice = apply_smart_dark_field_effect(slice_enhanced, slice_intensity_params, random_state=random_state)
        
        # Display with colormap for dark field effect
        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bicubic', vmin=0)
        axes[i].set_facecolor('black')  # Set axes background to black
        
        # Create mask for this slice if requested
        if save_masks and has_streamlines:
            if label_bundles:
                # Get both binary mask and labeled mask
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=min_bundle_size
                )
                # Rotate masks to match visualization
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=min_bundle_size
                )
                # Rotate mask to match visualization
                mask = np.rot90(mask)
                fiber_masks.append(mask)
        
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Extract coordinates
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                # Only include streamlines near this coronal slice
                distance_to_slice = np.abs(y - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:  # Only show streamlines within 2 voxels of the slice
                    continue
                
                # For coronal view, we need x and z coordinates, with z flipped for display
                z_plot = dims[2] - z - 1
                points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate varied color for this streamline
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Apply the user-specified linewidth (with a small random variation for realism)
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and fiber_masks:
            mask_dir = os.path.dirname(output_file)
            if not mask_dir:
                mask_dir = "../synthesis"
            mask_basename = os.path.splitext(os.path.basename(output_file))[0]
            
            for i, mask in enumerate(fiber_masks):
                slice_id = slice_positions[i]
                mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_id}.png"
                plt.imsave(mask_filename, mask, cmap='gray')
                print(f"Saved mask for slice {slice_id} to {mask_filename}")
                
                # Save labeled bundles if requested
                if label_bundles and labeled_masks:
                    labeled_mask = labeled_masks[i]
                    # Create a visualization of the labeled bundles
                    labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_id}.png"
                    visualize_labeled_bundles(labeled_mask, labeled_filename)
                    print(f"Saved labeled bundles for slice {slice_id} to {labeled_filename}")
    else:
        plt.show()

    # Return labeled masks if they were created
    if label_bundles and labeled_masks:
        return fig, axes, fiber_masks, labeled_masks
    else:
        return fig, axes, fiber_masks if save_masks else None


def visualize_multiple_views(nifti_file, trk_file, output_file=None, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None):
    """
    Visualize axial, coronal, and sagittal views of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01)
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8)
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    streamline_percentage : float, optional
        Percentage of streamlines to randomly select (1-100, default: 100 = all streamlines)
    roi_sphere : tuple, optional
        (center_x, center_y, center_z, radius) for spherical ROI selection in voxel coordinates
    tract_linewidth : float, optional
        Width of the tract lines (default: 1.0). Values less than 1.0 create thinner lines,
        values greater than 1.0 create thicker lines.
    save_masks : bool, optional
        Whether to save ground truth masks for segmentation (default: False)
    mask_thickness : int, optional
        Thickness of the mask lines in pixels (default: 1)
    density_threshold : float, optional
        Threshold for fiber density map (0.0-1.0) (default: 0.15)
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of density map (default: 2.0)
    random_state : int, optional
        Random seed for reproducible results
    close_gaps : bool, optional
        Whether to apply morphological closing to create contiguous regions (default: False)
    closing_footprint_size : int, optional
        Size of the footprint for morphological closing operations (default: 5)
    label_bundles : bool, optional
        Whether to label distinct fiber bundles (default: False)
    min_bundle_size : int, optional
        Minimum size (in pixels) for a region to be considered a bundle (default: 20)
    contrast_method : str, optional
        Contrast enhancement method: 'clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'rescale_intensity', 'none' (default: 'clahe')
    contrast_params : dict, optional
        Parameters for contrast enhancement method. If None, uses clahe_clip_limit and clahe_tile_grid_size for backward compatibility
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        # Apply streamline selection if requested
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")
            
        if roi_sphere is not None:
            center_x, center_y, center_z, radius = roi_sphere
            print(f"Selecting streamlines passing through sphere at ({center_x}, {center_y}, {center_z}) with radius {radius}")
            streamlines_voxel = select_streamlines_by_sphere(
                streamlines_voxel, (center_x, center_y, center_z), radius
            )
            print(f"Selected {len(streamlines_voxel)} streamlines through ROI")

    # Create figure with three orientations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Distance threshold for tractography visualization
    distance_threshold = 1.0

    # Use the same parameters for all views for consistency
    axial_params = intensity_params.copy()
    coronal_params = intensity_params.copy()
    sagittal_params = intensity_params.copy()

    # Initialize masks for each view if needed
    fiber_masks = {'axial': None, 'coronal': None, 'sagittal': None}
    labeled_masks = {'axial': None, 'coronal': None, 'sagittal': None}

    # 1. Axial view (middle slice)
    axial_slice_idx = dims[2] // 2
    axial_slice = nii_data[:, :, axial_slice_idx]
    axial_enhanced = apply_contrast_enhancement(axial_slice, method=contrast_method, **contrast_params)
    
    # Apply dark field effect
    axial_dark_field = apply_smart_dark_field_effect(axial_enhanced, axial_params, random_state=random_state)
    
    axes[0].imshow(np.rot90(axial_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='bicubic', vmin=0)
    axes[0].set_facecolor('black')
    
    # Create mask for axial view if requested
    if save_masks and has_streamlines:
        if label_bundles:
            mask, labeled_mask = create_fiber_mask(
                streamlines_voxel, axial_slice_idx, orientation='axial', 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                label_bundles=True, min_bundle_size=min_bundle_size
            )
            # Rotate masks to match visualization
            mask = np.rot90(mask)
            labeled_mask = np.rot90(labeled_mask)
            fiber_masks['axial'] = mask
            labeled_masks['axial'] = labeled_mask
        else:
            mask = create_fiber_mask(
                streamlines_voxel, axial_slice_idx, orientation='axial', 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                min_bundle_size=min_bundle_size
            )
            # Rotate mask to match visualization
            mask = np.rot90(mask)
            fiber_masks['axial'] = mask
    
    if has_streamlines:
        segments = []
        colors = []
        for sl in streamlines_voxel:
            sl_dense = densify_streamline(sl)
            # Filter points near the slice
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(z - axial_slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > distance_threshold:
                continue
                
            # Project onto axial plane
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            y_plot = dims[1] - y - 1
            points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Generate color variation for this streamline
            tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
            
            # Adjust opacity based on distance to slice
            base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
            
            for seg in segs:
                segments.append(seg)
                colors.append(tract_color + (base_opacity,))  # Add variable alpha
                
        if segments:
            # Apply the user-specified linewidth with small random variation
            linewidth = tract_linewidth * random.uniform(0.9, 1.1)
            lc = LineCollection(segments, colors=colors, linewidths=linewidth)
            axes[0].add_collection(lc)
            
        # 2. Coronal view (middle slice)
        coronal_slice_idx = dims[1] // 2
        coronal_slice = nii_data[:, coronal_slice_idx, :]
        coronal_enhanced = apply_contrast_enhancement(coronal_slice, method=contrast_method, **contrast_params)
        
        # Apply dark field effect
        coronal_dark_field = apply_smart_dark_field_effect(coronal_enhanced, coronal_params, random_state=random_state)
        
        axes[1].imshow(np.rot90(coronal_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='bicubic', vmin=0)
        axes[1].set_facecolor('black')

        # Create mask for coronal view if requested
        if save_masks and has_streamlines:
            mask = create_fiber_mask(
                streamlines_voxel, coronal_slice_idx, orientation='coronal', 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                min_bundle_size=min_bundle_size
            )
            # Rotate mask to match visualization
            mask = np.rot90(mask)
            fiber_masks['coronal'] = mask

        # Add streamlines to coronal view
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Filter points near the slice
                y = sl_dense[:, 1]
                distance_to_slice = np.abs(y - coronal_slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > distance_threshold:
                    continue
                    
                # Project points onto coronal plane (x,z)
                x = sl_dense[:, 0]
                z = sl_dense[:, 2]
                # For np.rot90, swap x/z and flip z
                z_plot = dims[2] - z - 1
                points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate color variation
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Apply the user-specified linewidth with small random variation
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[1].add_collection(lc)

        # 3. Sagittal view (middle slice)
        sagittal_slice_idx = dims[0] // 2
        sagittal_slice = nii_data[sagittal_slice_idx, :, :]
        sagittal_enhanced = apply_contrast_enhancement(sagittal_slice, method=contrast_method, **contrast_params)
        
        # Apply dark field effect
        sagittal_dark_field = apply_smart_dark_field_effect(sagittal_enhanced, sagittal_params, random_state=random_state)
        
        axes[2].imshow(np.rot90(sagittal_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='bicubic', vmin=0)
        axes[2].set_facecolor('black')

        # Create mask for sagittal view if requested
        if save_masks and has_streamlines:
            mask = create_fiber_mask(
                streamlines_voxel, sagittal_slice_idx, orientation='sagittal', 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                min_bundle_size=min_bundle_size
            )
            # Rotate mask to match visualization
            mask = np.rot90(mask)
            fiber_masks['sagittal'] = mask

        # Add streamlines to sagittal view
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Filter points near the slice
                x = sl_dense[:, 0]
                distance_to_slice = np.abs(x - sagittal_slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > distance_threshold:
                    continue
                    
                # Project points onto sagittal plane (y,z)
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                # For np.rot90, swap y/z and flip z
                z_plot = dims[2] - z - 1
                points = np.array([y, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate color variation
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Apply the user-specified linewidth with small random variation
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[2].add_collection(lc)

    # Remove axes
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and has_streamlines:
            mask_dir = os.path.dirname(output_file)
            if not mask_dir:
                mask_dir = "../synthesis"
            mask_basename = os.path.splitext(os.path.basename(output_file))[0]
            
            # Save masks for each view
            for view, mask in fiber_masks.items():
                if mask is not None:
                    mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
                    plt.imsave(mask_filename, mask, cmap='gray')
                    print(f"Saved {view} mask to {mask_filename}")
    else:
        plt.show()

    return fig, axes, fiber_masks if save_masks else None


def generate_varied_examples(nifti_file, trk_file, output_dir, n_examples=5, prefix="synthetic_", 
                             slice_mode="coronal", intensity_variation=True, tract_color_variation=True,
                             specific_slice=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1, 
                             min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, use_high_density_masks=False,
                             contrast_method='clahe', contrast_params=None):
    """
    Generate multiple varied examples with different contrast settings
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_dir : str
        Directory to save output images
    n_examples : int, optional
        Number of examples to generate (default: 5)
    prefix : str, optional
        Prefix for output filenames (default: "synthetic_")
    slice_mode : str, optional
        "axial", "coronal", or "all" for multiple views (default: "coronal")
    intensity_variation : bool, optional
        Whether to vary intensity parameters (default: True)
    tract_color_variation : bool, optional
        Whether to vary tract colors (default: True)
    specific_slice : int, optional
        If provided, use this specific slice index for all examples
    streamline_percentage : float, optional
        Percentage of streamlines to randomly select (1-100, default: 100 = all streamlines)
        If min_fiber_percentage and max_fiber_percentage are provided, this is ignored.
    roi_sphere : tuple, optional
        (center_x, center_y, center_z, radius) for spherical ROI selection in voxel coordinates
    tract_linewidth : float, optional
        Width of the tract lines (default: 1.0)
    save_masks : bool, optional
        Whether to save ground truth masks for segmentation (default: False)
    mask_thickness : int, optional
        Thickness of the mask lines in pixels (default: 1)
    min_fiber_percentage : float, optional
        Minimum percentage of fibers to use (default: 10.0)
    max_fiber_percentage : float, optional
        Maximum percentage of fibers to use (default: 100.0)
    density_threshold : float, optional
        Threshold for fiber density map (0.0-1.0) (default: 0.15)
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of density map (default: 2.0)
    random_state : int, optional
        Random seed for reproducible results. If provided, examples will be deterministic.
    close_gaps : bool, optional
        Whether to apply morphological closing to create contiguous regions (default: False)
    closing_footprint_size : int, optional
        Size of the footprint for morphological closing operations (default: 5)
    label_bundles : bool, optional
        Whether to label distinct fiber bundles (default: False)
    min_bundle_size : int, optional
        Minimum size (in pixels) for a region to be considered a bundle (default: 20)
    use_high_density_masks : bool, optional
        Whether to generate masks only from high-density fibers and apply them to all
        density variations (default: False)
    contrast_method : str, optional
        Contrast enhancement method: 'clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'rescale_intensity', 'none' (default: 'clahe')
    contrast_params : dict, optional
        Parameters for contrast enhancement method. If None, uses clahe_clip_limit and clahe_tile_grid_size for backward compatibility
    """
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Set up contrast enhancement parameters
    if contrast_params is None:
        # Use default CLAHE parameters for backward compatibility
        contrast_params = {
            'clip_limit': 0.01,
            'tile_grid_size': (8, 8)
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print the contrast method being used
    print(f"🖼️  Using contrast enhancement method: {contrast_method}")
    
    # Determine the slice index if specific_slice is provided
    if specific_slice is not None:
        # Load the nifti to get dimensions
        nii_img = nib.load(nifti_file)
        dims = nii_img.shape
        
        # Verify the slice index is within bounds
        if slice_mode == "axial" and (specific_slice < 0 or specific_slice >= dims[2]):
            print(f"Warning: Slice index {specific_slice} out of bounds for axial dimension {dims[2]}")
            specific_slice = dims[2] // 2
        elif slice_mode == "coronal" and (specific_slice < 0 or specific_slice >= dims[1]):
            print(f"Warning: Slice index {specific_slice} out of bounds for coronal dimension {dims[1]}")
            specific_slice = dims[1] // 2
    
    # Use fiber density range if provided
    use_fiber_range = (min_fiber_percentage > 0 and max_fiber_percentage > min_fiber_percentage)
    
    # For high-density mask approach, first generate masks using maximum fiber density
    high_density_masks = {}
    high_density_labeled_masks = {}
    
    if use_high_density_masks and save_masks:
        print(f"Generating high-density masks using {max_fiber_percentage}% of fibers...")
        
        # Create a temporary example using maximum fiber percentage
        high_density_example_random_state = random_state if random_state is not None else 42
        high_density_intensity_params = {
            'gamma': 1.0,
            'threshold': 0.05,
            'contrast_stretch': (0.5, 99.5),
            'background_boost': 1.0,
            'color_scheme': 'bw',
            'blue_tint': 0.0
        }
        
        # Generate temporary high-density examples to extract masks
        if slice_mode == "axial":
            # Generate axial high-density mask
            temp_output = os.path.join(output_dir, f"{prefix}_temp_high_density.png")
            result = visualize_nifti_with_trk(
                nifti_file, trk_file, temp_output, n_slices=1,
                intensity_params=high_density_intensity_params,
                tract_color_base=(1.0, 1.0, 0.0), 
                tract_color_variation=0.0,
                slice_idx=specific_slice,
                streamline_percentage=max_fiber_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=True,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=high_density_example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Handle result based on label_bundles
            if label_bundles:
                _, _, masks, labeled_masks = result
            else:
                _, _, masks = result
                labeled_masks = None
            
            # Store masks for each slice position
            if specific_slice is not None:
                slice_positions = [specific_slice]
            else:
                # Calculate slice positions for axial view (z-axis)
                nii_img = nib.load(nifti_file)
                dims = nii_img.shape
                slice_positions = np.linspace(dims[2] // 4, 3 * dims[2] // 4, 1).astype(int)
                
            for i, slice_idx in enumerate(slice_positions):
                high_density_masks[f"axial_{slice_idx}"] = masks[i]
                if label_bundles and labeled_masks is not None:
                    # Apply min_bundle_size filtering
                    from skimage import measure
                    labeled_mask = labeled_masks[i]
                    
                    # Create a filtered mask with only large enough bundles
                    filtered_mask = np.zeros_like(labeled_mask)
                    regions = measure.regionprops(labeled_mask)
                    valid_regions = 0
                    
                    for region in regions:
                        if region.area >= min_bundle_size:
                            # Keep only regions that meet the size threshold
                            filtered_mask[labeled_mask == region.label] = 1
                            valid_regions += 1
                    
                    # Relabel the filtered mask
                    if valid_regions > 0:
                        filtered_labeled_mask = measure.label(filtered_mask, connectivity=2)
                        print(f"Applied min_bundle_size={min_bundle_size} filter to axial_{slice_idx} mask: {np.max(labeled_mask)} → {np.max(filtered_labeled_mask)} bundles")
                        high_density_labeled_masks[f"axial_{slice_idx}"] = filtered_labeled_mask
                    else:
                        print(f"No bundles in axial_{slice_idx} meet the min_bundle_size={min_bundle_size} threshold")
                        high_density_labeled_masks[f"axial_{slice_idx}"] = np.zeros_like(labeled_mask)
            
            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        elif slice_mode == "coronal":
            # Generate coronal high-density mask
            temp_output = os.path.join(output_dir, f"{prefix}_temp_high_density.png")
            result = visualize_nifti_with_trk_coronal(
                nifti_file, trk_file, temp_output, n_slices=1,
                intensity_params=high_density_intensity_params,
                tract_color_base=(1.0, 1.0, 0.0), 
                tract_color_variation=0.0,
                slice_idx=specific_slice,
                streamline_percentage=max_fiber_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=True,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=high_density_example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Handle result based on label_bundles
            if label_bundles:
                _, _, masks, labeled_masks = result
            else:
                _, _, masks = result
                labeled_masks = None
            
            # Store masks for each slice position
            if specific_slice is not None:
                slice_positions = [specific_slice]
            else:
                # Calculate slice positions for coronal view (y-axis)
                nii_img = nib.load(nifti_file)
                dims = nii_img.shape
                slice_positions = np.linspace(dims[1] // 4, 3 * dims[1] // 4, 1).astype(int)
                
            for i, slice_idx in enumerate(slice_positions):
                high_density_masks[f"coronal_{slice_idx}"] = masks[i]
                if label_bundles and labeled_masks is not None:
                    # Apply min_bundle_size filtering
                    from skimage import measure
                    labeled_mask = labeled_masks[i]
                    
                    # Create a filtered mask with only large enough bundles
                    filtered_mask = np.zeros_like(labeled_mask)
                    regions = measure.regionprops(labeled_mask)
                    valid_regions = 0
                    
                    for region in regions:
                        if region.area >= min_bundle_size:
                            # Keep only regions that meet the size threshold
                            filtered_mask[labeled_mask == region.label] = 1
                            valid_regions += 1
                    
                    # Relabel the filtered mask
                    if valid_regions > 0:
                        filtered_labeled_mask = measure.label(filtered_mask, connectivity=2)
                        print(f"Applied min_bundle_size={min_bundle_size} filter to coronal_{slice_idx} mask: {np.max(labeled_mask)} → {np.max(filtered_labeled_mask)} bundles")
                        high_density_labeled_masks[f"coronal_{slice_idx}"] = filtered_labeled_mask
                    else:
                        print(f"No bundles in coronal_{slice_idx} meet the min_bundle_size={min_bundle_size} threshold")
                        high_density_labeled_masks[f"coronal_{slice_idx}"] = np.zeros_like(labeled_mask)
            
            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        else:  # "all" - multiple views
            # Generate masks for all orientations
            temp_output = os.path.join(output_dir, f"{prefix}_temp_high_density.png")
            result = visualize_multiple_views(
                nifti_file, trk_file, temp_output,
                intensity_params=high_density_intensity_params,
                tract_color_base=(1.0, 1.0, 0.0), 
                tract_color_variation=0.0,
                streamline_percentage=max_fiber_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=True,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=high_density_example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Handle result based on label_bundles
            if label_bundles:
                _, _, view_masks, view_labeled_masks = result
            else:
                _, _, view_masks = result
                view_labeled_masks = None
            
            # Store masks for each view
            for view, mask in view_masks.items():
                high_density_masks[view] = mask
                if label_bundles and view_labeled_masks is not None:
                    # Apply min_bundle_size filtering
                    from skimage import measure
                    labeled_mask = view_labeled_masks[view]
                    
                    # Create a filtered mask with only large enough bundles
                    filtered_mask = np.zeros_like(labeled_mask)
                    regions = measure.regionprops(labeled_mask)
                    valid_regions = 0
                    
                    for region in regions:
                        if region.area >= min_bundle_size:
                            # Keep only regions that meet the size threshold
                            filtered_mask[labeled_mask == region.label] = 1
                            valid_regions += 1
                    
                    # Relabel the filtered mask
                    if valid_regions > 0:
                        filtered_labeled_mask = measure.label(filtered_mask, connectivity=2)
                        print(f"Applied min_bundle_size={min_bundle_size} filter to {view} mask: {np.max(labeled_mask)} → {np.max(filtered_labeled_mask)} bundles")
                        high_density_labeled_masks[view] = filtered_labeled_mask
                    else:
                        print(f"No bundles in {view} meet the min_bundle_size={min_bundle_size} threshold")
                        high_density_labeled_masks[view] = np.zeros_like(labeled_mask)
            
            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        print(f"Generated high-density masks for {len(high_density_masks)} views/slices.")
    
    # Generate all examples with varying fiber densities
    for i in range(n_examples):
        # Create a unique random seed for each example if base random_state is provided
        example_random_state = None
        if random_state is not None:
            example_random_state = random_state + i
        
        # Generate random parameters for this example
        if intensity_variation:
            # Vary color scheme every other example to ensure we get both types
            if i % 3 == 0:
                color_scheme = 'bw'  # Black and white
                blue_tint = 0
            elif i % 3 == 1:
                color_scheme = 'blue'  # Blue tint
                blue_tint = random.uniform(0.2, 0.4)
            else:
                # Random choice with weighted preference for blue
                color_scheme = random.choices(['bw', 'blue'], weights=[0.3, 0.7])[0]
                blue_tint = random.uniform(0.1, 0.4) if color_scheme == 'blue' else 0
            
            intensity_params = {
                'gamma': random.uniform(0.8, 1.2),
                'threshold': random.uniform(0.02, 0.08),
                'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
                'background_boost': random.uniform(0.9, 1.1),
                'color_scheme': color_scheme,
                'blue_tint': blue_tint
            }
        else:
            intensity_params = None
            
        # Generate random tract color if enabled
        if tract_color_variation:
            # Randomly select from a range of yellowish colors
            base_r = random.uniform(0.9, 1.0)
            base_g = random.uniform(0.9, 1.0)
            base_b = random.uniform(0.0, 0.2)
            tract_color_base = (base_r, base_g, base_b)
            color_var = random.uniform(0.05, 0.2)
        else:
            tract_color_base = (1.0, 1.0, 0.0)  # Default yellow
            color_var = 0.0
            
        # Set output filename
        output_file = os.path.join(output_dir, f"{prefix}{i+1:03d}.png")
        
        
        # Vary fiber density if range is specified
        if use_fiber_range:
            # Linear interpolation between min and max
            if n_examples > 1:
                # Map i to a value between min_fiber_percentage and max_fiber_percentage
                t = i / (n_examples - 1)
                fiber_pct = min_fiber_percentage + t * (max_fiber_percentage - min_fiber_percentage)
            else:
                fiber_pct = (min_fiber_percentage + max_fiber_percentage) / 2
            print(f"Example {i+1}: Using {fiber_pct:.1f}% of fibers")
        else:
            fiber_pct = streamline_percentage
        
        # Determine whether to save masks for this example
        should_save_masks = save_masks and (not use_high_density_masks or fiber_pct >= max_fiber_percentage)
        
        # Generate visualization
        if slice_mode == "axial":
            # Use single slice for variation
            n_slices = 1
            result = visualize_nifti_with_trk(
                nifti_file, trk_file, output_file, n_slices=n_slices,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                # Get slice positions
                if specific_slice is not None:
                    slice_positions = [specific_slice]
                else:
                    # Calculate slice positions for axial view (z-axis)
                    nii_img = nib.load(nifti_file)
                    dims = nii_img.shape
                    slice_positions = np.linspace(dims[2] // 4, 3 * dims[2] // 4, n_slices).astype(int)
                
                # Save high-density masks for each slice
                mask_dir = os.path.dirname(output_file)
                if not mask_dir:
                    mask_dir = "../synthesis"
                mask_basename = os.path.splitext(os.path.basename(output_file))[0]
                
                for idx, slice_idx in enumerate(slice_positions):
                    mask_key = f"axial_{slice_idx}"
                    if mask_key in high_density_masks:
                        # Save binary mask
                        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
                        plt.imsave(mask_filename, high_density_masks[mask_key], cmap='gray')
                        print(f"Saved high-density mask for slice {slice_idx} to {mask_filename}")
                        
                        # Save labeled mask if available
                        if label_bundles and mask_key in high_density_labeled_masks:
                            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
                            visualize_labeled_bundles(high_density_labeled_masks[mask_key], labeled_filename)
                            print(f"Saved high-density labeled bundles for slice {slice_idx} to {labeled_filename}")
            
        elif slice_mode == "coronal":
            # Use single slice for variation
            result = visualize_nifti_with_trk_coronal(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                # Get slice positions
                if specific_slice is not None:
                    slice_positions = [specific_slice]
                else:
                    # Calculate slice positions for coronal view (y-axis)
                    nii_img = nib.load(nifti_file)
                    dims = nii_img.shape
                    slice_positions = np.linspace(dims[1] // 4, 3 * dims[1] // 4, 1).astype(int)
                
                # Save high-density masks for each slice
                mask_dir = os.path.dirname(output_file)
                if not mask_dir:
                    mask_dir = "../synthesis"
                mask_basename = os.path.splitext(os.path.basename(output_file))[0]
                
                for idx, slice_idx in enumerate(slice_positions):
                    mask_key = f"coronal_{slice_idx}"
                    if mask_key in high_density_masks:
                        # Save binary mask
                        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
                        plt.imsave(mask_filename, high_density_masks[mask_key], cmap='gray')
                        print(f"Saved high-density mask for slice {slice_idx} to {mask_filename}")
                        
                        # Save labeled mask if available
                        if label_bundles and mask_key in high_density_labeled_masks:
                            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
                            visualize_labeled_bundles(high_density_labeled_masks[mask_key], labeled_filename)
                            print(f"Saved high-density labeled bundles for slice {slice_idx} to {labeled_filename}")
            
        else:  # "all" - multiple views
            result = visualize_multiple_views(
                nifti_file, trk_file, output_file,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                # Save high-density masks for each view
                mask_dir = os.path.dirname(output_file)
                if not mask_dir:
                    mask_dir = "../synthesis"
                mask_basename = os.path.splitext(os.path.basename(output_file))[0]
                
                for view, mask in high_density_masks.items():
                    # Save binary mask
                    mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
                    plt.imsave(mask_filename, mask, cmap='gray')
                    print(f"Saved high-density mask for {view} view to {mask_filename}")
                    
                    # Save labeled mask if available
                    if label_bundles and view in high_density_labeled_masks:
                        labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_{view}.png"
                        visualize_labeled_bundles(high_density_labeled_masks[view], labeled_filename)
                        print(f"Saved high-density labeled bundles for {view} view to {labeled_filename}")
        
        print(f"Generated example {i+1}/{n_examples}: {output_file}")


def visualize_labeled_bundles(labeled_mask, output_file=None, colormap='viridis', background_color='black'):
    """
    Visualize labeled fiber bundles with different colors.
    
    Parameters
    ----------
    labeled_mask : ndarray
        Labeled mask where each bundle has a unique integer ID
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    colormap : str, optional
        Colormap to use for visualization (default: 'viridis')
    background_color : str, optional
        Color for the background (default: 'black')
        
    Returns
    -------
    tuple
        (fig, ax) - figure and axis objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Get colormap with a bit of transparency
    cmap = plt.get_cmap(colormap)
    
    # Count number of unique bundles (excluding background)
    num_bundles = np.max(labeled_mask)
    
    if num_bundles == 0:
        ax.text(0.5, 0.5, "No bundles found", color='white', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        # Create a mask for background (value 0)
        background = (labeled_mask == 0)
        
        # Create a colored image with transparent background
        colored_mask = np.zeros((*labeled_mask.shape, 4), dtype=np.float32)
        
        # Assign colors to each bundle
        for i in range(1, num_bundles + 1):
            bundle_mask = (labeled_mask == i)
            
            # Skip if this bundle is empty (could happen if we removed some)
            if not np.any(bundle_mask):
                continue
                
            # Get a distinct color from the colormap
            color_val = (i - 0.5) / num_bundles  # Spread colors evenly
            rgba = cmap(color_val)
            
            # Apply color to this bundle in the colored mask
            colored_mask[bundle_mask] = rgba
            
        # Make background fully transparent
        colored_mask[background, 3] = 0
        
        # Display the colored mask
        ax.imshow(colored_mask)
        
        # Add a title
        ax.set_title(f"Fiber Bundles ({num_bundles} total)", color='white', fontsize=14)
        
        # Add labels for each bundle
        for i in range(1, num_bundles + 1):
            bundle_mask = (labeled_mask == i)
            if np.any(bundle_mask):
                # Find the center of mass for this bundle
                y_indices, x_indices = np.where(bundle_mask)
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                
                # Add a label at the center of the bundle
                ax.text(center_x, center_y, str(i), color='white', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Labeled bundles visualization saved to {output_file}")
    else:
        plt.show()
        
    return fig, ax


def create_smart_brain_mask(image, method='adaptive_morphology', **kwargs):
    """
    Create an intelligent brain mask that distinguishes external background
    from internal brain structures.
    
    Parameters
    ----------
    image : ndarray
        2D brain slice image
    method : str
        Masking method: 'adaptive_morphology', 'edge_based', 'watershed', 'combined'
    **kwargs : dict
        Method-specific parameters:
        - initial_threshold: float, lower threshold for more inclusive masking (default: 0.03)
        - min_object_size: int, minimum size for brain regions (default: 500)
        - keep_all_components: bool, whether to keep multiple brain components (default: True)
        - closing_disk_size: int, size for morphological closing (default: 15)
        - opening_disk_size: int, size for morphological opening (default: 3)
        
    Returns
    -------
    ndarray
        Binary mask where 1=brain tissue, 0=external background
    """
    # Normalize image to [0, 1]
    image_norm = (image - np.min(image)) / (np.ptp(image) + 1e-8)
    
    # Set default parameters for more inclusive masking
    default_kwargs = {
        'initial_threshold': 0.03,  # Lower threshold
        'min_object_size': 500,     # Smaller minimum size
        'keep_all_components': True, # Keep multiple brain parts
        'closing_disk_size': 15,    # Larger closing for better connectivity
        'opening_disk_size': 3      # Smaller opening to preserve details
    }
    
    # Update with user parameters
    default_kwargs.update(kwargs)
    
    if method == 'adaptive_morphology':
        return _adaptive_morphology_mask(image_norm, **default_kwargs)
    elif method == 'edge_based':
        return _edge_based_mask(image_norm, **default_kwargs)
    elif method == 'watershed':
        return _watershed_mask(image_norm, **default_kwargs)
    elif method == 'combined':
        return _combined_mask(image_norm, **default_kwargs)
    else:
        raise ValueError(f"Unknown masking method: {method}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize nifti slices with overlaid tractography')
    parser.add_argument('nifti_file', help='Path to the nifti file')
    parser.add_argument('trk_file', help='Path to the tractography file')
    parser.add_argument('--output', '-o', help='Output image file (optional)', default=None)
    parser.add_argument('--slices', '-s', type=int, help='Number of slices to display (for axial/coronal view)', default=1)
    parser.add_argument('--slice_idx', '-i', type=int, help='Specific slice index to visualize', default=None)
    parser.add_argument('--mode', '-m', choices=['axial', 'coronal', 'all'], help='Visualization mode', default='coronal')
    parser.add_argument('--clahe_clip_limit', type=float, default=0.01, help='CLAHE clip limit (default: 0.01)')
    parser.add_argument('--clahe_tile_grid_size', type=int, default=8, help='CLAHE tile grid size (default: 8)')
    # Add parameters for contrast enhancement
    parser.add_argument('--contrast_method', '-cm', choices=['clahe', 'histogram_eq', 'adaptive_eq', 'gamma', 'rescale_intensity', 'none'], 
                        default='clahe', help='Contrast enhancement method (default: clahe)')
    parser.add_argument('--gamma_value', type=float, default=1.2, 
                        help='Gamma value for gamma correction (default: 1.2)')
    parser.add_argument('--contrast_p_low', type=float, default=2, 
                        help='Lower percentile for rescale_intensity method (default: 2)')
    parser.add_argument('--contrast_p_high', type=float, default=98, 
                        help='Upper percentile for rescale_intensity method (default: 98)')
    parser.add_argument('--generate_examples', '-g', type=int, default=0, 
                        help='Generate N varied examples (default: 0 = disabled)')
    parser.add_argument('--output_dir', '-d', default='./synthetic_examples',
                        help='Output directory for generated examples (default: ./synthetic_examples)')
    parser.add_argument('--color_scheme', '-c', choices=['bw', 'blue', 'random'], default='random',
                        help='Color scheme to use (black-white, bluish, or random)')
    parser.add_argument('--blue_tint', '-b', type=float, default=0.3,
                        help='Amount of blue tint (0.0-1.0) when using blue color scheme')
    # Add parameters for streamline selection
    parser.add_argument('--streamline_percentage', type=float, default=100.0, 
                        help='Percentage of streamlines to randomly select (1-100, default: 100 = all streamlines)')
    parser.add_argument('--roi_center', type=float, nargs=3, default=None,
                        help='Center coordinates (x,y,z) for ROI sphere in voxel space')
    parser.add_argument('--roi_radius', type=float, default=10.0,
                        help='Radius for ROI sphere in voxel units (default: 10.0)')
    parser.add_argument('--tract_linewidth', type=float, default=1.0,
                        help='Width of the tract lines (default: 1.0)')
    # Add parameters for ground truth mask generation
    parser.add_argument('--save_masks', action='store_true',
                        help='Save ground truth masks for segmentation training')
    parser.add_argument('--mask_thickness', type=int, default=1,
                        help='Thickness of the mask lines in pixels (default: 1)')
    parser.add_argument('--density_threshold', type=float, default=0.15,
                        help='Threshold for fiber density map (0.0-1.0) (default: 0.15)')
    parser.add_argument('--gaussian_sigma', type=float, default=2.0,
                        help='Sigma for Gaussian smoothing of density map (default: 2.0)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for reproducible results')
    parser.add_argument('--close_gaps', action='store_true',
                        help='Apply morphological closing to create contiguous regions for fiber bundles')
    parser.add_argument('--closing_footprint_size', type=int, default=5,
                        help='Size of the footprint for morphological closing operations (default: 5)')
    # Add parameters for fiber density range
    parser.add_argument('--min_fiber_pct', type=float, default=0.0,
                        help='Minimum percentage of fibers to use in examples (default: 0 = use --streamline_percentage)')
    parser.add_argument('--max_fiber_pct', type=float, default=100.0,
                        help='Maximum percentage of fibers to use in examples (default: 100)')
    # Add parameters for bundle labeling
    parser.add_argument('--label_bundles', action='store_true',
                        help='Label distinct fiber bundles (requires --save_masks)')
    parser.add_argument('--min_bundle_size', type=int, default=20,
                        help='Minimum size (in pixels) for a region to be considered a bundle (default: 20)')

    args = parser.parse_args()

    # Validate file existence
    if not os.path.exists(args.nifti_file):
        print(f"Error: Nifti file {args.nifti_file} does not exist")
        exit(1)

    if not os.path.exists(args.trk_file):
        print(f"Error: Tractography file {args.trk_file} does not exist")
        exit(1)

    # Set up color scheme parameters
    if args.color_scheme != 'random':
        intensity_params = {
            'color_scheme': args.color_scheme,
            'blue_tint': args.blue_tint if args.color_scheme == 'blue' else 0.0
        }
    else:
        intensity_params = None  # Use random default

    # Create ROI sphere parameter if center is provided
    roi_sphere = None
    if args.roi_center is not None:
        roi_sphere = (args.roi_center[0], args.roi_center[1], args.roi_center[2], args.roi_radius)
        print(f"Using ROI sphere at ({args.roi_center[0]}, {args.roi_center[1]}, {args.roi_center[2]}) with radius {args.roi_radius}")

    # Set up contrast enhancement parameters
    contrast_params = {}
    if args.contrast_method == 'clahe':
        contrast_params = {
            'clip_limit': args.clahe_clip_limit,
            'tile_grid_size': (args.clahe_tile_grid_size, args.clahe_tile_grid_size)
        }
    elif args.contrast_method == 'gamma':
        contrast_params = {'gamma_value': args.gamma_value}
    elif args.contrast_method == 'rescale_intensity':
        contrast_params = {
            'p_low': args.contrast_p_low,
            'p_high': args.contrast_p_high
        }
    elif args.contrast_method == 'adaptive_eq':
        contrast_params = {'clip_limit': args.clahe_clip_limit}
    # histogram_eq and none don't need parameters

    # Check if bundle labeling is requested but save_masks isn't enabled
    if args.label_bundles and not args.save_masks:
        print("Warning: --label_bundles requires --save_masks; enabling mask saving automatically")
        args.save_masks = True

    # Generate examples if requested
    if args.generate_examples > 0:
        generate_varied_examples(
            args.nifti_file, args.trk_file, args.output_dir, 
            n_examples=args.generate_examples, 
            slice_mode=args.mode,
            specific_slice=args.slice_idx,
            streamline_percentage=args.streamline_percentage,
            roi_sphere=roi_sphere,
            tract_linewidth=args.tract_linewidth,
            save_masks=args.save_masks,
            mask_thickness=args.mask_thickness,
            min_fiber_percentage=args.min_fiber_pct,
            max_fiber_percentage=args.max_fiber_pct,
            density_threshold=args.density_threshold,
            gaussian_sigma=args.gaussian_sigma,
            random_state=args.random_state,
            close_gaps=args.close_gaps,
            closing_footprint_size=args.closing_footprint_size,
            label_bundles=args.label_bundles,
            min_bundle_size=args.min_bundle_size,
            contrast_method=args.contrast_method,
            contrast_params=contrast_params
        )
    else:
        # Run visualization based on mode
        if args.mode == 'axial':
            visualize_nifti_with_trk(
                args.nifti_file, args.trk_file, args.output, args.slices,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params,
                slice_idx=args.slice_idx,
                streamline_percentage=args.streamline_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=args.tract_linewidth,
                save_masks=args.save_masks,
                mask_thickness=args.mask_thickness,
                density_threshold=args.density_threshold,
                gaussian_sigma=args.gaussian_sigma,
                random_state=args.random_state,
                close_gaps=args.close_gaps,
                closing_footprint_size=args.closing_footprint_size,
                label_bundles=args.label_bundles,
                min_bundle_size=args.min_bundle_size,
                contrast_method=args.contrast_method,
                contrast_params=contrast_params
            )
        elif args.mode == 'coronal':
            visualize_nifti_with_trk_coronal(
                args.nifti_file, args.trk_file, args.output, args.slices,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params,
                slice_idx=args.slice_idx,
                streamline_percentage=args.streamline_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=args.tract_linewidth,
                save_masks=args.save_masks,
                mask_thickness=args.mask_thickness,
                density_threshold=args.density_threshold,
                gaussian_sigma=args.gaussian_sigma,
                random_state=args.random_state,
                close_gaps=args.close_gaps,
                closing_footprint_size=args.closing_footprint_size,
                label_bundles=args.label_bundles,
                min_bundle_size=args.min_bundle_size,
                contrast_method=args.contrast_method,
                contrast_params=contrast_params
            )
        else:
            visualize_multiple_views(
                args.nifti_file, args.trk_file, args.output,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params,
                streamline_percentage=args.streamline_percentage,
                roi_sphere=roi_sphere,
                tract_linewidth=args.tract_linewidth,
                save_masks=args.save_masks,
                mask_thickness=args.mask_thickness,
                density_threshold=args.density_threshold,
                gaussian_sigma=args.gaussian_sigma,
                random_state=args.random_state,
                close_gaps=args.close_gaps,
                closing_footprint_size=args.closing_footprint_size,
                label_bundles=args.label_bundles,
                min_bundle_size=args.min_bundle_size,
                contrast_method=args.contrast_method,
                contrast_params=contrast_params
            )
