"""
Masking functions for brain segmentation and fiber mask generation.
"""

import numpy as np
from skimage import morphology, filters, feature, draw, measure, segmentation
from scipy import ndimage
from scipy.signal import find_peaks


def create_fiber_mask(streamlines_voxel, slice_idx, orientation='axial', dims=(256, 256, 256), 
                     thickness=1, dilate=True, density_threshold=0.6, gaussian_sigma=2.0,
                     close_gaps=False, closing_footprint_size=5, label_bundles=False,
                     min_bundle_size=1, static_streamline_threshold=0.1, white_mask_slice=None):
    """
    Create a binary mask of fiber bundle outlines in a specific slice.
    
    Parameters
    ----------
    white_mask_slice : ndarray, optional
        2D white matter mask for the slice. If provided, only draw segments where both 
        endpoints are in white matter (same filtering as visualization).
    """
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
    
    density_map = np.zeros(mask_shape, dtype=np.float32)
    distance_threshold = thickness * 2
    
    # Project streamlines onto the slice
    for sl in streamlines_voxel:
        if len(sl) < 2:
            continue
            
        coords = sl.astype(np.float32)
        distance_to_slice = np.abs(coords[:, axis] - slice_idx)
        min_distance = np.min(distance_to_slice)
        
        if min_distance > distance_threshold:
            continue
            
        mask_points = []
        mask_point_indices = []  # Track original coord indices
        for i in range(len(coords)):
            if distance_to_slice[i] <= distance_threshold:
                if orientation == 'axial':
                    x, y = int(coords[i, 0]), int(coords[i, 1])
                    # ALWAYS use the same coordinate transformation as streamline plotting
                    # This ensures masks and streamlines are in the same coordinate system
                    y_plot = dims[1] - y - 1
                    if 0 <= x < dims[0] and 0 <= y_plot < dims[1]:
                        mask_points.append((x, y_plot))
                        mask_point_indices.append(i)
                elif orientation == 'coronal':
                    x, z = int(coords[i, 0]), int(coords[i, 2])
                    # Use the same coordinate transformation as streamline plotting
                    z_plot = dims[2] - z - 1
                    if 0 <= x < dims[0] and 0 <= z_plot < dims[2]:
                        mask_points.append((x, z_plot))
                        mask_point_indices.append(i)
                elif orientation == 'sagittal':
                    y, z = int(coords[i, 1]), int(coords[i, 2])
                    # ALWAYS use the same coordinate transformation as streamline plotting
                    # This ensures masks and streamlines are in the same coordinate system
                    z_plot = dims[2] - z - 1
                    if 0 <= y < dims[1] and 0 <= z_plot < dims[2]:
                        mask_points.append((y, z_plot))
                        mask_point_indices.append(i)
        
        # Increment density map
        for j in range(len(mask_points) - 1):
            p1 = mask_points[j]
            p2 = mask_points[j + 1]
            
            # Apply white mask filtering at segment level (same as visualization)
            if white_mask_slice is not None:
                # Get original coordinates (before plot transformation)
                if orientation == 'coronal':
                    # Get original indices from coords array
                    orig_idx1 = mask_point_indices[j]
                    orig_idx2 = mask_point_indices[j + 1]
                    orig_z1 = int(coords[orig_idx1, 2])
                    orig_z2 = int(coords[orig_idx2, 2])
                    
                    # Check if both endpoints are in white matter
                    x1_idx = int(np.clip(p1[0], 0, white_mask_slice.shape[0] - 1))
                    z1_idx = int(np.clip(orig_z1, 0, white_mask_slice.shape[1] - 1))
                    x2_idx = int(np.clip(p2[0], 0, white_mask_slice.shape[0] - 1))
                    z2_idx = int(np.clip(orig_z2, 0, white_mask_slice.shape[1] - 1))
                    
                    start_in_wm = white_mask_slice[x1_idx, z1_idx] > 0
                    end_in_wm = white_mask_slice[x2_idx, z2_idx] > 0
                    
                    # Skip segment if neither endpoint is in white matter
                    if not (start_in_wm or end_in_wm):
                        continue
            
            rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
            valid_indices = (
                (rr >= 0) & (rr < mask_shape[0]) & 
                (cc >= 0) & (cc < mask_shape[1])
            )
            if np.any(valid_indices):
                density_map[rr[valid_indices], cc[valid_indices]] += 1
    
    if np.max(density_map) == 0:
        print(f"WARNING: No streamlines found for slice {slice_idx} in {orientation} orientation - mask will be empty")
        if label_bundles:
            return np.zeros(mask_shape, dtype=np.uint8), np.zeros(mask_shape, dtype=np.uint8)
        else:
            return np.zeros(mask_shape, dtype=np.uint8)
    
    # Apply Gaussian smoothing
    density_map = filters.gaussian(density_map, sigma=gaussian_sigma)
    
    # STATIC ABSOLUTE THRESHOLD: Use fixed streamline count instead of relative density
    # This provides consistent, predictable bundle detection across all datasets
    # 
    # Approach:
    # - Set a fixed minimum number of streamlines that must pass through a pixel
    # - No normalization needed - absolute threshold based on actual streamline count
    # - Consistent results regardless of dataset characteristics
    # - Easy to understand and tune
    
    # Static threshold: require at least N streamlines per pixel to consider it a bundle
    # This is an absolute count, not a relative percentage
    # Higher values = more aggressive filtering (fewer bundles detected)
    
    # Apply absolute threshold directly to density map (no normalization)
    mask = (density_map >= static_streamline_threshold).astype(np.uint8)
    
    # Debug: Report mask statistics
    mask_pixels = np.sum(mask > 0)
    if mask_pixels > 0:
        print(f"  Mask generated: {mask_pixels} pixels, max density: {np.max(density_map):.1f}, threshold: {static_streamline_threshold}")
    else:
        print(f"  WARNING: Mask is empty after thresholding! Max density: {np.max(density_map):.1f}, threshold: {static_streamline_threshold}")
    
    # SMART DILATION: Label bundles first, then dilate each independently
    if dilate and np.any(mask):
        # First, label all separate bundles
        labeled_mask = measure.label(mask, connectivity=2)
        num_bundles = labeled_mask.max()
        
        if num_bundles > 0:
            # Dilate each bundle independently to preserve gaps
            dilated_mask = np.zeros_like(mask)
            
            for bundle_id in range(1, num_bundles + 1):
                # Extract this bundle
                bundle = (labeled_mask == bundle_id).astype(np.uint8)
                
                # Light dilation only - don't use closing which fills gaps aggressively
                # Use smaller disk to prevent merging nearby bundles
                bundle_dilated = morphology.binary_dilation(bundle, morphology.disk(thickness))
                
                # Add to result (bundles won't merge because we process them separately)
                dilated_mask = np.maximum(dilated_mask, bundle_dilated.astype(np.uint8))
            
            mask = dilated_mask
            
            # NO extra merging dilation - keep bundles completely separate
        
        # Remove only single-pixel noise (minimal filtering)
        if np.sum(mask) > 0:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=1).astype(np.uint8)
    
    # Apply smooth gap closing
    if close_gaps and np.any(mask):
        mask = _apply_smooth_gap_closing(mask, closing_footprint_size)
    
    # Apply min_bundle_size filtering
    if not label_bundles and min_bundle_size > 0 and np.any(mask):
        temp_labeled = measure.label(mask, connectivity=2)
        regions = measure.regionprops(temp_labeled)
        
        filtered_mask = np.zeros_like(mask)
        filtered_regions = 0
        
        for region in regions:
            if region.area >= min_bundle_size:
                filtered_mask[temp_labeled == region.label] = 1
                filtered_regions += 1
        
        if filtered_regions > 0:
            mask = filtered_mask
        else:
            mask = np.zeros_like(mask)
    
    # Bundle labeling
    if label_bundles and np.any(mask):
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        filtered_mask = np.zeros_like(labeled_mask)
        valid_regions = 0
        
        for region in regions:
            if region.area >= min_bundle_size:
                filtered_mask[labeled_mask == region.label] = 1
                valid_regions += 1
        
        if valid_regions > 0:
            labeled_mask = measure.label(filtered_mask, connectivity=2)
            num_labels = np.max(labeled_mask)
            print(f"Found {num_labels} distinct fiber bundles in slice {slice_idx}")
            return filtered_mask.astype(np.uint8), labeled_mask
        else:
            print(f"No significant fiber bundles found in slice {slice_idx}")
            return np.zeros_like(mask), np.zeros_like(mask)
    
    return mask


def create_smart_brain_mask(image, method='adaptive_morphology', **kwargs):
    """
    Create an intelligent brain mask that distinguishes external background
    from internal brain structures.
    """
    image_norm = (image - np.min(image)) / (np.ptp(image) + 1e-8)
    
    default_kwargs = {
        'initial_threshold': 0.03,
        'min_object_size': 500,
        'keep_all_components': True,
        'closing_disk_size': 15,
        'opening_disk_size': 3
    }
    
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


def _adaptive_morphology_mask(image, initial_threshold=0.03, min_object_size=500, 
                             closing_disk_size=15, opening_disk_size=3, keep_all_components=True):
    """Create brain mask using adaptive morphological operations."""
    # Use much higher threshold to completely avoid background noise
    safe_threshold = max(initial_threshold, 0.03)  # Even higher threshold
    brain_rough = image > safe_threshold
    brain_rough = morphology.remove_small_objects(brain_rough, min_size=100)  # Remove even more small objects
    
    # Be extremely conservative with morphological operations
    brain_filled = morphology.binary_closing(brain_rough, morphology.disk(min(closing_disk_size, 3)))  # Much smaller closing
    brain_smooth = morphology.binary_opening(brain_filled, morphology.disk(max(opening_disk_size, 1)))
    
    # Additional aggressive cleanup to remove isolated regions
    brain_smooth = morphology.remove_small_objects(brain_smooth, min_size=200)
    
    brain_labels = measure.label(brain_smooth)
    
    if brain_labels.max() > 0:
        if keep_all_components:
            regions = measure.regionprops(brain_labels)
            brain_mask = np.zeros_like(brain_labels, dtype=bool)
            
            total_brain_area = np.sum(brain_smooth)
            size_threshold = max(min_object_size, total_brain_area * 0.05)
            
            for region in regions:
                if region.area >= size_threshold:
                    brain_mask[brain_labels == region.label] = True
        else:
            largest_region = np.argmax(np.bincount(brain_labels.flat)[1:]) + 1
            brain_mask = (brain_labels == largest_region)
    else:
        brain_mask = brain_smooth
    
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size//4)
    return brain_mask.astype(np.uint8)


def _edge_based_mask(image, sigma=1.5, low_threshold=0.01, high_threshold=0.08, 
                    dilation_size=8, min_object_size=500, keep_all_components=True):
    """Create brain mask using edge detection and region growing."""
    smoothed = filters.gaussian(image, sigma=sigma)
    edges = feature.canny(smoothed, sigma=sigma/2, low_threshold=low_threshold, 
                         high_threshold=high_threshold)
    
    initial_mask = smoothed > np.percentile(smoothed[smoothed > 0], 5)
    edge_zones = morphology.binary_dilation(edges, morphology.disk(1))
    
    brain_mask = initial_mask.copy()
    external_edges = edge_zones & (~morphology.binary_erosion(initial_mask, morphology.disk(5)))
    brain_mask[external_edges] = False
    
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(dilation_size))
    
    brain_labels = measure.label(brain_mask)
    
    if brain_labels.max() > 0 and keep_all_components:
        regions = measure.regionprops(brain_labels)
        final_mask = np.zeros_like(brain_labels, dtype=bool)
        
        total_area = np.sum(brain_mask)
        size_threshold = max(min_object_size, total_area * 0.03)
        
        for region in regions:
            if region.area >= size_threshold:
                final_mask[brain_labels == region.label] = True
        
        brain_mask = final_mask
    elif brain_labels.max() > 0:
        largest_region = np.argmax(np.bincount(brain_labels.flat)[1:]) + 1
        brain_mask = (brain_labels == largest_region)
    
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size//4)
    return brain_mask.astype(np.uint8)


def _watershed_mask(image, min_distance=20, threshold_rel=0.7, min_object_size=1000):
    """Create brain mask using watershed segmentation."""
    threshold = filters.threshold_otsu(image[image > 0]) * threshold_rel
    binary = image > threshold
    
    distance = ndimage.distance_transform_edt(binary)
    local_maxima = feature.peak_local_maxima(distance, min_distance=min_distance, 
                                           threshold_abs=min_distance/2)
    
    if len(local_maxima[0]) == 0:
        return _adaptive_morphology_mask(image, min_object_size=min_object_size)
    
    markers = np.zeros_like(image, dtype=np.int32)
    markers[local_maxima] = np.arange(1, len(local_maxima[0]) + 1)
    
    labels = segmentation.watershed(-distance, markers, mask=binary)
    brain_mask = labels > 0
    
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=min_object_size)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    
    return brain_mask.astype(np.uint8)


def _combined_mask(image, **kwargs):
    """Combine multiple masking approaches for robust results."""
    morph_mask = _adaptive_morphology_mask(image, **kwargs)
    edge_mask = _edge_based_mask(image, **kwargs)
    
    combined = morph_mask & edge_mask
    
    if np.sum(combined) < 0.1 * np.sum(morph_mask):
        combined = morph_mask | edge_mask
        combined = morphology.remove_small_objects(combined, min_size=1000)
        combined = ndimage.binary_fill_holes(combined)
    
    return combined.astype(np.uint8)


def adaptive_ventricle_preservation(image, brain_mask, ventricle_threshold_percentile=2):
    """Specifically preserve ventricle regions that might be suppressed."""
    brain_pixels = image[brain_mask > 0]
    
    if len(brain_pixels) == 0:
        return brain_mask
    
    # Much more conservative ventricle detection
    ventricle_threshold = np.percentile(brain_pixels, ventricle_threshold_percentile)
    potential_ventricles = (image <= ventricle_threshold) & (brain_mask > 0)
    
    # Remove small objects and apply stricter size requirements
    ventricle_regions = morphology.remove_small_objects(potential_ventricles, min_size=200)
    
    # Additional validation: ventricles should be reasonably central and connected
    if np.any(ventricle_regions):
        # Label connected components
        labeled_ventricles = measure.label(ventricle_regions)
        regions = measure.regionprops(labeled_ventricles)
        
        # Only keep regions that look like actual ventricles
        validated_ventricles = np.zeros_like(ventricle_regions)
        total_brain_area = np.sum(brain_mask)
        
        for region in regions:
            # Size constraints: not too small, not too large
            area_ratio = region.area / total_brain_area
            if area_ratio < 0.001 or area_ratio > 0.15:  # Too small or too large
                continue
            
            # Shape constraints: ventricles should have reasonable aspect ratios
            bbox = region.bbox
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            aspect_ratio = max(height, width) / (min(height, width) + 1e-6)
            if aspect_ratio > 5:  # Too elongated
                continue
            
            # Compactness constraint: ventricles should be reasonably compact
            if region.area > 0 and region.perimeter > 0:
                compactness = (4 * np.pi * region.area) / (region.perimeter ** 2)
                if compactness < 0.1:  # Too irregular
                    continue
            
            # This region passes validation
            validated_ventricles[labeled_ventricles == region.label] = True
        
        ventricle_regions = validated_ventricles
    
    enhanced_mask = brain_mask.copy()
    enhanced_mask[ventricle_regions] = 1
    
    return enhanced_mask


def create_aggressive_brain_mask(original_slice, augmented_slice):
    """
    Create a brain mask to suppress Cornucopia noise artifacts while preserving blockface areas.
    Modified to be less aggressive and preserve bright tissue areas.
    """
    original_norm = (original_slice - np.min(original_slice)) / (np.ptp(original_slice) + 1e-8)
    
    # Use a more sophisticated threshold that preserves bright areas
    # Check for bimodal distribution (brain vs background)
    hist, bin_edges = np.histogram(original_norm[original_norm > 0], bins=50, density=True)
    
    # Find valleys in histogram to determine threshold
    valleys = find_peaks(-hist)[0]
    
    if len(valleys) > 0 and len(bin_edges) > valleys[0]:
        # Use the first valley as threshold, but not too low
        brain_threshold = max(0.01, bin_edges[valleys[0]])
    else:
        # Fallback: use percentile-based threshold that's less aggressive
        brain_threshold = max(0.01, np.percentile(original_norm[original_norm > 0], 5))
    
    # Create initial mask
    brain_mask = original_norm > brain_threshold
    
    # Additional check for very bright areas (potential blockface) - always include them
    bright_areas = original_norm > 0.7  # Very bright areas are likely tissue
    brain_mask = brain_mask | bright_areas
    
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(10))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=200)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_dilation(brain_mask, morphology.disk(3))
    
    # Additional check for augmented slice
    augmented_norm = (augmented_slice - np.min(augmented_slice)) / (np.ptp(augmented_slice) + 1e-8)
    outside_brain = augmented_norm * (~brain_mask)
    
    # Also check for bright areas in augmented slice
    augmented_bright = augmented_norm > 0.6
    brain_mask = brain_mask | augmented_bright
    
    outside_values = outside_brain[outside_brain > 0]
    if len(outside_values) > 0 and np.percentile(outside_values, 95) > 0.1:
        brain_mask = morphology.binary_dilation(brain_mask, morphology.disk(5))
    
    return brain_mask.astype(np.uint8)


def _apply_smooth_gap_closing(mask, closing_footprint_size):
    """
    Apply ultra-smooth, round gap closing that creates organic, blob-like shapes.
    
    This function prioritizes roundness and smoothness over geometric precision,
    creating very organic, flowing boundaries without sharp edges or tears.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask to process
    closing_footprint_size : int
        Size parameter for closing operations
        
    Returns
    -------
    np.ndarray
        Ultra-smoothly gap-closed mask with round boundaries and no tears
    """
    result = mask.astype(bool)
    
    # Step 1: Distance-based smooth closing for ultra-round results and tear prevention
    if np.any(result):
        # Apply multiple progressive gaussian smoothing passes for ultra-round boundaries
        smooth_mask = result.astype(np.float32)
        
        # Apply multiple progressive gaussian smoothing passes for ultra-round boundaries (tear-free)
        base_sigma = max(0.8, closing_footprint_size * 0.12)  # Reduced base sigma to prevent over-smoothing
        for i in range(3):  # Multiple smoothing passes
            sigma = base_sigma * (1 + i * 0.3)  # Gentler progression
            smooth_mask = filters.gaussian(smooth_mask, sigma=sigma)
            
            # Progressive threshold with tear prevention
            threshold = 0.5 - (i * 0.05)  # More conservative thresholds: 0.5, 0.45, 0.4
            result = smooth_mask > threshold
            smooth_mask = result.astype(np.float32)
    
    # Step 2: Progressive circular closing with only circular elements for roundness (tear-free)
    max_size = min(closing_footprint_size, 35)  # Reduced max size to prevent large artifacts
    steps = min(5, max(3, max_size // 5))  # Fewer steps for smoother progression
    
    for i in range(steps):
        # Always use circular footprints for maximum roundness and tear prevention
        size = int(1 + (max_size - 1) * (i + 1) / steps)
        footprint = morphology.disk(size)
        
        # Apply gentle closing to prevent sharp boundaries
        result = morphology.binary_closing(result, footprint)
        
        # Apply smoothing after each step for tear-free edges (reduced smoothing)
        if i < steps - 1:  # Don't smooth on final step
            smooth_temp = filters.gaussian(result.astype(np.float32), sigma=0.5)  # Reduced from 0.8
            result = smooth_temp > 0.5  # More conservative threshold
    
    # Step 3: Gentle gaussian smoothing for ultra-soft, tear-free boundaries
    ultra_smooth = result.astype(np.float32)
    
    # Apply moderate gaussian blur for blob-like appearance (reduced to prevent tears)
    final_sigma = max(1.5, closing_footprint_size * 0.15)  # Reduced sigma
    ultra_smooth = filters.gaussian(ultra_smooth, sigma=final_sigma)
    
    # Use moderate threshold for good roundness without creating tears
    result = ultra_smooth > 0.3  # Conservative threshold
    
    # Step 4: Distance-based smoothing for perfectly round boundaries (tear-free)
    if np.any(result):
        # Use distance transform to create smooth boundaries without tears
        distance = ndimage.distance_transform_edt(~result)
        smooth_boundary_size = max(1, closing_footprint_size * 0.08)  # Reduced boundary size
        smooth_boundary = distance < smooth_boundary_size
        result = result | smooth_boundary
    
    # Step 5: Remove small holes with conservative threshold for blob-like continuity
    hole_threshold = max(150, closing_footprint_size * 12)  # Reduced threshold
    result = morphology.remove_small_holes(result, area_threshold=hole_threshold)
    
    # Step 6: Final gentle smoothing with conservative median filter (tear prevention)
    median_size = min(3, max(1, closing_footprint_size // 12))  # Smaller median filter
    if median_size > 1:
        result = ndimage.median_filter(result.astype(np.uint8), size=median_size).astype(bool)
    
    # Step 7: Final round morphological operations - minimal operations for tear prevention
    final_round_size = min(2, max(1, closing_footprint_size // 20))  # Smaller operations
    if final_round_size > 0:
        # Single pass of small circular operations for tear-free boundaries
        final_footprint = morphology.disk(final_round_size)
        result = morphology.binary_closing(result, final_footprint)
        # Skip opening to prevent creating tears
    
    # Step 8: Final gentle gaussian pass for blob-like smoothness (tear-free)
    final_result = filters.gaussian(result.astype(np.float32), sigma=1.0)  # Reduced from 1.5
    result = final_result > 0.4  # More conservative threshold
    
    # Step 9: Remove small objects but keep size reasonable for continuity
    min_object_size = max(40, closing_footprint_size * 2)  # Reduced minimum size
    result = morphology.remove_small_objects(result, min_size=min_object_size)
    
    return result.astype(np.uint8)