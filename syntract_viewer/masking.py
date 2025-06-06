"""
Masking functions for brain segmentation and fiber mask generation.
"""

import numpy as np
from skimage import morphology, filters, feature, draw, measure, segmentation
from scipy import ndimage


def create_fiber_mask(streamlines_voxel, slice_idx, orientation='axial', dims=(256, 256, 256), 
                     thickness=1, dilate=True, density_threshold=0.15, gaussian_sigma=2.0,
                     close_gaps=False, closing_footprint_size=5, label_bundles=False,
                     min_bundle_size=20):
    """
    Create a binary mask of fiber bundle outlines in a specific slice.
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
        
        # Increment density map
        for j in range(len(mask_points) - 1):
            p1 = mask_points[j]
            p2 = mask_points[j + 1]
            
            rr, cc = draw.line(p1[0], p1[1], p2[0], p2[1])
            valid_indices = (
                (rr >= 0) & (rr < mask_shape[0]) & 
                (cc >= 0) & (cc < mask_shape[1])
            )
            if np.any(valid_indices):
                density_map[rr[valid_indices], cc[valid_indices]] += 1
    
    if np.max(density_map) == 0:
        if label_bundles:
            return np.zeros(mask_shape, dtype=np.uint8), np.zeros(mask_shape, dtype=np.uint8)
        else:
            return np.zeros(mask_shape, dtype=np.uint8)
    
    # Apply Gaussian smoothing
    density_map = filters.gaussian(density_map, sigma=gaussian_sigma)
    density_map = density_map / np.max(density_map)
    
    # Threshold the density map
    mask = (density_map > density_threshold).astype(np.uint8)
    
    # Apply morphological operations
    if dilate and np.any(mask):
        mask = morphology.binary_closing(mask, morphology.disk(thickness))
        mask = morphology.binary_dilation(mask, morphology.disk(thickness))
        
        if np.sum(mask) > 0:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=10).astype(np.uint8)
    
    # Apply gap closing
    if close_gaps and np.any(mask):
        closing_footprint = morphology.disk(closing_footprint_size)
        mask = morphology.binary_closing(mask, closing_footprint)
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=500).astype(np.uint8)
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50).astype(np.uint8)
    
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
            return mask, labeled_mask
        else:
            print(f"No significant fiber bundles found in slice {slice_idx}")
            return mask, np.zeros_like(mask)
    
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
    brain_rough = image > initial_threshold
    brain_rough = morphology.remove_small_objects(brain_rough, min_size=25)
    brain_filled = morphology.binary_closing(brain_rough, morphology.disk(closing_disk_size))
    brain_smooth = morphology.binary_opening(brain_filled, morphology.disk(opening_disk_size))
    
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


def adaptive_ventricle_preservation(image, brain_mask, ventricle_threshold_percentile=5):
    """Specifically preserve ventricle regions that might be suppressed."""
    brain_pixels = image[brain_mask > 0]
    
    if len(brain_pixels) == 0:
        return brain_mask
    
    ventricle_threshold = np.percentile(brain_pixels, ventricle_threshold_percentile)
    potential_ventricles = (image <= ventricle_threshold) & (brain_mask > 0)
    ventricle_regions = morphology.remove_small_objects(potential_ventricles, min_size=50)
    
    enhanced_mask = brain_mask.copy()
    enhanced_mask[ventricle_regions] = 1
    
    return enhanced_mask


def create_aggressive_brain_mask(original_slice, augmented_slice):
    """
    Create an aggressive brain mask to suppress Cornucopia noise artifacts.
    """
    original_norm = (original_slice - np.min(original_slice)) / (np.ptp(original_slice) + 1e-8)
    
    brain_threshold = 0.02
    brain_mask = original_norm > brain_threshold
    
    brain_mask = morphology.binary_closing(brain_mask, morphology.disk(10))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=200)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_dilation(brain_mask, morphology.disk(3))
    
    # Additional check for augmented slice
    augmented_norm = (augmented_slice - np.min(augmented_slice)) / (np.ptp(augmented_slice) + 1e-8)
    outside_brain = augmented_norm * (~brain_mask)
    
    outside_values = outside_brain[outside_brain > 0]
    if len(outside_values) > 0 and np.percentile(outside_values, 95) > 0.1:
        brain_mask = morphology.binary_dilation(brain_mask, morphology.disk(5))
    
    return brain_mask.astype(np.uint8) 