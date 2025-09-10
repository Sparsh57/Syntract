#!/usr/bin/env python
"""
Coronal Slice Selection for Brain Analysis

This module provides functionality to select 50 diverse coronal slices from a 3D brain volume
and extract corresponding streamline subsets. The selection uses stratified random sampling
to ensure good coverage across the anterior-posterior span while maintaining reproducibility.

Key features:
- RAS coordinate system handling
- Brain mask-based valid slice detection
- Stratified random sampling with fixed seed
- World-space slab intersection for streamline filtering
- Comprehensive metadata and validation
"""

import os
import json
import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk, save as save_trk
from nibabel.streamlines.trk import TrkFile, Tractogram
from scipy.ndimage import binary_erosion, binary_dilation, label
import warnings

try:
    from .nifti_preprocessing import resample_nifti
    from .transform import build_new_affine
except ImportError:
    from nifti_preprocessing import resample_nifti
    from transform import build_new_affine


def ensure_ras_orientation(img):
    """
    Ensure the image is in RAS orientation.
    
    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image
        
    Returns
    -------
    nibabel.Nifti1Image
        Image reoriented to RAS
    """
    # Get current orientation
    orig_ornt = nib.orientations.io_orientation(img.affine)
    
    # Define RAS orientation (Right-Anterior-Superior)
    ras_ornt = np.array([[0, 1],   # Right (positive X)
                         [1, 1],   # Anterior (positive Y) 
                         [2, 1]])  # Superior (positive Z)
    
    # Check if already RAS
    if np.array_equal(orig_ornt, ras_ornt):
        return img
    
    # Transform to RAS
    ornt_transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    ras_img = img.as_reoriented(ornt_transform)
    
    print(f"Reoriented image from {nib.orientations.ornt2axcodes(orig_ornt)} to RAS")
    return ras_img


def compute_brain_mask(img_data, threshold_percentile=95, min_component_size=1000):
    """
    Compute brain mask using intensity thresholding and connected components.
    
    Parameters
    ----------
    img_data : np.ndarray
        3D image data
    threshold_percentile : float
        Percentile for intensity thresholding
    min_component_size : int
        Minimum size for connected components
        
    Returns
    -------
    np.ndarray
        Binary brain mask
    """
    # Remove very low intensities (background)
    threshold = np.percentile(img_data[img_data > 0], threshold_percentile / 4)
    binary_mask = img_data > threshold
    
    # Clean up the mask with morphological operations
    binary_mask = binary_erosion(binary_mask, iterations=1)
    binary_mask = binary_dilation(binary_mask, iterations=2)
    
    # Keep only the largest connected component
    labeled_mask, num_labels = label(binary_mask)
    if num_labels > 0:
        component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background
        largest_component = np.argmax(component_sizes) + 1
        binary_mask = (labeled_mask == largest_component)
    
    return binary_mask.astype(np.uint8)


def get_valid_coronal_range(brain_mask, margin_slices=5):
    """
    Get valid coronal slice range from brain mask.
    
    Parameters
    ----------
    brain_mask : np.ndarray
        3D binary brain mask
    margin_slices : int
        Number of slices to remove from each end
        
    Returns
    -------
    tuple
        (y_min, y_max) valid slice indices
    """
    # Project mask along Y axis (coronal direction)
    y_projection = np.any(brain_mask, axis=(0, 2))
    
    # Find first and last slices with brain tissue
    valid_slices = np.where(y_projection)[0]
    
    if len(valid_slices) == 0:
        raise ValueError("No brain tissue found in mask!")
    
    y_min = valid_slices[0] + margin_slices
    y_max = valid_slices[-1] - margin_slices
    
    if y_min >= y_max:
        # Fallback: use smaller margins
        margin_slices = min(2, len(valid_slices) // 10)
        y_min = valid_slices[0] + margin_slices
        y_max = valid_slices[-1] - margin_slices
    
    return y_min, y_max


def stratified_random_selection(y_min, y_max, n_slices=50, seed=57):
    """
    Select slice indices using stratified random sampling.
    
    Parameters
    ----------
    y_min, y_max : int
        Valid slice range
    n_slices : int
        Number of slices to select
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Selected slice indices (sorted)
    """
    np.random.seed(seed)
    
    total_range = y_max - y_min + 1
    
    if total_range < n_slices:
        # Fallback: uniform sampling with replacement
        print(f"Warning: Valid range ({total_range}) < requested slices ({n_slices})")
        print("Using uniform sampling with replacement")
        selected = np.random.choice(range(y_min, y_max + 1), size=n_slices, replace=True)
        return np.sort(np.unique(selected))
    
    # Stratified sampling
    bin_size = total_range / n_slices
    selected_indices = []
    
    for i in range(n_slices):
        bin_start = y_min + int(i * bin_size)
        bin_end = y_min + int((i + 1) * bin_size)
        
        # Ensure we don't go beyond bounds
        bin_end = min(bin_end, y_max + 1)
        
        if bin_start < bin_end:
            # Randomly select one slice from this bin
            selected_slice = np.random.randint(bin_start, bin_end)
            selected_indices.append(selected_slice)
    
    # Remove duplicates and sort
    selected_indices = np.sort(np.unique(selected_indices))
    
    # If we have fewer than requested due to duplicates, add more
    while len(selected_indices) < n_slices and len(selected_indices) < total_range:
        # Add random slices from unselected ones
        all_slices = set(range(y_min, y_max + 1))
        selected_set = set(selected_indices)
        remaining = list(all_slices - selected_set)
        
        if remaining:
            additional = np.random.choice(remaining, 
                                        size=min(n_slices - len(selected_indices), len(remaining)),
                                        replace=False)
            selected_indices = np.sort(np.concatenate([selected_indices, additional]))
        else:
            break
    
    return selected_indices


def slice_indices_to_world_y(slice_indices, affine, volume_shape):
    """
    Convert slice indices to world-space Y coordinates using full affine transformation.
    
    Parameters
    ----------
    slice_indices : array-like
        Slice indices along Y axis
    affine : np.ndarray
        4x4 affine transformation matrix
    volume_shape : tuple
        Shape of the volume (x, y, z) for computing centers
        
    Returns
    -------
    np.ndarray
        World-space Y coordinates
    """
    # Use volume center for X and Z coordinates
    x_center = (volume_shape[0] - 1) / 2.0
    z_center = (volume_shape[2] - 1) / 2.0
    
    world_y = []
    for slice_idx in slice_indices:
        # Create homogeneous voxel coordinate at volume center for this slice
        voxel_coord = np.array([x_center, slice_idx, z_center, 1.0])
        
        # Transform to world coordinates using full affine
        world_coord = affine @ voxel_coord
        
        # Extract Y coordinate
        world_y.append(world_coord[1])
    
    return np.array(world_y)


def filter_streamlines_by_slab(streamlines, world_y, slab_half_thickness=1.0, affine=None):
    """
    Filter streamlines that intersect a world-space slab around a coronal slice.
    
    Parameters
    ----------
    streamlines : list
        List of streamlines (each is an Nx3 array)
    world_y : float
        World-space Y coordinate of the slice
    slab_half_thickness : float
        Half-thickness of the slab in mm
    affine : np.ndarray, optional
        Affine matrix to convert streamlines to world space if needed
        
    Returns
    -------
    list
        Indices of streamlines that intersect the slab
    """
    intersecting_indices = []
    
    for idx, streamline in enumerate(streamlines):
        if len(streamline) == 0:
            continue
            
        # Convert streamline to world coordinates if needed
        if affine is not None:
            # Assuming streamlines are in voxel space
            homogeneous = np.hstack([streamline, np.ones((len(streamline), 1))])
            world_coords = homogeneous @ affine.T
            stream_world = world_coords[:, :3]
        else:
            # Assuming streamlines are already in world space
            stream_world = streamline
        
        # Get Y coordinates of all points in the streamline
        y_coords = stream_world[:, 1]
        
        # Check if any point falls within the slab
        y_min_slab = world_y - slab_half_thickness
        y_max_slab = world_y + slab_half_thickness
        
        if np.any((y_coords >= y_min_slab) & (y_coords <= y_max_slab)):
            intersecting_indices.append(idx)
    
    return intersecting_indices


def extract_coronal_slices_and_streamlines(
    nifti_path,
    trk_path,
    output_dir,
    target_dimensions,  # Target dimensions (x, y, z) - required
    target_voxel_size=None,
    slab_half_thickness=None,  # Will auto-calculate if None
    margin_slices=5,
    n_slices=50,
    use_stratified_sampling=True
):
    """
    Main function to extract diverse coronal slices and matching streamline subsets.
    
    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI file
    trk_path : str
        Path to input TRK file
    output_dir : str
        Output directory
    target_dimensions : tuple
        Target dimensions (x, y, z) for resampling
    target_voxel_size : float or tuple, optional
        Target voxel size for resampling
    slab_half_thickness : float, optional
        Half-thickness of slab for streamline filtering (mm).
        If None, automatically calculated as half the voxel size in Y direction.
    margin_slices : int
        Number of slices to exclude from edges when using stratified sampling
    n_slices : int
        Number of diverse slices to extract (default: 50)
    use_stratified_sampling : bool
        Whether to use stratified sampling for diverse slice selection (default: True)
        
    Returns
    -------
    dict
        Results dictionary with metadata
    """
    print(f"=== Coronal Slice Extraction ===")
    print(f"Input NIfTI: {nifti_path}")
    print(f"Input TRK: {trk_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target dimensions: {target_dimensions}")
    if use_stratified_sampling:
        print(f"Mode: Extract {n_slices} diverse slices using stratified sampling")
    else:
        print(f"Mode: Extract ALL coronal slices ({target_dimensions[1]} total)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and reorient NIfTI to RAS
    print("\n=== Loading and Processing NIfTI ===")
    img = nib.load(nifti_path)
    img_ras = ensure_ras_orientation(img)
    
    # Resample to target dimensions
    old_shape = img_ras.shape[:3]
    old_voxel_sizes = np.array(img_ras.header.get_zooms()[:3])
    
    print(f"Resampling to exact dimensions: {target_dimensions}")
    new_shape = target_dimensions
    
    if target_voxel_size is not None:
        if isinstance(target_voxel_size, (int, float)):
            target_voxel_size = (target_voxel_size,) * 3
        voxel_size_for_affine = target_voxel_size
    else:
        # Calculate voxel size to fit target dimensions
        old_physical_size = old_voxel_sizes * np.array(old_shape)
        voxel_size_for_affine = old_physical_size / np.array(target_dimensions)
        print(f"Calculated voxel size for target dimensions: {voxel_size_for_affine}")
    
    # Build new affine
    new_affine = build_new_affine(
        img_ras.affine, old_shape, voxel_size_for_affine, new_shape
    )
    
    # Resample
    new_data, tmp_mmap = resample_nifti(
        img_ras, new_affine, new_shape, use_gpu=False, max_output_gb=16
    )
    
    # Create new image
    img_resampled = nib.Nifti1Image(new_data, new_affine)
    
    # Clean up temporary file if it exists
    if tmp_mmap and os.path.exists(tmp_mmap):
        os.remove(tmp_mmap)
    
    print(f"Final image shape: {img_resampled.shape[:3]}")
    print(f"Final voxel sizes: {img_resampled.header.get_zooms()[:3]}")
    
    # Auto-calculate slab thickness if not provided
    if slab_half_thickness is None:
        voxel_size_y = img_resampled.header.get_zooms()[1]
        slab_half_thickness = voxel_size_y / 2.0
        print(f"Auto-calculated slab half-thickness: {slab_half_thickness:.3f}mm (= voxel_size_y/2)")
        print(f"This matches the actual slice thickness for precise alignment")
    else:
        print(f"Using provided slab half-thickness: {slab_half_thickness:.3f}mm")
    
    # Select slices based on mode
    if use_stratified_sampling:
        print(f"\n=== Computing Brain Mask for Slice Selection ===")
        brain_mask = compute_brain_mask(new_data, threshold_percentile=85)
        valid_y_min, valid_y_max = get_valid_coronal_range(brain_mask, margin_slices)
        print(f"Valid brain region: Y slices {valid_y_min} to {valid_y_max}")
        
        print(f"\n=== Selecting {n_slices} Diverse Slices ===")
        selected_indices = stratified_random_selection(valid_y_min, valid_y_max, n_slices=n_slices)
        print(f"Selected {len(selected_indices)} diverse slices: {selected_indices}")
        
    else:
        print(f"\n=== Extracting ALL Coronal Slices ===")
        volume_y_dim = img_resampled.shape[1]  # Y dimension of the volume
        selected_indices = np.arange(0, volume_y_dim)  # All slices from 0 to Y_max
        print(f"Extracting all {len(selected_indices)} slices from volume (0 to {volume_y_dim-1})")
    
    print(f"Volume dimensions: {img_resampled.shape[:3]}")
    
    # Convert to world coordinates
    world_y_coords = slice_indices_to_world_y(selected_indices, img_resampled.affine, img_resampled.shape[:3])
    print(f"World Y range: {world_y_coords.min():.2f} to {world_y_coords.max():.2f} mm")
    
    # Load streamlines
    print(f"\n=== Loading Streamlines ===")
    trk_obj = load_trk(trk_path)
    streamlines = trk_obj.streamlines
    print(f"Loaded {len(streamlines)} streamlines")
    
    # Get TRK coordinate information
    print(f"=== Loading TRK Coordinate Information ===")
    original_trk_affine = trk_obj.header.get('vox_to_ras', trk_obj.header.get('voxel_to_rasmm', np.eye(4)))
    resampled_affine = img_resampled.affine
    
    print(f"TRK header affine (voxel_to_rasmm):\n{original_trk_affine}")
    print(f"Resampled image affine:\n{resampled_affine}")
    
    if len(streamlines) > 0:
        sample_streamline = streamlines[0]
        if len(sample_streamline) > 0:
            print(f"Sample streamline coordinate range:")
            print(f"  X: {sample_streamline[:,0].min():.2f} to {sample_streamline[:,0].max():.2f}")
            print(f"  Y: {sample_streamline[:,1].min():.2f} to {sample_streamline[:,1].max():.2f}")
            print(f"  Z: {sample_streamline[:,2].min():.2f} to {sample_streamline[:,2].max():.2f}")
            
            # Check coordinate scale to determine if streamlines are in voxel or world coordinates
            max_coord = np.abs(sample_streamline).max()
            if max_coord < 1000:  # Typical voxel coordinates are < 1000
                print("✓ Streamlines appear to be in voxel coordinates (standard TRK format)")
                streamlines_are_in_ras = False
            else:
                print("⚠ Streamlines appear to be in world/RAS coordinates (non-standard)")
                streamlines_are_in_ras = True
        else:
            streamlines_are_in_ras = False
    else:
        streamlines_are_in_ras = False
    
    # Process each slice
    print(f"\n=== Processing Slices ===")
    per_slice_trk_indices = []
    
    for i, (slice_idx, world_y) in enumerate(zip(selected_indices, world_y_coords)):
        print(f"Processing slice {i+1}/{len(selected_indices)}: index {slice_idx}, Y={world_y:.2f}mm")
        
        # Create subfolder for this slice
        slice_dir = os.path.join(output_dir, f"slice_{slice_idx:03d}")
        os.makedirs(slice_dir, exist_ok=True)
        
        # Filter streamlines based on their coordinate system
        A = img_resampled.affine
        A_inv = np.linalg.inv(A)
        
        intersecting_indices = []
        
        if streamlines_are_in_ras:
            # Streamlines are already in RAS/world coordinates - filter directly in world space
            print(f"  Filtering streamlines in world/RAS coordinates")
            y_min_slab = world_y - slab_half_thickness
            y_max_slab = world_y + slab_half_thickness
            
            for idx, streamline in enumerate(streamlines):
                if len(streamline) == 0:
                    continue
                    
                # Get Y coordinates directly (already in world space)
                y_coords = streamline[:, 1]
                
                # Check if any point falls within the world-space slab
                if np.any((y_coords >= y_min_slab) & (y_coords <= y_max_slab)):
                    intersecting_indices.append(idx)
        else:
            # Streamlines are in voxel coordinates - transform to voxel space for filtering
            print(f"  Filtering streamlines by transforming to voxel coordinates")
            # Convert slab thickness from mm to voxel units
            voxel_size_y_mm = img_resampled.header.get_zooms()[1]
            slab_vox_half = slab_half_thickness / voxel_size_y_mm
            
            for idx, streamline in enumerate(streamlines):
                if len(streamline) == 0:
                    continue
                    
                # Transform streamline points to resampled voxel space
                homogeneous = np.hstack([streamline, np.ones((len(streamline), 1))])
                voxel_coords = homogeneous @ A_inv.T
                p_vox = voxel_coords[:, :3]
                
                # Check if any point is within the slice slab
                y_vox_coords = p_vox[:, 1]
                if np.any(np.abs(y_vox_coords - slice_idx) <= slab_vox_half):
                    intersecting_indices.append(idx)
        
        per_slice_trk_indices.append(intersecting_indices)
        print(f"  Found {len(intersecting_indices)} intersecting streamlines")
        
        # Save slice NIfTI
        slice_data = new_data[:, slice_idx, :]
        # Create 3D volume with single slice
        slice_3d = np.expand_dims(slice_data, axis=1)
        
        # Create slice-specific affine matrix that properly represents this single plane
        # This ensures the NIfTI and TRK files use the same coordinate system
        slice_affine = img_resampled.affine.copy()
        
        # Update the Y translation to position this specific slice correctly
        # The slice represents Y=slice_idx in voxel space, which corresponds to world_y in mm
        slice_center_vox = np.array([(slice_3d.shape[0]-1)/2.0, 0.0, (slice_3d.shape[2]-1)/2.0, 1.0])
        expected_world_center = img_resampled.affine @ np.array([
            (img_resampled.shape[0]-1)/2.0, slice_idx, (img_resampled.shape[2]-1)/2.0, 1.0
        ])
        
        # Adjust translation so that voxel (center,0,center) maps to the correct world position
        slice_affine[:3, 3] = expected_world_center[:3] - slice_affine[:3, :3] @ slice_center_vox[:3]
        
        slice_nifti = nib.Nifti1Image(slice_3d, slice_affine)
        nifti_path_out = os.path.join(slice_dir, f"slice_{slice_idx:03d}.nii.gz")
        nib.save(slice_nifti, nifti_path_out)
        
        # Save individual TRK file (only if streamlines exist)
        if len(intersecting_indices) > 0:
            slice_streamlines_raw = [streamlines[idx] for idx in intersecting_indices]
            
            # Convert streamlines to the slice's voxel coordinate system for consistency
            slice_affine_inv = np.linalg.inv(slice_affine)
            
            slice_streamlines_voxel = []
            for streamline in slice_streamlines_raw:
                if streamlines_are_in_ras:
                    # Streamlines are already in RAS/world coordinates - convert to voxel
                    homogeneous = np.hstack([streamline, np.ones((len(streamline), 1))])
                    voxel_coords = homogeneous @ slice_affine_inv.T
                    slice_streamlines_voxel.append(voxel_coords[:, :3].astype(np.float32))
                else:
                    # Streamlines are in original voxel space - transform them properly to slice coordinates
                    # First convert from original volume voxel space to world space
                    homogeneous = np.hstack([streamline, np.ones((len(streamline), 1))])
                    world_coords = homogeneous @ original_trk_affine.T
                    
                    # Then convert from world space to slice voxel space
                    world_homogeneous = np.hstack([world_coords[:, :3], np.ones((len(world_coords), 1))])
                    slice_voxel_coords = world_homogeneous @ slice_affine_inv.T
                    
                    # For the Y coordinate, center it in the slice (Y dimension = 1)
                    # The slice represents a single plane, so all Y coordinates should be 0.0 in voxel space
                    # This ensures that when transformed by the slice affine, they map to the correct world Y
                    # For the Y coordinate, center it in the slice (Y dimension = 1)
                    # The slice represents a single plane, so all Y coordinates should be 0.0 in voxel space
                    slice_voxel_coords[:, 1] = 0.0
                    slice_streamlines_voxel.append(slice_voxel_coords[:, :3].astype(np.float32))
            
            # Convert voxel coordinates to world coordinates explicitly for proper TRK format
            # This ensures streamlines are in world coordinates with identity affine
            world_streamlines = []
            for streamline_vox in slice_streamlines_voxel:
                homogeneous = np.hstack([streamline_vox, np.ones((len(streamline_vox), 1))])
                world_coords = homogeneous @ slice_affine.T
                world_streamlines.append(world_coords[:, :3].astype(np.float32))
            
            # Create tractogram with identity affine since streamlines are already in world coordinates
            slice_tractogram = Tractogram(world_streamlines, affine_to_rasmm=np.eye(4))
            
            # Create header with identity affine since streamlines are in world coordinates
            slice_header = {
                'voxel_sizes': np.array(img_resampled.header.get_zooms()[:3], dtype=np.float32),
                'dimensions': np.array(slice_3d.shape[:3], dtype=np.int16),
                'voxel_to_rasmm': np.eye(4).astype(np.float32),  # Identity since streamlines are in world coords
                'voxel_order': 'RAS'
            }
            
            trk_path_out = os.path.join(slice_dir, f"slice_{slice_idx:03d}_streamlines.trk")
            save_trk(slice_tractogram, trk_path_out, header=slice_header)
    
    # Save metadata (convert all numpy types to Python native types for JSON serialization)
    metadata = {
        "extraction_mode": "stratified_sampling" if use_stratified_sampling else "all_slices",
        "n_slices_requested": int(n_slices) if use_stratified_sampling else "all",
        "target_dimensions": list(target_dimensions),
        "n_slices_extracted": int(len(selected_indices)),
        "voxel_size_mm": [float(x) for x in img_resampled.header.get_zooms()[:3]],
        "affine_matrix": [[float(x) for x in row] for row in img_resampled.affine.tolist()],
        "slab_half_thickness_mm": float(slab_half_thickness),
        "margin_slices": int(margin_slices) if use_stratified_sampling else "not_used",
        "selected_slice_indices": [int(x) for x in selected_indices.tolist()],
        "selected_slice_world_Y_mm": [float(x) for x in world_y_coords.tolist()],
        "streamlines_per_slice": [int(len(indices)) for indices in per_slice_trk_indices],
        "total_input_streamlines": int(len(streamlines)),
        "input_files": {
            "nifti": str(nifti_path),
            "trk": str(trk_path)
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Results Summary ===")
    print(f"Selected {len(selected_indices)} slices")
    
    # Handle case where some slices might have no streamlines
    streamline_counts = [len(idx) for idx in per_slice_trk_indices]
    if streamline_counts:
        print(f"Streamlines per slice: min={min(streamline_counts)}, "
              f"max={max(streamline_counts)}, "
              f"mean={np.mean(streamline_counts):.1f}")
        
        # Count slices with no streamlines
        empty_slices = sum(1 for count in streamline_counts if count == 0)
        if empty_slices > 0:
            print(f"Warning: {empty_slices}/{len(selected_indices)} slices have no intersecting streamlines")
            print("Consider increasing --slab_thickness if too many slices are empty")
    else:
        print("Warning: No streamline data processed")
        
    print(f"Outputs saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # File count summary
    print(f"Organization: Subfolder structure (slice_XXX/)")
    print(f"NIfTI slices: {len(selected_indices)} files saved")
    non_empty_trks = sum(1 for count in streamline_counts if count > 0)
    print(f"TRK files: {non_empty_trks} non-empty files saved")
    
    return {
        "selected_slice_indices": selected_indices,
        "selected_slice_world_Y_mm": world_y_coords,
        "per_slice_trk_indices": per_slice_trk_indices,
        "metadata": metadata,
        "output_dir": output_dir
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ALL coronal slices and streamline subsets")
    parser.add_argument("--nifti", type=str, required=True, help="Input NIfTI file")
    parser.add_argument("--trk", type=str, required=True, help="Input TRK file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--target_dim", type=int, nargs=3, required=True, help="Target dimensions (x, y, z)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Target voxel size for resampling")
    parser.add_argument("--slab_thickness", type=float, default=None, 
                        help="Full thickness of slab for streamline filtering (mm). If not provided, auto-calculated as voxel size.")
    
    args = parser.parse_args()
    
    # Handle slab thickness argument
    if args.slab_thickness is not None:
        slab_half_thickness = args.slab_thickness / 2.0
    else:
        slab_half_thickness = None  # Will be auto-calculated
    
    extract_coronal_slices_and_streamlines(
        nifti_path=args.nifti,
        trk_path=args.trk,
        output_dir=args.output,
        target_dimensions=tuple(args.target_dim),
        target_voxel_size=args.voxel_size,
        slab_half_thickness=slab_half_thickness
    )
