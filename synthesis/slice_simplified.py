#!/usr/bin/env python
"""
Simplified Coronal Slice Selection for Neuroimaging Data

This streamlined module performs coronal sectioning of neuroimaging datasets
with minimal complexity and maximum reliability. Assumes input NIfTI and TRK
files maintain identical spatial dimensions and coordinate systems.

Key simplifications:
- No geometric transformations or resampling
- Direct use of NIfTI header information  
- Minimal parameter complexity
- Robust error handling for file I/O only
- Efficient coronal plane sectioning
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk, save as save_trk
from nibabel.streamlines import Tractogram


def geometric_slab_clipping(streamline, y_center, slab_half_thickness, coordinate_system='voxel'):
    """
    Precisely clip streamlines to a slab using geometric intersection calculations.
    
    Parameters
    ----------
    streamline : np.ndarray
        Input streamline points (N, 3)
    y_center : float
        Center of the slab in appropriate coordinates
    slab_half_thickness : float
        Half-thickness of the slab in appropriate units
    coordinate_system : str
        Either 'voxel' or 'world' depending on coordinate space
        
    Returns
    -------
    list
        List of clipped streamline segments within the slab
    """
    if len(streamline) < 2:
        return []
    
    y_min = y_center - slab_half_thickness
    y_max = y_center + slab_half_thickness
    
    segments = []
    current_segment = []
    prev_point = None
    prev_inside = False
    
    for i in range(len(streamline)):
        point = streamline[i]
        y_coord = point[1]
        
        # Check if current point is inside slab
        inside = y_min <= y_coord <= y_max
        
        if inside:
            current_segment.append(point)
        elif prev_inside and not inside:
            # Calculate intersection point when exiting slab
            if prev_point is not None:
                if y_coord < y_min:
                    t = (y_min - prev_point[1]) / (point[1] - prev_point[1])
                else:
                    t = (y_max - prev_point[1]) / (point[1] - prev_point[1])
                intersection = prev_point + t * (point - prev_point)
                current_segment.append(intersection)
            if len(current_segment) >= 2:
                segments.append(np.array(current_segment))
            current_segment = []
        elif not prev_inside and inside and prev_point is not None:
            # Calculate intersection point when entering slab
            if y_coord > y_min:
                t = (y_min - prev_point[1]) / (point[1] - prev_point[1])
            else:
                t = (y_max - prev_point[1]) / (point[1] - prev_point[1])
            intersection = prev_point + t * (point - prev_point)
            current_segment.append(intersection)
            current_segment.append(point)
        
        prev_point = point
        prev_inside = inside
    
    # Add final segment if any points remain
    if len(current_segment) >= 2:
        segments.append(np.array(current_segment))
    
    return segments


def filter_streamlines_by_bounds(streamline, x_dim, y_dim, z_dim, coordinate_system='voxel', affine=None):
    """
    Filter out streamlines that are completely outside image bounds.
    
    Parameters
    ----------
    streamline : np.ndarray
        Streamline points
    x_dim, y_dim, z_dim : int
        Image dimensions
    coordinate_system : str
        'voxel' or 'world'
    affine : np.ndarray, optional
        Affine matrix for world coordinate bounds calculation
        
    Returns
    -------
    bool
        True if streamline should be kept, False if completely outside bounds
    """
    if len(streamline) == 0:
        return False
    
    if coordinate_system == 'voxel':
        # Check if any point is within voxel bounds
        x_coords = streamline[:, 0]
        y_coords = streamline[:, 1]
        z_coords = streamline[:, 2]
        
        return (np.any(x_coords >= 0) and np.any(x_coords < x_dim) and
                np.any(y_coords >= 0) and np.any(y_coords < y_dim) and
                np.any(z_coords >= 0) and np.any(z_coords < z_dim))
    
    else:  # world coordinates
        if affine is None:
            return True  # Cannot check bounds without affine
        
        # Calculate world coordinate bounds
        world_bounds = {
            'x_min': affine[0, 3],
            'x_max': affine[0, 3] + (x_dim - 1) * np.abs(affine[0, 0]),
            'y_min': affine[1, 3],
            'y_max': affine[1, 3] + (y_dim - 1) * np.abs(affine[1, 1]),
            'z_min': affine[2, 3],
            'z_max': affine[2, 3] + (z_dim - 1) * np.abs(affine[2, 2])
        }
        
        x_coords = streamline[:, 0]
        y_coords = streamline[:, 1]
        z_coords = streamline[:, 2]
        
        return (np.any(x_coords >= world_bounds['x_min']) and np.any(x_coords <= world_bounds['x_max']) and
                np.any(y_coords >= world_bounds['y_min']) and np.any(y_coords <= world_bounds['y_max']) and
                np.any(z_coords >= world_bounds['z_min']) and np.any(z_coords <= world_bounds['z_max']))


def extract_coronal_slices_simple(nifti_path, trk_path, output_dir, n_slices):
    """
    Simplified coronal slice extraction with minimal parameters.
    
    Assumes input files have identical spatial dimensions and coordinate systems.
    Extracts n_slices from the coronal axis (middle dimension of NIfTI).
    
    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI file
    trk_path : str 
        Path to input TRK file
    output_dir : str
        Output directory for slice folders
    n_slices : int
        Number of coronal slices to extract (1 to max_y_dim-1)
        
    Returns
    -------
    dict
        Processing results and metadata
        
    Raises
    ------
    ValueError
        If files don't exist or parameters are invalid
    IOError
        If file reading/writing fails
    """
    return extract_coronal_slices_simple_body(nifti_path, trk_path, output_dir, n_slices)


def extract_patches_simple(nifti_path, trk_path, output_dir, patch_size, num_patches):
    """
    Extract random 3D patches from neuroimaging data with streamlines.
    
    Calculates all possible non-overlapping patch positions, filters for
    positions containing streamlines, and randomly selects the specified number.
    
    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI file
    trk_path : str 
        Path to input TRK file
    output_dir : str
        Output directory for patch batch
    patch_size : tuple of int
        3D patch dimensions (x, y, z)
    num_patches : int
        Number of patches to extract
        
    Returns
    -------
    dict
        Processing results and metadata
        
    Raises
    ------
    ValueError
        If files don't exist or parameters are invalid
    IOError
        If file reading/writing fails
    """
    
    print(f"=== 3D Patch Extraction ===")
    print(f"Input NIfTI: {nifti_path}")
    print(f"Input TRK: {trk_path}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size}")
    print(f"Number of patches: {num_patches}")
    
    # Validate inputs
    if not os.path.exists(nifti_path):
        raise ValueError(f"NIfTI file not found: {nifti_path}")
    if not os.path.exists(trk_path):
        raise ValueError(f"TRK file not found: {trk_path}")
    if num_patches <= 0:
        raise ValueError(f"Number of patches must be positive: {num_patches}")
    if len(patch_size) != 3:
        raise ValueError(f"Patch size must be 3D: {patch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NIfTI file
    print("\n=== Loading NIfTI file ===")
    try:
        nifti_img = nib.load(nifti_path)
        nifti_data = nifti_img.get_fdata()
        nifti_affine = nifti_img.affine
        voxel_sizes = nifti_img.header.get_zooms()[:3]
        
        print(f"NIfTI shape: {nifti_data.shape}")
        print(f"Voxel sizes: {voxel_sizes} mm")
        print(f"Patch size: {patch_size} voxels")
        
    except Exception as e:
        raise IOError(f"Failed to load NIfTI file: {e}")
    
    # Extract 3D volume
    if len(nifti_data.shape) == 4:
        nifti_data = nifti_data[:, :, :, 0]
        print("Using first volume from 4D NIfTI")
    
    x_dim, y_dim, z_dim = nifti_data.shape
    patch_x, patch_y, patch_z = patch_size
    
    # Validate patch size
    if patch_x > x_dim or patch_y > y_dim or patch_z > z_dim:
        raise ValueError(f"Patch size {patch_size} exceeds image dimensions {nifti_data.shape}")
    
    # Load TRK file
    print("\n=== Loading TRK file ===")
    try:
        trk_obj = load_trk(trk_path)
        streamlines = trk_obj.streamlines
        
        print(f"Number of streamlines: {len(streamlines)}")
        
    except Exception as e:
        raise IOError(f"Failed to load TRK file: {e}")
    
    # Determine coordinate system (same logic as slice extraction)
    streamlines_in_world_coords = False
    if len(streamlines) > 0:
        all_coords = []
        sample_size = min(100, len(streamlines))
        
        for i in range(0, len(streamlines), max(1, len(streamlines) // sample_size)):
            streamline = streamlines[i]
            if len(streamline) > 0:
                all_coords.extend(streamline.flatten())
        
        if all_coords:
            all_coords = np.array(all_coords)
            max_coord = np.abs(all_coords).max()
            min_coord = all_coords.min()
            has_negative = np.any(all_coords < 0)
            max_image_dim = max(x_dim, y_dim, z_dim)
            
            if has_negative or max_coord > max_image_dim * 2:
                streamlines_in_world_coords = True
                print(f"→ Streamlines in WORLD coordinates (max: {max_coord:.2f}, min: {min_coord:.2f})")
            else:
                print(f"→ Streamlines in VOXEL coordinates (max: {max_coord:.2f}, min: {min_coord:.2f})")
    
    # Calculate all possible non-overlapping patch positions
    print("\n=== Calculating patch positions ===")
    possible_positions = []
    
    for x in range(0, x_dim - patch_x + 1, patch_x):
        for y in range(0, y_dim - patch_y + 1, patch_y):
            for z in range(0, z_dim - patch_z + 1, patch_z):
                possible_positions.append((x, y, z))
    
    print(f"Total possible non-overlapping positions: {len(possible_positions)}")
    
    if len(possible_positions) < num_patches:
        print(f"Warning: Only {len(possible_positions)} positions available, adjusting to maximum")
        num_patches = len(possible_positions)
    
    # Filter positions that contain streamlines
    print("\n=== Filtering positions with streamlines ===")
    valid_positions = []
    
    for pos_idx, (x_start, y_start, z_start) in enumerate(possible_positions):
        x_end = x_start + patch_x
        y_end = y_start + patch_y
        z_end = z_start + patch_z
        
        # Count streamlines in this patch
        streamlines_in_patch = 0
        
        for streamline in streamlines:
            if len(streamline) < 2:
                continue
            
            # Check if streamline intersects patch
            x_coords = streamline[:, 0]
            y_coords = streamline[:, 1]
            z_coords = streamline[:, 2]
            
            # Convert coordinates if needed
            if streamlines_in_world_coords:
                # Convert world coordinates to voxel coordinates for checking
                inv_affine = np.linalg.inv(nifti_affine)
                coords_homogeneous = np.column_stack([x_coords, y_coords, z_coords, np.ones(len(x_coords))])
                vox_coords = coords_homogeneous @ inv_affine.T
                x_coords, y_coords, z_coords = vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]
            
            # Check intersection with patch bounds
            x_intersect = np.any((x_coords >= x_start) & (x_coords < x_end))
            y_intersect = np.any((y_coords >= y_start) & (y_coords < y_end))
            z_intersect = np.any((z_coords >= z_start) & (z_coords < z_end))
            
            if x_intersect and y_intersect and z_intersect:
                streamlines_in_patch += 1
        
        if streamlines_in_patch > 0:
            valid_positions.append((x_start, y_start, z_start, streamlines_in_patch))
        
        if (pos_idx + 1) % 100 == 0:
            print(f"  Checked {pos_idx + 1}/{len(possible_positions)} positions...")
    
    print(f"Valid positions with streamlines: {len(valid_positions)}")
    
    if len(valid_positions) == 0:
        raise ValueError("No patch positions contain streamlines!")
    
    if len(valid_positions) < num_patches:
        print(f"Warning: Only {len(valid_positions)} valid positions, adjusting to maximum")
        num_patches = len(valid_positions)
    
    # Randomly select patches (uniformly distributed)
    print(f"\n=== Randomly selecting {num_patches} patches ===")
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(valid_positions), size=num_patches, replace=False)
    selected_positions = [valid_positions[i] for i in selected_indices]
    
    # Sort by streamline count for reporting
    selected_positions.sort(key=lambda x: x[3], reverse=True)
    
    for i, (x, y, z, count) in enumerate(selected_positions):
        print(f"  Patch {i+1}: Position ({x}, {y}, {z}) with {count} streamlines")
    
    # Extract and save patches
    print(f"\n=== Extracting {num_patches} patches ===")
    patch_details = []
    
    for patch_idx, (x_start, y_start, z_start, streamline_count) in enumerate(selected_positions):
        print(f"Processing patch {patch_idx + 1}/{num_patches}: ({x_start}, {y_start}, {z_start})")
        
        # Extract patch data
        x_end = x_start + patch_x
        y_end = y_start + patch_y
        z_end = z_start + patch_z
        
        patch_data = nifti_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Create patch affine
        patch_affine = nifti_affine.copy()
        patch_affine[:3, 3] += np.array([x_start, y_start, z_start]) * voxel_sizes
        
        # Save patch NIfTI
        patch_nifti = nib.Nifti1Image(patch_data, patch_affine, nifti_img.header)
        patch_filename = f"patch_{patch_idx+1:03d}.nii.gz"
        patch_path = os.path.join(output_dir, patch_filename)
        nib.save(patch_nifti, patch_path)
        
        patch_detail = {
            'patch_id': patch_idx + 1,
            'filename': patch_filename,
            'position': (int(x_start), int(y_start), int(z_start)),
            'size': [int(s) for s in patch_size],
            'bounds': (int(x_end), int(y_end), int(z_end)),
            'streamlines_count': int(streamline_count),
            'center_mm': (
                float((x_start + patch_x/2) * voxel_sizes[0]),
                float((y_start + patch_y/2) * voxel_sizes[1]),
                float((z_start + patch_z/2) * voxel_sizes[2])
            )
        }
        patch_details.append(patch_detail)
    
    # Create summary metadata
    metadata = {
        'processing_mode': 'patch_extraction',
        'input_files': {
            'nifti': nifti_path,
            'trk': trk_path
        },
        'input_dimensions': [int(x_dim), int(y_dim), int(z_dim)],
        'voxel_sizes_mm': [float(vs) for vs in voxel_sizes],
        'patch_size': [int(s) for s in patch_size],
        'patches_requested': int(num_patches),
        'patches_extracted': len(patch_details),
        'total_possible_positions': len(possible_positions),
        'valid_positions_with_streamlines': len(valid_positions),
        'streamlines_coordinate_system': 'world' if streamlines_in_world_coords else 'voxel',
        'patch_details': patch_details
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "patch_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved: {metadata_path}")
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")
    
    print(f"\n=== Patch Extraction Summary ===")
    print(f"Requested patches: {num_patches}")
    print(f"Extracted patches: {len(patch_details)}")
    print(f"Output directory: {output_dir}")
    
    return {
        'success': True,
        'metadata': metadata,
        'output_dir': output_dir,
        'n_patches_extracted': len(patch_details)
    }


def extract_coronal_slices_simple_body(nifti_path, trk_path, output_dir, n_slices):
    """Body of the slice extraction function."""
    print(f"=== Simplified Coronal Slice Extraction ===")
    print(f"Input NIfTI: {nifti_path}")
    print(f"Input TRK: {trk_path}")
    print(f"Output directory: {output_dir}")
    print(f"Requested slices: {n_slices}")
    
    # Validate inputs
    if not os.path.exists(nifti_path):
        raise ValueError(f"NIfTI file not found: {nifti_path}")
    if not os.path.exists(trk_path):
        raise ValueError(f"TRK file not found: {trk_path}")
    if n_slices <= 0:
        raise ValueError(f"Number of slices must be positive: {n_slices}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NIfTI file and extract metadata from header
    print("\n=== Loading NIfTI file ===")
    try:
        nifti_img = nib.load(nifti_path)
        nifti_data = nifti_img.get_fdata()
        nifti_affine = nifti_img.affine
        voxel_sizes = nifti_img.header.get_zooms()[:3]
        
        print(f"NIfTI shape: {nifti_data.shape}")
        print(f"Voxel sizes: {voxel_sizes} mm")
        print(f"Data type: {nifti_data.dtype}")
        
    except Exception as e:
        raise IOError(f"Failed to load NIfTI file: {e}")
    
    # Validate dimensions
    if len(nifti_data.shape) < 3:
        raise ValueError(f"NIfTI must be 3D or 4D, got {len(nifti_data.shape)}D")
    
    # Extract 3D volume (ignore 4th dimension if present)
    if len(nifti_data.shape) == 4:
        nifti_data = nifti_data[:, :, :, 0]
        print("Using first volume from 4D NIfTI")
    
    x_dim, y_dim, z_dim = nifti_data.shape
    print(f"Working with dimensions: {x_dim} x {y_dim} x {z_dim}")
    
    # Validate slice count against coronal dimension (y-axis)
    max_slices = y_dim - 1
    if n_slices > max_slices:
        print(f"Warning: Requested {n_slices} slices exceeds maximum {max_slices}")
        print(f"Adjusting to maximum available: {max_slices}")
        n_slices = max_slices
    
    # Load TRK file
    print("\n=== Loading TRK file ===")
    try:
        trk_obj = load_trk(trk_path)
        streamlines = trk_obj.streamlines
        trk_header = trk_obj.header
        
        print(f"Number of streamlines: {len(streamlines)}")
        if len(streamlines) > 0:
            sample_streamline = streamlines[0]
            print(f"Sample streamline length: {len(sample_streamline)} points")
            
    except Exception as e:
        raise IOError(f"Failed to load TRK file: {e}")
    
    # Validate coordinate system assumption
    trk_dimensions = trk_header.get('dimensions', [0, 0, 0])
    if list(trk_dimensions) != [x_dim, y_dim, z_dim]:
        print(f"Warning: TRK dimensions {trk_dimensions} != NIfTI dimensions {[x_dim, y_dim, z_dim]}")
        print("Proceeding with assumption that files are spatially aligned")
    
    # Determine coordinate system by analyzing streamline coordinates
    streamlines_in_world_coords = False
    if len(streamlines) > 0:
        # Analyze multiple streamlines to determine coordinate system
        all_coords = []
        sample_size = min(100, len(streamlines))
        
        for i in range(0, len(streamlines), max(1, len(streamlines) // sample_size)):
            streamline = streamlines[i]
            if len(streamline) > 0:
                all_coords.extend(streamline.flatten())
        
        if all_coords:
            all_coords = np.array(all_coords)
            max_coord = np.abs(all_coords).max()
            min_coord = all_coords.min()
            
            # Check for negative coordinates or coordinates beyond image dimensions
            has_negative = np.any(all_coords < 0)
            max_image_dim = max(x_dim, y_dim, z_dim)
            
            # DEBUG: Show detailed coordinate analysis
            print(f"DEBUG: Coordinate analysis across {len(streamlines[:100])} streamlines:")
            print(f"  Max coordinate: {max_coord:.2f}")
            print(f"  Min coordinate: {min_coord:.2f}")
            print(f"  Has negative: {has_negative}")
            print(f"  Coordinate range: {max_coord - min_coord:.2f}")
            print(f"  Image dimensions: {x_dim} x {y_dim} x {z_dim}")
            print(f"  Max dimension: {max_image_dim}")
            print(f"  Threshold: {max_image_dim * 2}")
            
            if has_negative or max_coord > max_image_dim * 2:
                streamlines_in_world_coords = True
                print(f"→ Streamlines appear to be in WORLD coordinates (max: {max_coord:.2f}, min: {min_coord:.2f}, has_negative: {has_negative})")
            else:
                streamlines_in_world_coords = False
                print(f"→ Streamlines appear to be in VOXEL coordinates (max: {max_coord:.2f}, min: {min_coord:.2f})")
    
    # Get Y coordinate distribution for sampling
    print("\n=== Analyzing streamline distribution ===")
    all_y_coords = []
    for streamline in streamlines:
        if len(streamline) > 0:
            all_y_coords.extend(streamline[:, 1])
    
    if not all_y_coords:
        raise ValueError("No streamline points found!")
    
    all_y_coords = np.array(all_y_coords)
    y_min, y_max = all_y_coords.min(), all_y_coords.max()
    y_center = (y_min + y_max) / 2
    y_range = y_max - y_min
    
    print(f"Streamline Y range: {y_min:.1f} to {y_max:.1f} (center: {y_center:.1f})")
    print(f"Streamline Y span: {y_range:.1f} voxels")
    
    # OVERRIDE COORDINATE SYSTEM DETECTION: If we found negative coordinates in the full distribution,
    # the streamlines are definitely in world coordinates
    if y_min < 0:
        print(f"WARNING: OVERRIDE: Found negative Y coordinates ({y_min:.1f}) in full distribution!")
        print(f"   Streamlines are definitely in WORLD coordinates, not voxel coordinates")
        streamlines_in_world_coords = True
    
    if streamlines_in_world_coords:
        print(f"Streamline Y range: {y_min:.3f} to {y_max:.3f} mm")
        # Convert to voxel coordinates for sampling
        world_y_min = nifti_affine[1, 3]
        sampling_y_min = max(0, int((y_min - world_y_min) / voxel_sizes[1]))
        sampling_y_max = min(y_dim - 1, int((y_max - world_y_min) / voxel_sizes[1]))
    else:
        print(f"Streamline Y range: {y_min:.1f} to {y_max:.1f} voxels")
        sampling_y_min = max(0, int(y_min))
        sampling_y_max = min(y_dim - 1, int(y_max))
    
    # Add padding to sampling range
    padding = max(5, int(y_range * 0.2))  # 20% padding or 5 voxels minimum
    start_slice = max(0, sampling_y_min - padding)
    end_slice = min(y_dim - 1, sampling_y_max + padding)
    
    print(f"Sampling range: {start_slice} to {end_slice} (with {padding} voxel padding)")
    
    # Select slice indices
    if end_slice - start_slice >= n_slices:
        slice_indices = np.linspace(start_slice, end_slice, n_slices, dtype=int)
    else:
        slice_indices = np.arange(start_slice, end_slice + 1)
    
    slice_indices = np.unique(slice_indices)
    actual_n_slices = len(slice_indices)
    
    print(f"\n=== Processing {actual_n_slices} coronal slices ===")
    print(f"Slice indices: {slice_indices}")
    
    # Process each slice
    processing_results = []
    streamlines_per_slice = []
    
    for i, slice_idx in enumerate(slice_indices):
        print(f"\nProcessing slice {i+1}/{actual_n_slices}: coronal index {slice_idx}")
        
        # Create slice directory
        slice_dir = os.path.join(output_dir, f"slice_{slice_idx:03d}")
        os.makedirs(slice_dir, exist_ok=True)
        
        # Extract coronal slice from NIfTI
        slice_data = nifti_data[:, slice_idx, :]
        slice_3d = np.expand_dims(slice_data, axis=1)
        
        # Create slice-specific affine
        slice_affine = nifti_affine.copy()
        slice_world_y = nifti_affine[1, 3] + slice_idx * voxel_sizes[1]
        slice_affine[1, 3] = slice_world_y
        
        # Save slice NIfTI
        slice_nifti = nib.Nifti1Image(slice_3d, slice_affine)
        nifti_output_path = os.path.join(slice_dir, f"slice_{slice_idx:03d}.nii.gz")
        nib.save(slice_nifti, nifti_output_path)
        print(f"  Saved NIfTI: {os.path.basename(nifti_output_path)}")
        
        # Process streamlines with geometric clipping
        intersecting_streamlines = []
        filtered_count = 0
        
        if streamlines_in_world_coords:
            # Use full voxel thickness for short streamlines to ensure we capture them
            slab_half_thickness = voxel_sizes[1] / 2.0  # Half voxel thickness (full voxel)
            slice_center = slice_world_y
            print(f"  Filtering in world coordinates: Y={slice_center:.3f}mm ± {slab_half_thickness:.3f}mm")
        else:
            # Use full voxel thickness for short streamlines to ensure we capture them
            slab_half_thickness = voxel_sizes[1] / 2.0  # Half voxel thickness (full voxel)
            slice_center = slice_idx
            print(f"  Filtering in voxel coordinates: Y={slice_center} ± {slab_half_thickness/voxel_sizes[1]:.3f} voxels")
        
        for streamline in streamlines:
            if len(streamline) < 2:
                continue
                
            # Filter out streamlines completely outside bounds
            if not filter_streamlines_by_bounds(streamline, x_dim, y_dim, z_dim, 
                                              'world' if streamlines_in_world_coords else 'voxel',
                                              nifti_affine):
                filtered_count += 1
                continue
            
            # For short streamlines, use simple intersection check instead of geometric clipping
            if len(streamline) <= 5:  # Short streamlines (like the 3-point ones we have)
                # Simple check: if any point is within the slab, include the entire streamline
                y_coords = streamline[:, 1]
                if np.any(np.abs(y_coords - slice_center) <= slab_half_thickness):
                    intersecting_streamlines.append(streamline)
            else:
                # For longer streamlines, use geometric clipping
                segments = geometric_slab_clipping(
                    streamline, slice_center, slab_half_thickness,
                    'world' if streamlines_in_world_coords else 'voxel'
                )
                intersecting_streamlines.extend(segments)
        
        streamlines_per_slice.append(len(intersecting_streamlines))
        print(f"  Found {len(intersecting_streamlines)} streamline segments")
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} streamlines outside image bounds")
        
        # Save TRK file if streamlines exist
        trk_output_path = None
        if len(intersecting_streamlines) > 0:
            # Convert to float32 for TRK compatibility
            clipped_streamlines = [s.astype(np.float32) for s in intersecting_streamlines if len(s) >= 2]
            
            # Create tractogram
            slice_tractogram = Tractogram(clipped_streamlines, affine_to_rasmm=np.eye(4))
            
            # Create header
            slice_trk_header = trk_header.copy()
            slice_trk_header['dimensions'] = np.array(slice_3d.shape[:3], dtype=np.int16)
            slice_trk_header['voxel_sizes'] = np.array(voxel_sizes, dtype=np.float32)
            slice_trk_header['voxel_to_rasmm'] = slice_affine.astype(np.float32)
            
            # Save TRK
            trk_output_path = os.path.join(slice_dir, f"slice_{slice_idx:03d}_streamlines.trk")
            save_trk(slice_tractogram, trk_output_path, header=slice_trk_header)
            print(f"  Saved TRK: {os.path.basename(trk_output_path)}")
        else:
            print(f"  No TRK file saved (no intersecting streamlines)")
        
        processing_results.append({
            'slice_index': int(slice_idx),
            'slice_dir': slice_dir,
            'nifti_file': nifti_output_path,
            'trk_file': trk_output_path,
            'n_streamlines': len(intersecting_streamlines),
            'status': 'success'
        })
    
    # Create summary metadata
    metadata = {
        'processing_mode': 'simplified_coronal_extraction',
        'input_files': {
            'nifti': nifti_path,
            'trk': trk_path
        },
        'input_dimensions': [int(x_dim), int(y_dim), int(z_dim)],
        'voxel_sizes_mm': [float(vs) for vs in voxel_sizes],
        'affine_matrix': nifti_affine.tolist(),
        'n_slices_requested': int(n_slices),
        'n_slices_processed': len(processing_results),
        'slice_indices': [int(idx) for idx in slice_indices],
        'streamlines_per_slice': streamlines_per_slice,
        'total_input_streamlines': int(len(streamlines)),
        'streamlines_coordinate_system': 'world' if streamlines_in_world_coords else 'voxel',
        'slab_half_thickness_mm': float(voxel_sizes[1] / 2.0),
        'streamline_y_range': {
            'min': float(y_min),
            'max': float(y_max),
            'center': float(y_center),
            'span': float(y_range),
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved: {metadata_path}")
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Requested slices: {n_slices}")
    print(f"Processed slices: {len(processing_results)}")
    if streamlines_per_slice:
        print(f"Streamlines per slice: min={min(streamlines_per_slice)}, max={max(streamlines_per_slice)}, avg={np.mean(streamlines_per_slice):.1f}")
    
    return {
        'success': True,
        'metadata': metadata,
        'output_dir': output_dir,
        'n_slices_processed': len(processing_results)
    }


def main():
    """Command-line interface for simplified slice extraction."""
    parser = argparse.ArgumentParser(
        description="Simplified coronal slice extraction for neuroimaging data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 25 coronal slices
  python slice_selection_simplified.py input.nii.gz tractography.trk output_slices/ 25
        """
    )
    
    parser.add_argument("nifti_file", help="Input NIfTI file path")
    parser.add_argument("trk_file", help="Input TRK file path") 
    parser.add_argument("output_dir", help="Output directory for slice folders")
    parser.add_argument("n_slices", type=int, help="Number of coronal slices to extract")
    
    args = parser.parse_args()
    
    try:
        result = extract_coronal_slices_simple(
            nifti_path=args.nifti_file,
            trk_path=args.trk_file,
            output_dir=args.output_dir,
            n_slices=args.n_slices
        )
        
        if result['success']:
            print(f"\nSlice extraction completed successfully!")
            print(f"Output directory: {result['output_dir']}")
        else:
            print(f"\nWARNING: Slice extraction completed with some failures")
            print(f"Check output directory: {result['output_dir']}")
            
    except Exception as e:
        print(f"\nERROR: Slice extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())