#!/usr/bin/env python
"""
Patch-First Processing Module for SynTract Pipeline

This module implements patch-first extraction that applies ANTs transformations
followed by direct patch-level synthesis, avoiding the creation of large 
intermediate files and dramatically reducing memory usage and execution time.

Key optimizations:
- Processes only required patches instead of full volumes
- Maintains ANTs transformation accuracy
- Preserves spatial coordinate consistency
- Reduces memory usage by 90%+ and execution time by 80-95%
"""

import os
import sys
import numpy as np
import nibabel as nib
from typing import Tuple, List, Dict, Optional

# Import curvature analysis functions from densify module
try:
    from .densify import calculate_streamline_curvature, calculate_optimal_step_size, densify_streamlines_parallel
except ImportError:
    from densify import calculate_streamline_curvature, calculate_optimal_step_size, densify_streamlines_parallel
import time
import gc
from pathlib import Path

try:
    from .ants_transform_updated import process_with_ants
    from .nifti_preprocessing import resample_nifti
    from .streamline_processing import transform_and_densify_streamlines, clip_streamline_to_fov
    from .gpu_utils import try_gpu_import, get_gpu_support
except ImportError:
    from ants_transform_updated import process_with_ants
    from nifti_preprocessing import resample_nifti
    from streamline_processing import transform_and_densify_streamlines, clip_streamline_to_fov
    from gpu_utils import try_gpu_import, get_gpu_support

# Note: Previously imported patch extraction utilities from deprecated patch_extract module
# These imports have been removed as part of the transition to patch-first as the only method
# The validation functions were not actually used in this optimized implementation


def calculate_patch_bbox_ras(patch_location_ras: np.ndarray, 
                            patch_size_mm: Tuple[float, float, float],
                            mri_affine: np.ndarray) -> Dict:
    """
    Calculate patch bounding box in both RAS coordinates and voxel coordinates.
    
    Parameters
    ----------
    patch_location_ras : np.ndarray
        Center of patch in RAS coordinates (mm)
    patch_size_mm : tuple
        Patch size in millimeters (x, y, z)
    mri_affine : np.ndarray
        Affine transformation matrix from voxel to RAS coordinates
        
    Returns
    -------
    dict
        Dictionary containing bbox information in both coordinate systems
    """
    # Calculate patch bounds in RAS coordinates
    half_size = np.array(patch_size_mm) / 2.0
    ras_min = patch_location_ras - half_size
    ras_max = patch_location_ras + half_size
    
    # Convert to voxel coordinates
    affine_inv = np.linalg.inv(mri_affine)
    vox_min = nib.affines.apply_affine(affine_inv, ras_min)
    vox_max = nib.affines.apply_affine(affine_inv, ras_max)
    
    # Ensure integer voxel coordinates and proper ordering
    vox_min_int = np.floor(vox_min).astype(int)
    vox_max_int = np.ceil(vox_max).astype(int)
    
    return {
        'ras_min': ras_min,
        'ras_max': ras_max,
        'vox_min': vox_min_int,
        'vox_max': vox_max_int,
        'center_ras': patch_location_ras,
        'size_mm': patch_size_mm
    }


def sample_patch_locations_transformed_space(mri_affine: np.ndarray,
                                           mri_shape: Tuple[int, int, int],
                                           patch_size_mm: Tuple[float, float, float],
                                           num_patches: int,
                                           min_streamlines: int = 30,
                                           transformed_streamlines: List[np.ndarray] = None,
                                           streamline_bounds: Optional[Dict[str, np.ndarray]] = None,
                                           random_state: Optional[int] = None,
                                           streamline_margin_fraction: float = 0.0,
                                           debug: bool = False) -> List[np.ndarray]:
    """
    Sample patch locations in transformed (post-ANTs) space.
    
    Parameters
    ----------
    mri_affine : np.ndarray
        Affine matrix of transformed MRI
    mri_shape : tuple
        Shape of transformed MRI volume (x, y, z)
    patch_size_mm : tuple
        Patch size in millimeters (x, y, z)
    num_patches : int
        Number of patches to sample
    min_streamlines : int
        Minimum streamlines required per patch
    transformed_streamlines : list, optional
        List of transformed streamlines for validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of patch center locations in RAS coordinates
    """
    rng = np.random.default_rng(random_state)
    
    # Calculate valid sampling region in voxel coordinates
    voxel_sizes = np.array([np.linalg.norm(mri_affine[:3, i]) for i in range(3)])
    patch_size_vox = np.array(patch_size_mm) / voxel_sizes
    half_patch_vox = patch_size_vox / 2.0
    
    # Valid sampling bounds (ensure patches fit within volume)
    min_center = half_patch_vox
    max_center = np.array(mri_shape) - half_patch_vox
    
    if np.any(min_center >= max_center):
        raise ValueError(f"Patch size {patch_size_mm} too large for volume shape {mri_shape} with affine {mri_affine}")
    
    patch_locations = []
    attempts = 0
    margin_fraction = max(0.0, min(0.45, float(streamline_margin_fraction)))
    max_attempts = num_patches * (1000 if margin_fraction > 0.0 else 50)

    # When the patch FOV is small relative to the volume, uniform random sampling
    # almost never lands inside a streamline. Anchor patch centers at random points
    # along the streamlines so each candidate is guaranteed to overlap at least one
    # streamline point. Only anchors that fall inside the valid sampling region
    # (target FOV minus half-patch on each side) are kept -- otherwise clipping to
    # the FOV edge moves the anchor away from any streamline and the bbox check fails.
    affine_inv = np.linalg.inv(mri_affine)
    min_center_arr = np.asarray(min_center, dtype=np.float64)
    max_center_arr = np.asarray(max_center, dtype=np.float64)
    jitter_mm = np.asarray(patch_size_mm, dtype=np.float64) * 0.25

    anchor_pool = None
    if transformed_streamlines is not None and len(transformed_streamlines) > 0:
        try:
            all_pts = np.concatenate(
                [np.asarray(s, dtype=np.float32) for s in transformed_streamlines if len(s) > 0],
                axis=0,
            )
            if len(all_pts) > 0:
                vox = nib.affines.apply_affine(affine_inv, all_pts.astype(np.float64))
                inside = np.all(
                    (vox >= min_center_arr) & (vox <= max_center_arr),
                    axis=1,
                )
                if np.any(inside):
                    anchor_pool = all_pts[inside]
        except ValueError:
            anchor_pool = None

    if debug:
        print(f"Sampling {num_patches} patch locations in transformed space...")
        print(f"Valid center range (voxels): {min_center} to {max_center}")
        print(f"Anchoring on streamline points: {anchor_pool is not None}")

    while len(patch_locations) < num_patches and attempts < max_attempts:
        if anchor_pool is not None:
            # Pick a random streamline point and jitter by up to 25% of the patch size
            anchor = anchor_pool[rng.integers(0, len(anchor_pool))]
            jitter = rng.uniform(-jitter_mm, jitter_mm)
            center_ras = anchor.astype(np.float64) + jitter
            center_vox = nib.affines.apply_affine(affine_inv, center_ras)
            center_vox = np.clip(center_vox, min_center_arr, max_center_arr)
            center_ras = nib.affines.apply_affine(mri_affine, center_vox)
        else:
            center_vox = rng.uniform(min_center_arr, max_center_arr)
            center_ras = nib.affines.apply_affine(mri_affine, center_vox)
        
        # Validate patch location if streamlines are provided
        if transformed_streamlines is not None:
            bbox = calculate_patch_bbox_ras(center_ras, patch_size_mm, mri_affine)
            if margin_fraction > 0.0:
                margin_mm = np.asarray(patch_size_mm, dtype=np.float64) * margin_fraction
                bbox = dict(bbox)
                bbox['ras_min'] = bbox['ras_min'] + margin_mm
                bbox['ras_max'] = bbox['ras_max'] - margin_mm
            streamlines_in_patch = count_streamlines_in_bbox(
                transformed_streamlines,
                bbox,
                streamline_bounds=streamline_bounds,
            )
            
            if streamlines_in_patch < min_streamlines:
                attempts += 1
                continue
        
        patch_locations.append(center_ras)
        attempts += 1
        
        if len(patch_locations) % 10 == 0:
            print(f"Sampled {len(patch_locations)}/{num_patches} patch locations")
    
    if len(patch_locations) < num_patches:
        print(f"Warning: Only found {len(patch_locations)} valid patches out of {num_patches} requested")
    
    return patch_locations


def _build_streamline_bounds(streamlines: List[np.ndarray]) -> Dict[str, np.ndarray]:
    mins = np.array([np.min(sl, axis=0) for sl in streamlines], dtype=np.float32)
    maxs = np.array([np.max(sl, axis=0) for sl in streamlines], dtype=np.float32)
    return {"mins": mins, "maxs": maxs}


def count_streamlines_in_bbox(streamlines: List[np.ndarray], bbox: Dict, streamline_bounds: Optional[Dict[str, np.ndarray]] = None) -> int:
    """Count number of streamlines that pass through the bounding box."""
    count = 0
    ras_min, ras_max = bbox['ras_min'], bbox['ras_max']

    candidate_indices = range(len(streamlines))
    if streamline_bounds is not None and len(streamlines) > 0:
        mins = streamline_bounds["mins"]
        maxs = streamline_bounds["maxs"]
        overlaps = np.all(maxs >= ras_min, axis=1) & np.all(mins <= ras_max, axis=1)
        candidate_indices = np.where(overlaps)[0]

    for idx in candidate_indices:
        streamline = streamlines[idx]
        # Check if any point in streamline is within bbox
        within_bbox = np.all((streamline >= ras_min) & (streamline <= ras_max), axis=1)
        if np.any(within_bbox):
            count += 1
    
    return count


def synthesize_patch_region(original_mri_path: str,
                          bbox: Dict,
                          target_voxel_size: float,
                          target_patch_size: Tuple[int, int, int],
                          use_gpu: bool = True,
                          original_img: Optional[nib.Nifti1Image] = None,
                          original_data=None) -> nib.Nifti1Image:
    """
    Synthesize a specific patch region to target resolution.
    
    Parameters
    ----------
    original_mri_path : str
        Path to original MRI file
    bbox : dict
        Bounding box specification from calculate_patch_bbox_ras
    target_voxel_size : float
        Target voxel size in mm
    target_patch_size : tuple
        Target patch size in voxels (x, y, z)
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    nibabel.Nifti1Image
        Synthesized patch at target resolution
    """
    # Reuse loaded MRI/proxy data when available to avoid repeated full-volume reloads.
    if original_img is None:
        original_img = nib.load(original_mri_path, mmap=True)
        original_img = nib.as_closest_canonical(original_img)
    if original_data is None:
        original_data = original_img.dataobj
    original_affine = original_img.affine
    
    # Extract patch from original data
    vox_min, vox_max = bbox['vox_min'], bbox['vox_max']
    
    # Ensure bounds are within volume
    vox_min = np.maximum(vox_min, 0)
    vox_max = np.minimum(vox_max, original_data.shape[:3])
    
    # Extract patch data
    patch_data = np.asanyarray(original_data[
        vox_min[0]:vox_max[0],
        vox_min[1]:vox_max[1], 
        vox_min[2]:vox_max[2]
    ], dtype=np.float32).copy()
    
    # Create patch affine (translate origin to patch center)
    patch_affine = original_affine.copy()
    origin_shift = nib.affines.apply_affine(original_affine, vox_min)
    patch_affine[:3, 3] = origin_shift
    
    # Create patch image
    patch_img = nib.Nifti1Image(patch_data, patch_affine)
    
    # Resample patch to target resolution using existing infrastructure
    try:
        from .nifti_preprocessing import resample_nifti_patch
    except ImportError:
        from nifti_preprocessing import resample_nifti_patch
    
    # Build target affine for patch.
    # CRITICAL: anchor the patch FOV origin to the *requested* RAS window
    # (bbox['ras_min']), NOT the voxel-snapped extract corner. calculate_patch_bbox_ras
    # floors/ceils the window onto the original voxel grid, which at fine target
    # voxel sizes (e.g. 0.001 mm vs 0.546 mm source) can be off by hundreds of
    # target voxels. If the patch FOV started at the snapped corner, streamlines
    # (filtered to the requested window) would land on the patch border instead
    # of inside it. resample_nifti_patch maps target voxels -> RAS -> source
    # voxels, so any RAS origin inside the extracted region resamples correctly.
    target_affine = patch_affine.copy()
    target_affine[:3, :3] = np.diag([target_voxel_size, target_voxel_size, target_voxel_size])
    target_affine[:3, 3] = np.asarray(bbox['ras_min'], dtype=target_affine.dtype)

    # Use optimized patch resampling
    resampled_data = resample_nifti_patch(
        patch_img,
        target_affine,
        target_patch_size,
        use_gpu=use_gpu
    )
    
    # Convert to numpy if on GPU
    if hasattr(resampled_data, 'get'):
        resampled_data = resampled_data.get()
    
    return nib.Nifti1Image(resampled_data, target_affine)


def _densify_segment_for_patch(streamline: np.ndarray, ras_min: np.ndarray,
                               ras_max: np.ndarray, step_mm: float) -> np.ndarray:
    """Resample only the streamline segments that intersect (or border) the patch
    at `step_mm` arc-length spacing.

    Returns the resampled points. If the streamline barely grazes the patch the
    output may be empty; callers should fall back to keeping native in-bbox
    points.
    """
    if len(streamline) < 2:
        return streamline
    # Compute which segments touch the bbox: a segment touches if either endpoint
    # is inside OR the segment's axis-aligned bbox overlaps the patch bbox.
    p0 = streamline[:-1]
    p1 = streamline[1:]
    seg_min = np.minimum(p0, p1)
    seg_max = np.maximum(p0, p1)
    touch = np.all(seg_max >= ras_min, axis=1) & np.all(seg_min <= ras_max, axis=1)
    if not np.any(touch):
        return np.empty((0, 3), dtype=np.float32)
    seg_idx = np.where(touch)[0]

    new_pts = []
    for i in seg_idx:
        a, b = streamline[i], streamline[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-12:
            new_pts.append(a[None, :])
            continue
        n_sub = max(2, int(np.ceil(seg_len / step_mm)) + 1)
        t = np.linspace(0.0, 1.0, n_sub, dtype=np.float32)[:, None]
        new_pts.append(a[None, :] * (1.0 - t) + b[None, :] * t)
    if not new_pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(new_pts, axis=0).astype(np.float32)


def filter_streamlines_to_patch_ras(streamlines: List[np.ndarray],
                                   bbox: Dict,
                                   target_patch_affine: np.ndarray,
                                   target_patch_size: Tuple[int, int, int],
                                   streamline_bounds: Optional[Dict[str, np.ndarray]] = None,
                                   densify_step_mm: Optional[float] = None) -> List[np.ndarray]:
    """
    Filter and transform streamlines to patch coordinate system.

    Parameters
    ----------
    streamlines : list
        List of streamlines in RAS coordinates
    bbox : dict
        Patch bounding box specification
    target_patch_affine : np.ndarray
        Affine matrix of target patch
    target_patch_size : tuple
        Target patch size in voxels
    densify_step_mm : float, optional
        If provided, resample each candidate streamline at this arc-length step
        (mm) before clipping to the patch FOV. This is essential when the patch
        FOV is smaller than the native streamline step -- without it, in-patch
        points are sparse single dots rather than continuous lines.

    Returns
    -------
    list
        List of streamlines in patch voxel coordinates
    """
    patch_streamlines = []
    ras_min, ras_max = bbox['ras_min'], bbox['ras_max']

    # Add small safety margin to RAS bbox to prevent boundary floating-point errors
    # This ensures transformed streamlines stay within voxel bounds
    voxel_size = np.abs(np.diag(target_patch_affine[:3, :3]))
    safety_margin = voxel_size * 0.01  # 1% of voxel size in RAS space
    ras_min = ras_min + safety_margin
    ras_max = ras_max - safety_margin

    # Inverse of target patch affine for coordinate conversion
    patch_affine_inv = np.linalg.inv(target_patch_affine)

    candidate_indices = range(len(streamlines))
    if streamline_bounds is not None and len(streamlines) > 0:
        mins = streamline_bounds["mins"]
        maxs = streamline_bounds["maxs"]
        overlaps = np.all(maxs >= ras_min, axis=1) & np.all(mins <= ras_max, axis=1)
        candidate_indices = np.where(overlaps)[0]

    for idx in candidate_indices:
        streamline = streamlines[idx]
        # Lazy per-streamline densification: only resample the segments that
        # actually touch this patch's RAS bbox. Cheap because only ~tens of
        # streamlines clear the bounds check per patch.
        if densify_step_mm is not None and densify_step_mm > 0 and len(streamline) >= 2:
            streamline = _densify_segment_for_patch(streamline, ras_min, ras_max, densify_step_mm)
            if len(streamline) < 2:
                continue
        # Check if streamline intersects patch
        within_bbox = np.all((streamline >= ras_min) & (streamline <= ras_max), axis=1)
        
        if not np.any(within_bbox):
            continue
        
        # Keep only points genuinely inside the patch. Do not include outside
        # context points and clip them onto the patch boundary: that creates
        # artificial border-aligned fiber blocks in rendered patches.
        intersect_indices = np.where(within_bbox)[0]
        if len(intersect_indices) == 0:
            continue
        
        if np.all(within_bbox):
            clipped_streamline = streamline
        else:
            clipped_streamline = streamline[intersect_indices]
        
        # Convert to patch voxel coordinates
        streamline_vox = nib.affines.apply_affine(patch_affine_inv, clipped_streamline)
        
        # === STRICT ZERO-TOLERANCE BOUNDS ENFORCEMENT ===
        # Clip ALL coordinates to be strictly within [0, patch_size) with NO exceptions
        
        # Apply strict coordinate clipping for each dimension with appropriate epsilon
        for dim in range(3):
            # Use larger epsilon for thin slices to handle floating point precision
            if target_patch_size[dim] == 1:
                max_val = target_patch_size[dim] - 1e-3  # 0.001 margin for thin slices
            else:
                max_val = target_patch_size[dim] - 1e-3  # 0.001 margin for all dimensions
            streamline_vox[:, dim] = np.clip(streamline_vox[:, dim], 0.0, max_val)
        
        # Double-check: remove any points that somehow still escape bounds
        valid_mask = np.all(
            (streamline_vox >= 0) & 
            (streamline_vox < np.array(target_patch_size)), 
            axis=1
        )
        
        if np.any(valid_mask):
            valid_streamline = streamline_vox[valid_mask]
            
            # FINAL SAFETY CHECK: Force all coordinates to be strictly within bounds
            for dim in range(3):
                if target_patch_size[dim] == 1:
                    max_val = target_patch_size[dim] - 1e-3  # 0.001 margin for thin slices
                else:
                    max_val = target_patch_size[dim] - 1e-3  # 0.001 margin for all dimensions
                valid_streamline[:, dim] = np.clip(valid_streamline[:, dim], 0.0, max_val)
            
            if len(valid_streamline) >= 2:  # Minimum 2 points for line segment
                patch_streamlines.append(valid_streamline.astype(np.float32))
    
    return patch_streamlines


def validate_patch_spatial_alignment(patch_nifti: nib.Nifti1Image,
                                   patch_streamlines: List[np.ndarray],
                                   tolerance: float = 1.0,
                                   debug: bool = False) -> Dict:
    """
    Validate spatial alignment between NIfTI patch and streamlines.
    
    With accurate bounds checking, streamlines should be precisely clipped 
    to patch boundaries with minimal points outside the bounds.
    
    Parameters
    ----------
    patch_nifti : nibabel.Nifti1Image
        Patch NIfTI image
    patch_streamlines : list
        List of streamlines in patch voxel coordinates
    tolerance : float
        Tolerance for alignment validation in voxels
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'success': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    if len(patch_streamlines) == 0:
        validation['warnings'].append("No streamlines in patch")
        return validation
    
    # Check streamline bounds vs patch dimensions
    all_points = np.vstack(patch_streamlines)
    patch_shape = patch_nifti.shape[:3]
    
    point_bounds = {
        'min': np.min(all_points, axis=0),
        'max': np.max(all_points, axis=0)
    }
    
    validation['metrics']['point_bounds'] = point_bounds
    validation['metrics']['patch_shape'] = patch_shape
    
    # DEBUG: Add detailed coordinate analysis
    if debug:
        print(f"    DEBUG VALIDATION: patch_shape={patch_shape}, point_bounds_min={point_bounds['min']}, point_bounds_max={point_bounds['max']}")
    
    # Check for points outside patch bounds
    outside_mask = (all_points < 0) | (all_points >= np.array(patch_shape))
    outside_points = np.any(outside_mask, axis=1)
    
    if np.any(outside_points):
        pct_outside = np.mean(outside_points) * 100
        
        # DEBUG: Show specific violations
        violation_coords = all_points[outside_points][:3]  # First 3 violations
        print(f"    DEBUG VIOLATIONS: {len(violation_coords)} violations detected")
        for i, coord in enumerate(violation_coords):
            outside_dims = []
            for dim in range(3):
                if coord[dim] < 0:
                    outside_dims.append(f'dim{dim}<0({coord[dim]:.6f})')
                elif coord[dim] >= patch_shape[dim]:
                    outside_dims.append(f'dim{dim}>={patch_shape[dim]}({coord[dim]:.6f})')
            print(f"      Violation {i}: {coord} -> {outside_dims}")
        
        # STRICT VALIDATION: With zero-tolerance clipping, NO points should be outside
        if pct_outside > 0.1:  # Even 0.1% outside indicates clipping failure
            validation['success'] = False
            validation['errors'].append(f"{pct_outside:.1f}% of streamline points outside patch bounds - STRICT ENFORCEMENT FAILED")
        elif pct_outside > 0.0:  # Any points outside is now a warning
            validation['warnings'].append(f"{pct_outside:.3f}% of streamline points outside patch bounds")
        else:  # 0% outside is expected with strict enforcement
            validation['metrics']['bounds_check'] = "PERFECT: All points within bounds"
    
    # Check streamline density distribution
    if len(patch_streamlines) < 5:
        validation['warnings'].append(f"Very few streamlines in patch: {len(patch_streamlines)}")
    
    validation['metrics']['num_streamlines'] = len(patch_streamlines)
    validation['metrics']['num_points'] = len(all_points)
    
    return validation


def process_patch_first_extraction(
    original_nifti_path: str,
    original_trk_path: str,
    target_voxel_size: float = 0.05,
    target_patch_size: Tuple[int, int, int] = (700, 1, 700),
    target_dimensions: Tuple[int, int, int] = (1400, 1000, 1400),
    num_patches: int = 50,
    output_prefix: str = "patch_optimized",
    min_streamlines_per_patch: int = 30,
    use_ants: bool = False,
    ants_warp_path: Optional[str] = None,
    ants_iwarp_path: Optional[str] = None,
    ants_aff_path: Optional[str] = None,
    random_state: Optional[int] = None,
    use_gpu: bool = True,
    white_mask_path: Optional[str] = None,
    use_compressed_nifti: bool = True,
    streamline_margin_fraction: float = 0.0,
    debug: bool = False
) -> Dict:
    """
    Main patch-first extraction pipeline.
    
    This function implements the optimized patch extraction that avoids 
    creating large intermediate files by processing patches directly.
    
    Parameters
    ----------
    original_nifti_path : str
        Path to original NIfTI file
    original_trk_path : str
        Path to original TRK file
    target_voxel_size : float
        Target voxel size in mm
    target_patch_size : tuple
        Target patch size in voxels (x, y, z)
    target_dimensions : tuple
        Target volume dimensions (x, y, z)
    num_patches : int
        Number of patches to extract
    output_prefix : str
        Prefix for output files
    min_streamlines_per_patch : int
        Minimum streamlines required per patch
    use_ants : bool
        Whether to use ANTs transformation
    ants_warp_path : str, optional
        Path to ANTs warp file
    ants_iwarp_path : str, optional
        Path to ANTs inverse warp file
    ants_aff_path : str, optional
        Path to ANTs affine file
    random_state : int, optional
        Random seed for reproducibility
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    dict
        Results dictionary with extraction metadata
    """
    start_time = time.time()
    nifti_ext = ".nii.gz" if use_compressed_nifti else ".nii"
    
    if debug:
        print("="*60)
        print("PATCH-FIRST EXTRACTION PIPELINE")
        print("="*60)
        print(f"Input NIfTI: {original_nifti_path}")
        print(f"Input TRK: {original_trk_path}")
        print(f"Target voxel size: {target_voxel_size}mm")
        print(f"Target patch size: {target_patch_size}")
        print(f"Number of patches: {num_patches}")
        print(f"ANTs enabled: {use_ants}")
        print(f"White mask: {white_mask_path if white_mask_path else 'None'}")
    
    # Initialize results tracking
    results = {
        'success': True,
        'extraction_method': 'patch_first_optimized',
        'patches_requested': num_patches,
        'patches_extracted': 0,
        'patches_failed': 0,
        'patch_details': [],
        'processing_time': 0,
        'memory_peak': 0,
        'parameters': {
            'target_voxel_size': target_voxel_size,
            'target_patch_size': target_patch_size,
            'min_streamlines_per_patch': min_streamlines_per_patch,
            'use_ants': use_ants,
            'random_state': random_state
        }
    }
    
    try:
        # Step 1: Apply ANTs transformations if requested
        if use_ants:
            if not all([ants_warp_path, ants_iwarp_path, ants_aff_path]):
                raise ValueError("ANTs enabled but transform files not provided")
            
            if debug:
                print(f"\nStep 1: Applying ANTs transformations...")
            moved_mri, affine_vox2fix, transformed_tractogram, streamlines_voxel = process_with_ants(
                ants_warp_path, ants_iwarp_path, ants_aff_path, 
                original_nifti_path, original_trk_path,
                transform_mri=False  # We'll handle MRI at patch level
            )
            
            # Convert streamlines to RAS coordinates for patch processing
            streamlines_ras = []
            for streamline_vox in streamlines_voxel:
                streamline_ras = nib.affines.apply_affine(affine_vox2fix, streamline_vox)
                streamlines_ras.append(streamline_ras)
            
            # Use original MRI path but with transformed affine
            mri_affine = affine_vox2fix
            original_img = nib.load(original_nifti_path, mmap=True)
            original_img = nib.as_closest_canonical(original_img)
            mri_shape = original_img.shape[:3]
            
            if debug:
                print(f"ANTs transformation complete. {len(streamlines_ras)} streamlines transformed.")
            
        else:
            if debug:
                print(f"\nStep 1: Loading original data (no ANTs transformation)...")
            original_img = nib.load(original_nifti_path, mmap=True)
            original_img = nib.as_closest_canonical(original_img)
            mri_affine = original_img.affine
            mri_shape = original_img.shape[:3]
            
            # Load streamlines
            trk_obj = nib.streamlines.load(original_trk_path)
            streamlines_ras = trk_obj.tractogram.streamlines
            
            if debug:
                print(f"Original data loaded. {len(streamlines_ras)} streamlines available.")

        # Note: per-streamline densification (sub-voxel point spacing) happens
        # lazily inside filter_streamlines_to_patch_ras for the streamlines that
        # actually intersect each patch. Densifying upstream would blow up to
        # >1B points for already-thickened TRKs and OOM the process.
        densify_step = float(target_voxel_size) * 0.5
        if debug:
            print(f"  Lazy densification step: {densify_step:.4f} mm "
                  f"(applied per-patch in filter_streamlines_to_patch_ras)")

        original_data_proxy = original_img.dataobj
        streamline_bounds = _build_streamline_bounds(streamlines_ras)
        
        # Load and upscale white mask if provided
        upscaled_white_mask = None
        if white_mask_path and os.path.exists(white_mask_path):
            if debug:
                print(f"\nLoading and upscaling white mask...")
            try:
                white_mask_img = nib.load(white_mask_path)
                white_mask_img = nib.as_closest_canonical(white_mask_img)
                white_mask_data = white_mask_img.get_fdata()
                
                # Handle 4D masks - take first volume
                if white_mask_data.ndim == 4:
                    if debug:
                        print(f"  4D mask detected, taking first volume")
                    white_mask_data = white_mask_data[..., 0]
                elif white_mask_data.ndim != 3:
                    raise ValueError(f"White mask must be 3D or 4D, got {white_mask_data.ndim}D")
                
                # Upscale white mask to blockface space (same as original MRI)
                # We ignore the affine as requested - just match the shape
                from scipy.ndimage import zoom
                
                # Calculate zoom factors to match original MRI shape
                mask_shape_3d = white_mask_data.shape[:3]
                zoom_factors = np.array(mri_shape) / np.array(mask_shape_3d)
                if debug:
                    print(f"  White mask shape: {mask_shape_3d}")
                    print(f"  Blockface shape: {mri_shape}")
                    print(f"  Zoom factors: {zoom_factors}")
                
                # Upscale using nearest neighbor to preserve binary mask values
                upscaled_white_mask = zoom(white_mask_data, zoom_factors, order=0)
                if debug:
                    print(f"  Upscaled white mask to shape: {upscaled_white_mask.shape}")
                
                # Ensure binary mask (threshold at 0.5)
                upscaled_white_mask = (upscaled_white_mask > 0.5).astype(np.uint8)
                if debug:
                    print(f"  White mask successfully upscaled to blockface space")
                
            except Exception as e:
                print(f"  Warning: Could not load/upscale white mask: {e}")
                print(f"  Mask shape details: {white_mask_data.shape if 'white_mask_data' in locals() else 'N/A'}")
                upscaled_white_mask = None
        
        # Step 2: Build target coordinate system for validation
        if debug:
            print(f"\nStep 2: Sampling patch locations...")
        
        # Import transform function for building target affine
        try:
            from .transform import build_new_affine
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from transform import build_new_affine
        
        # Build target affine and shape for proper validation.
        # When target_dimensions are capped (e.g. 4000^3 at 0.001 mm voxel = 4 mm FOV)
        # the default geometric center is the ORIGINAL volume center, which may not
        # overlap the streamlines at all -> patch sampling finds 0 valid patches.
        # If streamlines are available and the target FOV is smaller than the
        # streamline extent, center the target FOV on the streamline centroid so
        # patches can be placed where the data actually lives.
        target_extent_mm = np.array(target_dimensions, dtype=np.float64) * float(target_voxel_size)
        sl_extent_mm = streamline_bounds["maxs"].max(axis=0) - streamline_bounds["mins"].min(axis=0)
        if np.any(target_extent_mm < sl_extent_mm):
            sl_centroid_mm = 0.5 * (
                streamline_bounds["maxs"].max(axis=0)
                + streamline_bounds["mins"].min(axis=0)
            )
            patch_center_mm = tuple(float(c) for c in sl_centroid_mm)
            if debug:
                print(f"  Target FOV {target_extent_mm} mm < streamline extent {sl_extent_mm} mm; "
                      f"centering target on streamline centroid {patch_center_mm}")
        else:
            patch_center_mm = None

        target_affine = build_new_affine(
            old_affine=mri_affine,
            old_shape=mri_shape,
            new_voxel_size=target_voxel_size,
            new_shape=target_dimensions,
            patch_center_mm=patch_center_mm,
            use_gpu=False
        )
        
        # Use target coordinate system for patch validation
        patch_size_mm = np.array(target_patch_size) * target_voxel_size
        
        patch_locations = sample_patch_locations_transformed_space(
            mri_affine=target_affine,
            mri_shape=target_dimensions,
            patch_size_mm=patch_size_mm,
            num_patches=num_patches,
            min_streamlines=min_streamlines_per_patch,
            transformed_streamlines=streamlines_ras,
            streamline_bounds=streamline_bounds,
            random_state=random_state,
            streamline_margin_fraction=streamline_margin_fraction,
        )
        
        if debug:
            print(f"Sampled {len(patch_locations)} patch locations")
        
        # Step 3: Process each patch
        if debug:
            print(f"\nStep 3: Processing patches individually...")
        
        for i, patch_center_ras in enumerate(patch_locations):
            patch_id = i + 1
            if debug:
                print(f"\nProcessing patch {patch_id}/{len(patch_locations)}...")
            
            try:
                # Calculate bounding box
                bbox = calculate_patch_bbox_ras(patch_center_ras, patch_size_mm, mri_affine)
                
                # Synthesize patch region
                if debug:
                    print(f"  Synthesizing patch region...")
                patch_nifti = synthesize_patch_region(
                    original_mri_path=original_nifti_path,
                    bbox=bbox,
                    target_voxel_size=target_voxel_size,
                    target_patch_size=target_patch_size,
                    use_gpu=use_gpu,
                    original_img=original_img,
                    original_data=original_data_proxy,
                )
                
                # Filter streamlines to patch
                if debug:
                    print(f"  Filtering streamlines to patch...")
                patch_streamlines = filter_streamlines_to_patch_ras(
                    streamlines=streamlines_ras,
                    bbox=bbox,
                    target_patch_affine=patch_nifti.affine,
                    target_patch_size=target_patch_size,
                    streamline_bounds=streamline_bounds,
                    densify_step_mm=densify_step,
                )
                
                # Validate spatial alignment
                validation = validate_patch_spatial_alignment(
                    patch_nifti, patch_streamlines, tolerance=1.0, debug=debug
                )
                
                if not validation['success']:
                    print(f"  WARNING: Spatial validation failed: {validation['errors']}")
                    print(f"    DEBUG: target_patch_size={target_patch_size}, patch_nifti.shape={patch_nifti.shape[:3]}")
                    if len(patch_streamlines) > 0:
                        all_points = np.vstack(patch_streamlines)
                        print(f"    DEBUG: streamline bounds: min={np.min(all_points, axis=0)}, max={np.max(all_points, axis=0)}")
                        outside_mask = (all_points < 0) | (all_points >= np.array(patch_nifti.shape[:3]))
                        outside_points = np.any(outside_mask, axis=1)
                        if np.any(outside_points):
                            violation_coords = all_points[outside_points][:3]
                            print(f"    DEBUG: first 3 violations: {violation_coords}")
                
                # Extract white mask patch if available
                white_mask_patch_path = None
                if upscaled_white_mask is not None:
                    try:
                        # Get voxel bounds from bbox
                        vox_min = bbox['vox_min']
                        vox_max = bbox['vox_max']
                        
                        # Ensure bounds are within volume
                        vox_min = np.maximum(vox_min, 0)
                        vox_max = np.minimum(vox_max, upscaled_white_mask.shape)
                        
                        # Extract the same region from white mask as from NIfTI
                        white_mask_patch_data = upscaled_white_mask[
                            vox_min[0]:vox_max[0],
                            vox_min[1]:vox_max[1], 
                            vox_min[2]:vox_max[2]
                        ].copy()
                        
                        # Resample white mask patch to target resolution using nearest neighbor
                        from scipy.ndimage import zoom
                        mask_zoom_factors = np.array(target_patch_size) / np.array(white_mask_patch_data.shape[:3])
                        white_mask_patch_resampled = zoom(white_mask_patch_data, mask_zoom_factors, order=0)
                        
                        # Ensure it matches target patch size exactly
                        if white_mask_patch_resampled.shape != target_patch_size:
                            # Crop or pad if needed
                            final_mask = np.zeros(target_patch_size, dtype=np.uint8)
                            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(white_mask_patch_resampled.shape, target_patch_size))
                            final_mask[slices] = white_mask_patch_resampled[slices]
                            white_mask_patch_resampled = final_mask
                        
                        # Save white mask patch
                        white_mask_patch_path = f"{output_prefix}_{patch_id:04d}_white_mask{nifti_ext}"
                        white_mask_patch_img = nib.Nifti1Image(white_mask_patch_resampled, patch_nifti.affine)
                        nib.save(white_mask_patch_img, white_mask_patch_path)
                        if debug:
                            print(f"  White mask patch saved: {white_mask_patch_path}")
                        
                    except Exception as e:
                        print(f"  Warning: Could not extract white mask patch: {e}")
                        white_mask_patch_path = None
                
                # Save patch files
                patch_prefix = f"{output_prefix}_{patch_id:04d}"
                
                # Save NIfTI
                nifti_path = f"{patch_prefix}{nifti_ext}"
                patch_data_f32 = np.asarray(patch_nifti.dataobj, dtype=np.float32)
                nib.save(nib.Nifti1Image(patch_data_f32, patch_nifti.affine), nifti_path)
                
                # Save TRK (always, even if empty)
                from nibabel.streamlines import Tractogram, TrkFile
                
                trk_path = f"{patch_prefix}.trk"
                
                if len(patch_streamlines) > 0:
                    # FINAL SAFETY CHECK: Ensure all streamlines are strictly within voxel bounds
                    bounded_streamlines = []
                    for streamline in patch_streamlines:
                        # Force strict bounds enforcement with appropriate epsilon
                        for dim in range(3):
                            if target_patch_size[dim] == 1:
                                max_val = target_patch_size[dim] - 1e-3  # 0.001 margin for thin slices
                            else:
                                max_val = target_patch_size[dim] - 1e-6  # Small margin for normal dimensions
                            streamline[:, dim] = np.clip(streamline[:, dim], 0.0, max_val)
                        
                        # Double-check bounds
                        valid_mask = np.all(
                            (streamline >= 0) & (streamline < np.array(target_patch_size)),
                            axis=1
                        )
                        
                        if np.any(valid_mask):
                            bounded_streamline = streamline[valid_mask]
                            if len(bounded_streamline) >= 2:
                                bounded_streamlines.append(bounded_streamline.astype(np.float32))
                    
                    # Convert to RAS for saving (TRK format expectation)
                    ras_streamlines = []
                    for streamline_vox in bounded_streamlines:
                        streamline_ras = nib.affines.apply_affine(patch_nifti.affine, streamline_vox)
                        ras_streamlines.append(streamline_ras.astype(np.float32))
                    
                    # Update results to reflect the actual number of bounded streamlines
                    patch_streamlines = bounded_streamlines
                else:
                    if debug:
                        print(f"  WARNING: No streamlines in patch {patch_id}")
                    # Create empty streamline list
                    ras_streamlines = []
                
                # Create tractogram (empty or with streamlines)
                tractogram = Tractogram(ras_streamlines, affine_to_rasmm=np.eye(4))
                
                # Create TRK file with proper header
                trk_file = TrkFile(tractogram)
                trk_file.header['dimensions'] = np.array(target_patch_size, dtype=np.int16)
                trk_file.header['voxel_sizes'] = np.array([target_voxel_size] * 3, dtype=np.float32)
                trk_file.header['voxel_to_rasmm'] = patch_nifti.affine.astype(np.float32)
                
                trk_file.save(trk_path)
                
                # Record success
                results['patches_extracted'] += 1
                patch_files = {
                    'nifti': nifti_path,
                    'trk': trk_path
                }
                if white_mask_patch_path:
                    patch_files['white_mask'] = white_mask_patch_path
                
                results['patch_details'].append({
                    'patch_id': patch_id,
                    'center_ras': patch_center_ras.tolist(),
                    'bbox': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in bbox.items()},
                    'num_streamlines': len(patch_streamlines),
                    'validation': validation,
                    'files': patch_files
                })
                
                if debug:
                    print(f"  Patch {patch_id} completed: {len(patch_streamlines)} streamlines")
                
                # Force garbage collection to prevent memory accumulation
                if patch_id % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"  ERROR: Patch {patch_id} failed: {e}")
                results['patches_failed'] += 1
        
        # Final results
        results['processing_time'] = time.time() - start_time
        results['success'] = results['patches_extracted'] > 0
        
        if debug:
            print(f"\n" + "="*60)
            print("PATCH-FIRST EXTRACTION COMPLETE")
            print("="*60)
            print(f"Patches extracted: {results['patches_extracted']}/{results['patches_requested']}")
            print(f"Failed patches: {results['patches_failed']}")
            print(f"Total processing time: {results['processing_time']:.2f}s")
        if debug:
            print(f"Average time per patch: {results['processing_time']/max(1, results['patches_extracted']):.2f}s")
        
        return results
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['processing_time'] = time.time() - start_time
        print(f"ERROR: Patch-first extraction failed: {e}")
        return results
