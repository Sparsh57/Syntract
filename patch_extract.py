#!/usr/bin/env python3
"""
Robust patch extractor for paired NIfTI (.nii/.nii.gz) and TRK (.trk) files
that share identical volume dimensions. Given a target patch size, this script:
  1) Validates NIfTI/TRK dimensional consistency
  2) Samples a random patch location that fits entirely inside the volume
  3) Crops the NIfTI to the patch and filters/retargets TRK streamlines
  4) Writes out patch_nifti and patch_trk with proper coordinate transformations
  5) Re-loads outputs and validates shape & spatial location

This replaces the previous patch extraction methodology with a more robust approach
that ensures proper coordinate transformations and dimensional consistency.

Usage:
  python patch_extract.py \
    --nifti input.nii.gz \
    --trk input.trk \
    --patch 50 50 50 \
    --seed 42 \
    --out_prefix sample_patch

Outputs:
  sample_patch.nii.gz
  sample_patch.trk
  sample_patch.meta.json  (records origin, patch size, validation summary)

Dependencies:
  nibabel >= 5.0, numpy
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from typing import Tuple, List
import os

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine


def validate_patch_extraction_environment():
    """
    Validate that the environment has all necessary dependencies for patch extraction.
    
    Returns:
        dict: Validation results with status and any missing dependencies
    """
    validation_results = {
        'success': True,
        'missing_dependencies': [],
        'version_info': {},
        'warnings': []
    }
    
    # Check nibabel
    try:
        validation_results['version_info']['nibabel'] = nib.__version__
        if tuple(map(int, nib.__version__.split('.')[:2])) < (5, 0):
            validation_results['warnings'].append(f"nibabel version {nib.__version__} may not have full compatibility. Recommended: >= 5.0")
    except ImportError:
        validation_results['success'] = False
        validation_results['missing_dependencies'].append('nibabel')
    
    # Check numpy
    try:
        validation_results['version_info']['numpy'] = np.__version__
    except ImportError:
        validation_results['success'] = False
        validation_results['missing_dependencies'].append('numpy')
    
    return validation_results


def validate_input_files(nifti_path: str, trk_path: str) -> dict:
    """
    Comprehensive validation of input NIfTI and TRK files.
    
    Parameters
    ----------
    nifti_path : str
        Path to NIfTI file
    trk_path : str
        Path to TRK file
        
    Returns
    -------
    dict
        Validation results including file existence, format checks, and compatibility
    """
    validation = {
        'success': True,
        'errors': [],
        'warnings': [],
        'nifti_info': {},
        'trk_info': {},
        'compatibility': {}
    }
    
    # Check file existence
    if not os.path.exists(nifti_path):
        validation['success'] = False
        validation['errors'].append(f"NIfTI file not found: {nifti_path}")
        return validation
        
    if not os.path.exists(trk_path):
        validation['success'] = False
        validation['errors'].append(f"TRK file not found: {trk_path}")
        return validation
    
    try:
        # Fast loading without validation overhead
        nimg = nib.load(nifti_path, mmap=False)  # Direct loading for speed
        data = nimg.get_fdata().astype(np.float32)
        
        validation['nifti_info']['shape'] = data.shape
        validation['nifti_info']['dtype'] = str(data.dtype)
        validation['nifti_info']['affine_det'] = np.linalg.det(nimg.affine[:3, :3])
        validation['nifti_info']['voxel_sizes'] = nib.affines.voxel_sizes(nimg.affine)
        
        # Check for reasonable dimensions
        if any(dim < 10 for dim in data.shape[:3]):
            validation['warnings'].append(f"Very small NIfTI dimensions: {data.shape[:3]}")
        if any(dim > 2000 for dim in data.shape[:3]):
            validation['warnings'].append(f"Very large NIfTI dimensions: {data.shape[:3]}")
            
        # Load and validate TRK
        trk = nib.streamlines.load(trk_path)
        validation['trk_info']['n_streamlines'] = len(trk.streamlines)
        validation['trk_info']['header_dims'] = _hdr_dimensions_from_trk(trk)
        
        # Check streamline count
        if len(trk.streamlines) == 0:
            validation['warnings'].append("TRK file contains no streamlines")
        elif len(trk.streamlines) < 100:
            validation['warnings'].append(f"Very few streamlines in TRK: {len(trk.streamlines)}")
            
        # Check compatibility
        nifti_dims = nimg.shape[:3]
        trk_dims = validation['trk_info']['header_dims']
        
        if nifti_dims != trk_dims:
            validation['success'] = False
            validation['errors'].append(f"Dimension mismatch: NIfTI {nifti_dims} vs TRK {trk_dims}")
        else:
            validation['compatibility']['dimensions_match'] = True
            
        # Check if streamlines are within volume bounds
        if validation['trk_info']['n_streamlines'] > 0:
            all_points = np.vstack([sl for sl in trk.streamlines])
            inv_affine = np.linalg.inv(nimg.affine)
            voxel_coords = apply_affine(inv_affine, all_points)
            
            bounds_check = {
                'x_in_bounds': np.all((voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < nifti_dims[0])),
                'y_in_bounds': np.all((voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < nifti_dims[1])),
                'z_in_bounds': np.all((voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < nifti_dims[2]))
            }
            
            if not all(bounds_check.values()):
                validation['warnings'].append("Some streamlines extend outside NIfTI volume bounds")
            
            validation['compatibility']['bounds_check'] = bounds_check
            
    except Exception as e:
        validation['success'] = False
        validation['errors'].append(f"Error during file validation: {e}")
    
    return validation


def validate_patch_parameters(vol_shape: Tuple[int, int, int], 
                            patch_size: Tuple[int, int, int],
                            num_patches: int = 1) -> dict:
    """
    Validate patch extraction parameters.
    
    Parameters
    ----------
    vol_shape : tuple
        Volume dimensions (x, y, z)
    patch_size : tuple  
        Patch dimensions (x, y, z)
    num_patches : int
        Number of patches to extract
        
    Returns
    -------
    dict
        Validation results with recommendations
    """
    validation = {
        'success': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check patch size vs volume
    for i, (vol_dim, patch_dim) in enumerate(zip(vol_shape, patch_size)):
        axis_name = ['X', 'Y', 'Z'][i]
        if patch_dim > vol_dim:
            validation['success'] = False
            validation['errors'].append(f"Patch {axis_name} dimension ({patch_dim}) larger than volume ({vol_dim})")
        elif patch_dim > vol_dim * 0.8:
            validation['warnings'].append(f"Patch {axis_name} dimension ({patch_dim}) is > 80% of volume ({vol_dim})")
    
    # Check for very small patches
    min_reasonable_size = 32
    if any(dim < min_reasonable_size for dim in patch_size):
        validation['warnings'].append(f"Very small patch size {patch_size}. Consider >= {min_reasonable_size} per dimension")
    
    # Check number of patches vs available space
    if all(patch_dim <= vol_dim for patch_dim, vol_dim in zip(patch_size, vol_shape)):
        # Estimate maximum non-overlapping patches
        max_patches = np.prod([(vol_dim // patch_dim) for vol_dim, patch_dim in zip(vol_shape, patch_size)])
        if num_patches > max_patches * 10:  # Allow some overlap
            validation['warnings'].append(f"Requesting {num_patches} patches, but only ~{max_patches} non-overlapping patches fit")
    
    # Recommendations
    if validation['success']:
        volume_size = np.prod(vol_shape)
        patch_volume = np.prod(patch_size)
        coverage = patch_volume / volume_size
        
        if coverage < 0.01:
            validation['recommendations'].append("Very small patches relative to volume. Consider larger patch size for better representation")
        elif coverage > 0.3:
            validation['recommendations'].append("Large patches relative to volume. Consider smaller patch size for more diversity")
    
    return validation


@dataclass
class PatchSpec:
    size: Tuple[int, int, int]  # (X, Y, Z)
    origin: Tuple[int, int, int]  # start indices in voxels


def _hdr_dimensions_from_trk(trk: nib.streamlines.tractogram.TrkFile) -> Tuple[int, int, int]:
    """Extract dimensions from TRK header."""
    hdr = trk.header
    # nibabel TRK header uses key 'dimensions' with order (x, y, z)
    dims = tuple(int(v) for v in hdr.get('dimensions', [0, 0, 0]))
    if any(d <= 0 for d in dims):
        raise ValueError("TRK header missing/invalid 'dimensions'.")
    return dims  # (X, Y, Z)


def _choose_random_origin(vol_shape: Tuple[int, int, int], 
                          patch: Tuple[int, int, int], 
                          rng: np.random.Generator) -> Tuple[int, int, int]:
    """Choose a random patch origin that ensures patch fits within volume bounds."""
    # Calculate maximum valid origin for each dimension
    max_origins = [max(s - p, 0) for s, p in zip(vol_shape, patch)]
    
    if any(p > s for p, s in zip(patch, vol_shape)):
        raise ValueError(f"Patch {patch} does not fit inside volume {vol_shape}.")
    
    # Generate random origins within valid ranges
    origins = tuple(rng.integers(0, max_o + 1) if max_o > 0 else 0 for max_o in max_origins)
    return origins


def _crop_nifti(nimg: nib.Nifti1Image, spec: PatchSpec) -> nib.Nifti1Image:
    """Crop NIfTI image to patch specification with proper affine transformation."""
    data = nimg.get_fdata(dtype=np.float32)
    x0, y0, z0 = spec.origin
    sx, sy, sz = spec.size
    
    # Extract patch data
    cropped = data[x0:x0+sx, y0:y0+sy, z0:z0+sz]
    if cropped.shape != spec.size:
        raise RuntimeError(f"NIfTI crop produced shape {cropped.shape}, expected {spec.size}.")

    # Build a fresh header whose dimensions match the cropped data
    aff = nimg.affine.copy()
    
    # Calculate world coordinate shift for patch origin
    origin_shift_world = aff[:3, :3] @ np.array([x0, y0, z0], dtype=float)
    new_aff = aff.copy()
    new_aff[:3, 3] = aff[:3, 3] + origin_shift_world

    # Create new header with correct dimensions
    new_hdr = nimg.header.copy()
    new_hdr.set_data_shape(cropped.shape)
    
    # Set qform/sform to the patch affine
    try:
        new_hdr.set_qform(new_aff, code=1)
        new_hdr.set_sform(new_aff, code=1)
    except Exception:
        # Fallback if header modification fails
        pass

    return nib.Nifti1Image(cropped, new_aff, new_hdr)


def _filter_and_retarget_trk(trk: nib.streamlines.tractogram.TrkFile, 
                              src_affine: np.ndarray, 
                              spec: PatchSpec, 
                              patch_affine: np.ndarray) -> nib.streamlines.tractogram.TrkFile:
    """
    Filter and retarget TRK streamlines for patch extraction.
    
    Return a new TRK whose streamlines are clipped to the patch and re-expressed
    in the patch's RASMM space defined by patch_affine.
    - Convert TRK points (RASMM) -> voxel (of source nifti) using inv(src_affine)
    - Keep only points within [origin, origin+size)
    - Rebase kept points to patch voxel coords by subtracting origin
    - Convert to RASMM using patch_affine
    - Discard streamlines with < 2 points after clipping
    """
    from nibabel.streamlines import Tractogram, TrkFile

    inv_src = np.linalg.inv(src_affine)
    x0, y0, z0 = spec.origin
    sx, sy, sz = spec.size

    def in_bounds(v):
        """Check if points are within patch bounds."""
        return (x0 <= v[..., 0]) & (v[..., 0] < x0 + sx) & \
               (y0 <= v[..., 1]) & (v[..., 1] < y0 + sy) & \
               (z0 <= v[..., 2]) & (v[..., 2] < z0 + sz)

    new_streamlines: List[np.ndarray] = []
    
    for sl in trk.streamlines:
        # Convert streamline from RASMM to voxel coordinates
        vox = apply_affine(inv_src, sl)
        mask = in_bounds(vox)
        
        if not np.any(mask):
            continue  # No points within patch bounds
            
        # STRICT BOUNDS: Keep segments that intersect patch but clip ALL coordinates to boundaries
        # This preserves curvature through longer segments while respecting exact patch dimensions
        
        # Find the first and last points within bounds
        indices_in_bounds = np.where(mask)[0]
        if len(indices_in_bounds) < 2:
            continue  # Need at least 2 points for a segment
            
        first_in = indices_in_bounds[0]
        last_in = indices_in_bounds[-1]
        
        # Get extended segment for curvature (but will clip coordinates to bounds)
        context_points = 5  # Include more points for better curvature context
        start_idx = max(0, first_in - context_points)
        end_idx = min(len(vox), last_in + context_points + 1)
        
        # Keep the extended segment
        vox_segment = vox[start_idx:end_idx]
        
        # STRICT CLIPPING: Ensure ALL coordinates stay within patch boundaries
        # Clamp X coordinates to patch bounds
        vox_segment[vox_segment[..., 0] < x0, 0] = x0 + 0.01  # Just inside X lower bound
        vox_segment[vox_segment[..., 0] >= x0 + sx, 0] = x0 + sx - 0.01  # Just inside X upper bound
        
        # Clamp Y coordinates to patch bounds  
        vox_segment[vox_segment[..., 1] < y0, 1] = y0 + 0.01  # Just inside Y lower bound
        vox_segment[vox_segment[..., 1] >= y0 + sy, 1] = y0 + sy - 0.01  # Just inside Y upper bound
        
        # Clamp Z coordinates to patch bounds
        vox_segment[vox_segment[..., 2] < z0, 2] = z0 + 0.01  # Just inside Z lower bound
        vox_segment[vox_segment[..., 2] >= z0 + sz, 2] = z0 + sz - 0.01  # Just inside Z upper bound
        
        vox_patch = vox_segment.copy()
        
        # Convert back to RASMM using patch affine  
        ras_patch = apply_affine(patch_affine, vox_patch - np.array([x0, y0, z0], dtype=float))
        
        if ras_patch.shape[0] >= 5:  # Need at least 5 points to show curvature
            new_streamlines.append(ras_patch.astype(np.float32))

    # Build new tractogram in RASMM coordinate system
    tgram = nib.streamlines.Tractogram(streamlines=new_streamlines, affine_to_rasmm=np.eye(4))
    
    # Ensure header consistency: dimensions & voxel sizes should reflect the patch
    hdr = trk.header.copy()
    
    # Set dimensions to patch size (in voxels)
    hdr['dimensions'] = np.asarray(spec.size, dtype=np.int16)
    
    # Calculate voxel sizes from source affine (magnitudes of axis vectors)
    vox_sizes = np.sqrt((src_affine[:3, :3] ** 2).sum(axis=0))
    hdr['voxel_sizes'] = vox_sizes.astype(np.float32)
    hdr['voxel_to_rasmm'] = patch_affine.astype(np.float32)

    return TrkFile(tgram, hdr)


def _validate_consistency(nimg: nib.Nifti1Image, trk: nib.streamlines.tractogram.TrkFile):
    """Validate dimensional consistency between NIfTI and TRK files."""
    vol_shape = tuple(int(s) for s in nimg.shape[:3])
    trk_dims = _hdr_dimensions_from_trk(trk)
    
    if trk_dims != vol_shape:
        raise ValueError(f"Input dimensions disagree: NIfTI {vol_shape} vs TRK header {trk_dims}.")


def _validate_outputs(out_img: nib.Nifti1Image, 
                     out_trk: nib.streamlines.tractogram.TrkFile, 
                     spec: PatchSpec, 
                     src_img: nib.Nifti1Image):
    """Comprehensive validation of patch extraction outputs."""
    
    # 1) Shape check
    if tuple(out_img.shape[:3]) != spec.size:
        raise RuntimeError(f"Output NIfTI shape {out_img.shape[:3]} != requested {spec.size}.")

    # 2) TRK header check
    out_dims = _hdr_dimensions_from_trk(out_trk)
    if out_dims != spec.size:
        raise RuntimeError(f"Output TRK header dimensions {out_dims} != requested {spec.size}.")

    # 3) Spatial location check: compare center voxel world coordinates
    src_aff = src_img.affine
    out_aff = out_img.affine

    # Calculate center coordinates
    cx_src = np.array(spec.origin) + (np.array(spec.size) / 2.0)
    cx_out = np.array(spec.size) / 2.0

    # Convert to world coordinates
    w_src = apply_affine(src_aff, cx_src)
    w_out = apply_affine(out_aff, cx_out)

    if not np.allclose(w_src, w_out, atol=1e-3):
        raise RuntimeError(f"Patch spatial misalignment: center world {w_out} vs expected {w_src}.")

    # 4) Intensity sanity check: flag if all zeros/NaNs (helps catch header/dim issues)
    arr = out_img.get_fdata(dtype=np.float32)
    if not np.isfinite(arr).any() or (np.nanmin(arr) == 0 and np.nanmax(arr) == 0):
        # Only raise if there were streamlines kept; otherwise empty background is plausible
        if len(out_trk.streamlines) > 0:
            raise RuntimeError("Patch image appears empty (all zeros/NaNs) despite non-empty TRK; "
                             "likely header/affine or slicing mismatch.")


def extract_single_patch(nifti_path: str, 
                        trk_path: str, 
                        patch_xyz: Tuple[int, int, int], 
                        seed: int | None, 
                        out_prefix: str,
                        min_streamlines: int = 30, 
                        max_trials: int = 100,
                        nimg_cached = None,
                        trk_cached = None) -> dict:
    """
    Extract a single random patch from NIfTI and TRK files with validation.
    
    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI file
    trk_path : str
        Path to input TRK file
    patch_xyz : tuple of int
        Patch dimensions (x, y, z) in voxels
    seed : int or None
        Random seed for reproducibility
    out_prefix : str
        Output prefix for generated files
    min_streamlines : int
        Minimum number of streamlines required in patch
    max_trials : int
        Maximum number of patch locations to try
    nimg_cached : nibabel.Nifti1Image, optional
        Pre-loaded NIfTI image to avoid redundant I/O
    trk_cached : nibabel.streamlines.TrkFile, optional
        Pre-loaded TRK file to avoid redundant I/O
        
    Returns
    -------
    dict
        Results dictionary with metadata and validation info
    """
    
    # Validate environment (only on first call)
    if nimg_cached is None:
        env_validation = validate_patch_extraction_environment()
        if not env_validation['success']:
            raise RuntimeError(f"Environment validation failed: {env_validation['missing_dependencies']}")

        # Validate input files
        file_validation = validate_input_files(nifti_path, trk_path)
        if not file_validation['success']:
            raise ValueError(f"Input file validation failed: {'; '.join(file_validation['errors'])}")
    
    # Load inputs with memory-mapping for efficiency
    if nimg_cached is not None:
        nimg = nimg_cached
    else:
        # Use memory-mapping to avoid loading entire volume into RAM
        nimg = nib.load(nifti_path, mmap=True)
    
    if trk_cached is not None:
        trk = trk_cached
    else:
        trk = nib.streamlines.load(trk_path)

    # Basic dimensional consistency check (redundant but explicit)
    _validate_consistency(nimg, trk)

    vol_shape = tuple(int(s) for s in nimg.shape[:3])
    
    # Validate patch parameters
    patch_validation = validate_patch_parameters(vol_shape, patch_xyz, num_patches=1)
    if not patch_validation['success']:
        raise ValueError(f"Patch parameter validation failed: {'; '.join(patch_validation['errors'])}")
    
    # Skip printing warnings and recommendations to reduce verbosity

    # Choose patch origins until threshold met (rejection sampling)
    rng = np.random.default_rng(seed)

    best = None
    best_kept = -1
    trials = 0

    # Search for suitable patch location (silent unless debugging)
    while trials < max_trials:
        origin = _choose_random_origin(vol_shape, patch_xyz, rng)
        spec = PatchSpec(size=patch_xyz, origin=origin)

        # Crop NIfTI & construct patch affine
        out_img = _crop_nifti(nimg, spec)
        
        # Filter/retarget TRK into patch space
        out_trk = _filter_and_retarget_trk(trk, nimg.affine, spec, out_img.affine)

        kept = len(out_trk.streamlines)
        if kept > best_kept:
            best = (spec, out_img, out_trk, kept)
            best_kept = kept

        trials += 1
        # Silent progress - only report final result
        if kept >= min_streamlines:
            best = (spec, out_img, out_trk, kept)
            best_kept = kept
            break

    if best is None:
        raise RuntimeError("Failed to generate a patch (no attempts completed).")

    spec, out_img, out_trk, kept = best
    
    # Skip warning about streamline count to reduce verbosity
    
    # Save outputs
    nii_out = f"{out_prefix}.nii.gz"
    trk_out = f"{out_prefix}.trk"
    nib.save(out_img, nii_out)
    nib.streamlines.save(out_trk, trk_out)

    # Validate by reloading (silent validation)
    v_img = nib.load(nii_out)
    v_trk = nib.streamlines.load(trk_out)
    _validate_outputs(v_img, v_trk, spec, nimg)

    # Write meta summary
    def _to_native(o):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(o, np.generic):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (list, tuple)):
            return [_to_native(i) for i in o]
        elif isinstance(o, dict):
            return {k: _to_native(v) for k, v in o.items()}
        elif hasattr(o, '__dict__'):
            # Handle objects with __dict__ (like custom classes)
            return _to_native(o.__dict__)
        else:
            return o

    meta = {
        'nifti_in': str(nifti_path),
        'trk_in': str(trk_path),
        'out_prefix': str(out_prefix),
        'volume_shape': [int(s) for s in vol_shape],
        'patch_size': [int(s) for s in spec.size],
        'patch_origin': [int(s) for s in spec.origin],
        'seed': int(seed) if seed is not None else None,
        'validations': {
            'input_dimensional_consistency': True,
            'output_shape_ok': True,
            'spatial_alignment_ok': True,
            'streamlines_kept': int(len(v_trk.streamlines)),
            'trials': int(trials),
            'min_streamlines_target': int(min_streamlines),
        },
        'file_validation': file_validation,
        'patch_validation': patch_validation,
        'environment_validation': env_validation
    }
    
    with open(f"{out_prefix}.meta.json", 'w') as f:
        json.dump(_to_native(meta), f, indent=2)

    return meta


def extract_multiple_patches(nifti_path: str,
                           trk_path: str,
                           patch_xyz: Tuple[int, int, int],
                           num_patches: int,
                           output_dir: str,
                           seed: int | None = None,
                           prefix: str = "patch",
                           min_streamlines: int = 30,
                           max_trials: int = 100,
                           batch_size: int = 50) -> dict:
    """
    Extract multiple random patches from NIfTI and TRK files with memory-efficient batch processing.
    
    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI file
    trk_path : str  
        Path to input TRK file
    patch_xyz : tuple of int
        Patch dimensions (x, y, z) in voxels
    num_patches : int
        Number of patches to extract
    output_dir : str
        Output directory for patches
    seed : int or None
        Random seed for reproducibility
    prefix : str
        Prefix for output files
    min_streamlines : int
        Minimum number of streamlines required in patch
    max_trials : int
        Maximum number of patch locations to try per patch
    batch_size : int
        Number of patches to process before cleanup (default: 50)
        
    Returns
    -------
    dict
        Results dictionary with metadata for all patches
    """
    import gc
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up random state
    rng = np.random.default_rng(seed)
    
    results = {
        'input_nifti': str(nifti_path),
        'input_trk': str(trk_path),
        'patch_size': patch_xyz,
        'num_patches_requested': num_patches,
        'num_patches_extracted': 0,
        'num_patches_failed': 0,
        'output_dir': str(output_dir),
        'patches': [],
        'extraction_params': {
            'seed': seed,
            'min_streamlines': min_streamlines,
            'max_trials': max_trials,
            'prefix': prefix,
            'batch_size': batch_size
        }
    }
    
    print(f"=== Extracting {num_patches} patches (batch size: {batch_size}) ===")
    print(f"Input NIfTI: {nifti_path}")
    print(f"Input TRK: {trk_path}")
    print(f"Patch size: {patch_xyz}")
    print(f"Output directory: {output_dir}")
    print(f"Min streamlines per patch: {min_streamlines}")
    
    # Pre-load volumes once with memory-mapping
    print("\n🔄 Loading volumes with memory-mapping...")
    nimg_cached = nib.load(nifti_path, mmap=True)
    trk_cached = nib.streamlines.load(trk_path)
    print("✅ Volumes loaded")
    
    # Process in batches to control memory usage
    num_batches = (num_patches + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_patches)
        
        print(f"\n=== Batch {batch_idx + 1}/{num_batches}: Patches {batch_start + 1}-{batch_end} ===")
        
        for i in range(batch_start, batch_end):
            patch_id = i + 1
            patch_seed = rng.integers(0, 2**32-1) if seed is not None else None
            out_prefix = os.path.join(output_dir, f"{prefix}_{patch_id:04d}")
            
            print(f"Extracting patch {patch_id}/{num_patches}...", end=' ')
            
            try:
                meta = extract_single_patch(
                    nifti_path=nifti_path,
                    trk_path=trk_path,
                    patch_xyz=patch_xyz,
                    seed=patch_seed,
                    out_prefix=out_prefix,
                    min_streamlines=min_streamlines,
                    max_trials=max_trials,
                    nimg_cached=nimg_cached,
                    trk_cached=trk_cached
                )
                
                results['patches'].append({
                    'patch_id': patch_id,
                    'success': True,
                    'streamlines_kept': meta['validations']['streamlines_kept'],
                    'trials': meta['validations']['trials'],
                    'origin': meta['patch_origin'],
                    'files': {
                        'nifti': f"{out_prefix}.nii.gz",
                        'trk': f"{out_prefix}.trk",
                        'meta': f"{out_prefix}.meta.json"
                    }
                })
                
                results['num_patches_extracted'] += 1
                print(f"✓ {meta['validations']['streamlines_kept']} streamlines ({meta['validations']['trials']} trials)")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                results['patches'].append({
                    'patch_id': patch_id,
                    'success': False,
                    'error': str(e)
                })
                results['num_patches_failed'] += 1
        
        # Cleanup after each batch to prevent memory accumulation
        print(f"\n🧹 Cleaning up after batch {batch_idx + 1}...")
        gc.collect()
        
        # Save checkpoint after each batch
        checkpoint_path = os.path.join(output_dir, f"{prefix}_checkpoint_batch{batch_idx + 1}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'completed_batches': batch_idx + 1,
                'total_batches': num_batches,
                'patches_extracted': results['num_patches_extracted'],
                'patches_failed': results['num_patches_failed']
            }, f, indent=2)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save overall summary
    summary_path = os.path.join(output_dir, f"{prefix}_extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Successful patches: {results['num_patches_extracted']}/{num_patches}")
    print(f"Failed patches: {results['num_patches_failed']}")
    print(f"Summary saved: {summary_path}")
    
    return results


def run(nifti_path: str, trk_path: str, patch_xyz: Tuple[int, int, int], seed: int | None, out_prefix: str,
        min_streamlines: int = 30, max_trials: int = 100):
    """
    Main function for single patch extraction - legacy interface.
    """
    try:
        meta = extract_single_patch(nifti_path, trk_path, patch_xyz, seed, out_prefix,
                                   min_streamlines, max_trials)
        print("✔ Success")
        print(json.dumps(meta, indent=2))
    except Exception as e:
        # On failure, emit a structured diagnostic and exit non-zero
        diag = {
            'error': str(e),
            'hint': (
                'Check that input NIfTI and TRK have identical dimensions and affines; '
                'verify patch size fits within the volume; ensure TRK coordinates are in RASMM '
                'matching the NIfTI affine.'
            )
        }
        print(json.dumps(diag, indent=2), file=sys.stderr)
        sys.exit(1)


def main():
    """Command-line interface for patch extraction."""
    ap = argparse.ArgumentParser(
        description="Extract random 3D patches from NIfTI and corresponding TRK files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single patch
  python patch_extract.py --nifti brain.nii.gz --trk fibers.trk \\
    --patch 64 64 64 --seed 42 --out_prefix sample_patch

  # Extract multiple patches  
  python patch_extract.py --nifti brain.nii.gz --trk fibers.trk \\
    --patch 64 64 64 --num_patches 10 --output_dir patches/ \\
    --seed 42 --prefix patch
        """
    )
    
    # Input files
    ap.add_argument('--nifti', required=True, help='Input NIfTI (.nii/.nii.gz)')
    ap.add_argument('--trk', required=True, help='Input TRK (.trk)')
    
    # Patch specification
    ap.add_argument('--patch', required=True, nargs=3, type=int, 
                   metavar=('PX','PY','PZ'), help='Patch size in voxels (X Y Z)')
    
    # Single patch mode
    ap.add_argument('--out_prefix', help='Output prefix for single patch extraction')
    
    # Multiple patch mode  
    ap.add_argument('--num_patches', type=int, help='Number of patches to extract')
    ap.add_argument('--output_dir', help='Output directory for multiple patches')
    ap.add_argument('--prefix', default='patch', help='Prefix for patch files')
    
    # Common parameters
    ap.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    ap.add_argument('--min_streamlines', type=int, default=30, 
                   help='Try new origins until at least this many streamlines are kept (or max_trials reached).')
    ap.add_argument('--max_trials', type=int, default=100, 
                   help='Maximum number of random origins to try per patch.')
    
    args = ap.parse_args()

    # Validate arguments
    if args.out_prefix and (args.num_patches or args.output_dir):
        print("Error: Use either --out_prefix (single patch) OR --num_patches/--output_dir (multiple patches)", 
              file=sys.stderr)
        sys.exit(1)
        
    if not args.out_prefix and not (args.num_patches and args.output_dir):
        print("Error: Must specify either --out_prefix (single) OR --num_patches and --output_dir (multiple)", 
              file=sys.stderr)
        sys.exit(1)

    try:
        if args.out_prefix:
            # Single patch extraction
            run(args.nifti, args.trk, tuple(args.patch), args.seed, args.out_prefix,
                min_streamlines=args.min_streamlines, max_trials=args.max_trials)
        else:
            # Multiple patch extraction
            results = extract_multiple_patches(
                nifti_path=args.nifti,
                trk_path=args.trk, 
                patch_xyz=tuple(args.patch),
                num_patches=args.num_patches,
                output_dir=args.output_dir,
                seed=args.seed,
                prefix=args.prefix,
                min_streamlines=args.min_streamlines,
                max_trials=args.max_trials
            )
            
    except Exception as e:
        # On failure, emit a structured diagnostic and exit non-zero
        diag = {
            'error': str(e),
            'hint': (
                'Check that input NIfTI and TRK have identical dimensions and affines; '
                'verify patch size fits within the volume; ensure TRK coordinates are in RASMM '
                'matching the NIfTI affine.'
            )
        }
        print(json.dumps(diag, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()