#!/usr/bin/env python
"""
Robust Patch Extraction Module for Syntract Viewer

This module provides functionality to extract random patches from NIfTI volumes
with overlaid tractography data. It uses a robust methodology that ensures:
- Proper dimensional validation between NIfTI and TRK files
- Accurate coordinate transformations and spatial alignment
- Comprehensive validation of output patches
- Rejection sampling to ensure adequate streamline density
- Backward compatibility with existing visualization pipeline

Features:
- 3D patch extraction with proper bounds checking
- Robust coordinate transformation handling 
- Spatial consistency validation
- Multiple TRK file support
- Integrated visualization generation
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.streamlines import load
from nibabel.affines import apply_affine
import random
from typing import List, Tuple, Dict, Optional
import json

try:
    from .core import visualize_nifti_with_trk_coronal
    from .masking import create_fiber_mask
    from .contrast import apply_enhanced_contrast_and_augmentation
    from .effects import apply_balanced_dark_field_effect
    from .utils import select_random_streamlines
except ImportError:
    from core import visualize_nifti_with_trk_coronal
    from masking import create_fiber_mask
    from contrast import apply_enhanced_contrast_and_augmentation
    from effects import apply_balanced_dark_field_effect
    from utils import select_random_streamlines

# Import robust patch extraction functions
try:
    # Try to import from the main module directory
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from patch_extract import (
        PatchSpec, _hdr_dimensions_from_trk, _choose_random_origin,
        _crop_nifti, _filter_and_retarget_trk, _validate_consistency,
        _validate_outputs, extract_single_patch
    )
    ROBUST_PATCH_AVAILABLE = True
except ImportError:
    ROBUST_PATCH_AVAILABLE = False
    print("Warning: Robust patch extraction not available, falling back to legacy implementation")


def extract_random_patches_robust(nifti_file: str, 
                                 trk_files: List[str], 
                                 output_dir: str,
                                 total_patches: int = 100,
                                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                                 min_streamlines_per_patch: int = 30,
                                 random_state: Optional[int] = None,
                                 prefix: str = "patch",
                                 save_masks: bool = True,
                                 contrast_method: str = 'clahe',
                                 background_enhancement: str = 'preserve_edges',
                                 cornucopia_preset: str = 'disabled',
                                 tract_linewidth: float = 1.0,
                                 mask_thickness: int = 1,
                                 density_threshold: float = 0.15,
                                 gaussian_sigma: float = 2.0,
                                 close_gaps: bool = False,
                                 closing_footprint_size: int = 5,
                                 label_bundles: bool = False,
                                 min_bundle_size: int = 20,
                                 max_trials: int = 100,
                                 enable_orange_blobs: bool = False,
                                 orange_blob_probability: float = 0.3) -> Dict:
    """
    Extract random patches using robust methodology with proper coordinate transformations.
    
    This function uses the improved patch extraction approach that ensures:
    - Dimensional consistency validation
    - Proper coordinate transformations
    - Spatial alignment verification
    - Rejection sampling for adequate streamline density
    
    Args:
        nifti_file: Path to the NIfTI file
        trk_files: List of paths to TRK files
        output_dir: Directory to save patches
        total_patches: Total number of patches to extract
        patch_size: Size of each patch (width, height, depth)
        min_streamlines_per_patch: Minimum streamlines required in a patch
        random_state: Random seed for reproducibility
        prefix: Prefix for output files
        save_masks: Whether to save fiber masks
        ... (visualization parameters as before)
        max_trials: Maximum trials per patch to find adequate streamlines
        
    Returns:
        Dictionary with extraction results and metadata
    """
    
    if not ROBUST_PATCH_AVAILABLE:
        print("Warning: Falling back to legacy patch extraction")
        return extract_random_patches_legacy(
            nifti_file, trk_files, output_dir, total_patches,
            (patch_size[0], patch_size[1]), min_streamlines_per_patch,
            random_state, prefix, save_masks, contrast_method,
            background_enhancement, cornucopia_preset, tract_linewidth,
            mask_thickness, density_threshold, gaussian_sigma,
            close_gaps, closing_footprint_size, label_bundles, min_bundle_size
        )
    
    print(f"=== Robust Random Patch Extraction ===")
    print(f"NIfTI file: {nifti_file}")
    print(f"TRK files: {len(trk_files)}")
    print(f"Total patches: {total_patches}")
    print(f"Patch size: {patch_size[0]}x{patch_size[1]}x{patch_size[2]}")
    print(f"Min streamlines per patch: {min_streamlines_per_patch}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up random state
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    rng = np.random.default_rng(random_state)
    
    # Results tracking
    results = {
        'total_patches_requested': total_patches,
        'patches_extracted': 0,
        'patches_failed': 0,
        'patch_details': [],
        'trk_file_stats': [],
        'extraction_params': {
            'patch_size': patch_size,
            'min_streamlines_per_patch': min_streamlines_per_patch,
            'random_state': random_state,
            'max_trials': max_trials,
            'method': 'robust'
        }
    }
    
    # Process each TRK file to distribute patches
    patches_per_file = max(1, total_patches // len(trk_files))
    remaining_patches = total_patches - (patches_per_file * len(trk_files))
    
    print(f"\nPatch distribution across {len(trk_files)} TRK files:")
    for i, trk_file in enumerate(trk_files):
        extra_patch = 1 if i < remaining_patches else 0
        patches_for_file = patches_per_file + extra_patch
        print(f"  {Path(trk_file).name}: {patches_for_file} patches")
    
    patch_counter = 0
    
    for file_idx, trk_file in enumerate(trk_files):
        extra_patch = 1 if file_idx < remaining_patches else 0
        patches_for_file = patches_per_file + extra_patch
        
        print(f"\nProcessing {Path(trk_file).name} ({patches_for_file} patches)...")
        
        file_results = {
            'file': trk_file,
            'patches_requested': patches_for_file,
            'patches_extracted': 0,
            'patches_failed': 0
        }
        
        for patch_idx in range(patches_for_file):
            patch_counter += 1
            patch_seed = rng.integers(0, 2**32-1) if random_state is not None else None
            
            # Use individual TRK file for this patch
            patch_prefix = f"{prefix}_{patch_counter:04d}"
            patch_output_prefix = str(output_path / patch_prefix)
            
            print(f"  Extracting patch {patch_counter}/{total_patches}...")
            
            try:
                # Use robust single patch extraction
                meta = extract_single_patch(
                    nifti_path=nifti_file,
                    trk_path=trk_file,
                    patch_xyz=patch_size,
                    seed=patch_seed,
                    out_prefix=patch_output_prefix,
                    min_streamlines=min_streamlines_per_patch,
                    max_trials=max_trials
                )
                
                # Generate visualization if files were created successfully
                if save_masks or True:  # Always generate basic visualization
                    _generate_patch_visualization(
                        f"{patch_output_prefix}.nii.gz",
                        f"{patch_output_prefix}.trk",
                        str(output_path),
                        patch_prefix,
                        save_masks,
                        contrast_method,
                        background_enhancement,
                        cornucopia_preset,
                        tract_linewidth,
                        mask_thickness,
                        density_threshold,
                        gaussian_sigma,
                        close_gaps,
                        closing_footprint_size,
                        label_bundles,
                        min_bundle_size,
                        enable_orange_blobs,
                        orange_blob_probability
                    )
                
                patch_detail = {
                    'patch_id': patch_counter,
                    'center_voxel': meta['patch_origin'],
                    'size': meta['patch_size'],
                    'streamlines_count': meta['validations']['streamlines_kept'],
                    'trials': meta['validations']['trials'],
                    'nifti_file': f"{patch_output_prefix}.nii.gz",
                    'trk_file': f"{patch_output_prefix}.trk",
                    'meta_file': f"{patch_output_prefix}.meta.json",
                    'source_file': trk_file
                }
                
                results['patch_details'].append(patch_detail)
                results['patches_extracted'] += 1
                file_results['patches_extracted'] += 1
                
                print(f"  Patch {patch_counter}: ✓ ({meta['validations']['streamlines_kept']} streamlines, "
                      f"{meta['validations']['trials']} trials)")
                
            except Exception as e:
                print(f"  Patch {patch_counter}: ✗ Error - {e}")
                results['patches_failed'] += 1
                file_results['patches_failed'] += 1
        
        results['trk_file_stats'].append(file_results)
    
    # Save results summary
    summary_path = output_path / "patch_extraction_summary.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def _convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: _convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert results to JSON-safe format
    json_safe_results = _convert_numpy_types(results)
    
    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    print(f"\n=== Robust Patch Extraction Complete ===")
    print(f"Total patches extracted: {results['patches_extracted']}/{total_patches}")
    print(f"Failed extractions: {results['patches_failed']}")
    print(f"Summary saved: {summary_path}")
    
    return results


def extract_random_patches_legacy(nifti_file: str, 
                                 trk_files: List[str], 
                                 output_dir: str,
                                 total_patches: int = 100,
                                 patch_size: Tuple[int, int] = (1024, 1024),
                                 min_streamlines_per_patch: int = 50,
                                 random_state: Optional[int] = None,
                                 prefix: str = "patch",
                                 save_masks: bool = True,
                                 contrast_method: str = 'clahe',
                                 background_enhancement: str = 'preserve_edges',
                                 cornucopia_preset: str = 'disabled',
                                 tract_linewidth: float = 1.0,
                                 mask_thickness: int = 1,
                                 density_threshold: float = 0.15,
                                 gaussian_sigma: float = 2.0,
                                 close_gaps: bool = False,
                                 closing_footprint_size: int = 5,
                                 label_bundles: bool = False,
                                 min_bundle_size: int = 20) -> Dict:
    """
    Legacy patch extraction implementation (2D patches).
    
    This is the original patch extraction method kept for backward compatibility.
    It extracts 2D patches from coronal sections.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    print(f"=== Random Patch Extraction ===")
    print(f"NIfTI file: {nifti_file}")
    print(f"TRK files: {len(trk_files)}")
    print(f"Total patches: {total_patches}")
    print(f"Patch size: {patch_size[0]}x{patch_size[1]}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load NIfTI data
    print(f"\nLoading NIfTI data...")
    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()
    nifti_shape = nifti_data.shape
    print(f"NIfTI shape: {nifti_shape}")
    
    # Load and analyze TRK files
    print(f"\nLoading TRK files...")
    trk_data = []
    total_streamlines = 0
    
    for i, trk_file in enumerate(trk_files):
        print(f"  Loading {trk_file}...")
        trk_obj = load(trk_file)
        streamlines = trk_obj.streamlines
        
        # Analyze streamline distribution
        y_coords = []
        for sl in streamlines:
            y_coords.extend(sl[:, 1])  # Y coordinate (coronal axis)
        
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        y_center = (y_min + y_max) / 2
        
        trk_info = {
            'file': trk_file,
            'streamlines': streamlines,
            'count': len(streamlines),
            'y_range': (y_min, y_max),
            'y_center': y_center,
            'trk_obj': trk_obj
        }
        
        trk_data.append(trk_info)
        total_streamlines += len(streamlines)
        print(f"    Streamlines: {len(streamlines)}, Y range: {y_min:.1f} - {y_max:.1f}")
    
    print(f"Total streamlines across all files: {total_streamlines}")
    
    # Calculate patches per TRK file
    patches_per_file = max(1, total_patches // len(trk_files))
    remaining_patches = total_patches - (patches_per_file * len(trk_files))
    
    print(f"\nPatch distribution:")
    for i, trk_info in enumerate(trk_data):
        extra_patch = 1 if i < remaining_patches else 0
        patches_for_file = patches_per_file + extra_patch
        print(f"  {Path(trk_info['file']).name}: {patches_for_file} patches")
    
    # Extract patches
    results = {
        'total_patches_requested': total_patches,
        'patches_extracted': 0,
        'patches_failed': 0,
        'patch_details': [],
        'trk_file_stats': [],
        'extraction_params': {
            'patch_size': patch_size,
            'min_streamlines_per_patch': min_streamlines_per_patch,
            'random_state': random_state,
            'nifti_shape': nifti_shape
        }
    }
    
    patch_counter = 0
    
    for file_idx, trk_info in enumerate(trk_data):
        trk_file = trk_info['file']
        streamlines = trk_info['streamlines']
        y_min, y_max = trk_info['y_range']
        
        # Calculate patches for this file
        extra_patch = 1 if file_idx < remaining_patches else 0
        patches_for_file = patches_per_file + extra_patch
        
        print(f"\nProcessing {Path(trk_file).name} ({patches_for_file} patches)...")
        
        file_results = {
            'file': trk_file,
            'patches_requested': patches_for_file,
            'patches_extracted': 0,
            'patches_failed': 0
        }
        
        for patch_idx in range(patches_for_file):
            patch_counter += 1
            
            try:
                # Check if patch size is compatible with NIfTI dimensions
                if patch_size[0] > nifti_shape[0] or patch_size[1] > nifti_shape[2]:
                    print(f"  Patch {patch_counter}: Skipped (patch size {patch_size} too large for NIfTI {nifti_shape})")
                    results['patches_failed'] += 1
                    file_results['patches_failed'] += 1
                    continue
                
                # Generate random center point (ensure patch fits within NIfTI bounds)
                half_patch_x = patch_size[0] // 2
                half_patch_z = patch_size[1] // 2
                
                # Calculate valid ranges for center points
                min_center_x = half_patch_x
                max_center_x = nifti_shape[0] - half_patch_x
                min_center_z = half_patch_z  
                max_center_z = nifti_shape[2] - half_patch_z
                
                if min_center_x >= max_center_x or min_center_z >= max_center_z:
                    print(f"  Patch {patch_counter}: Skipped (no valid center positions for patch size {patch_size})")
                    results['patches_failed'] += 1
                    file_results['patches_failed'] += 1
                    continue
                
                center_x = np.random.randint(min_center_x, max_center_x)
                # Focus Y sampling on regions where streamlines actually exist
                # Use a weighted sampling that favors dense regions
                center_y = np.random.uniform(y_min, y_max)  # Random Y in streamline range
                center_z = np.random.randint(min_center_z, max_center_z)
                
                # Calculate patch bounds
                x_start = max(0, center_x - half_patch_x)
                x_end = min(nifti_shape[0], center_x + half_patch_x)
                y_start = max(0, int(center_y - half_patch_x))  # Using consistent half_patch_x for Y
                y_end = min(nifti_shape[1], int(center_y + half_patch_x))
                z_start = max(0, center_z - half_patch_z)
                z_end = min(nifti_shape[2], center_z + half_patch_z)
                
                # Extract patch data
                patch_data = nifti_data[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Filter streamlines for this patch region
                patch_streamlines = _filter_streamlines_for_patch(
                    streamlines, x_start, x_end, y_start, y_end, z_start, z_end
                )
                
                if len(patch_streamlines) < min_streamlines_per_patch:
                    print(f"  Patch {patch_counter}: Skipped (only {len(patch_streamlines)} streamlines, need {min_streamlines_per_patch})")
                    results['patches_failed'] += 1
                    file_results['patches_failed'] += 1
                    continue
                
                # Save patch files directly in output directory (no subdirectories)
                patch_filename_base = f"{prefix}_{patch_counter:04d}"
                
                # Save patch NIfTI
                patch_nifti_path = output_path / f"{patch_filename_base}.nii.gz"
                
                # Create proper affine for the patch
                # The patch affine should map patch voxels to the same world coordinates as the original
                patch_affine = nifti_img.affine.copy()
                
                # Adjust the translation part of the affine to account for the patch offset
                # This ensures that patch voxel (0,0,0) maps to world coordinate of the patch start
                patch_offset_world = np.array([x_start, y_start, z_start, 1])
                world_offset = nifti_img.affine @ patch_offset_world
                patch_affine[:3, 3] = world_offset[:3]
                
                patch_nifti = nib.Nifti1Image(patch_data, patch_affine, nifti_img.header)
                nib.save(patch_nifti, patch_nifti_path)
                
                # Create TRK file for this patch
                patch_trk_path = output_path / f"{patch_filename_base}_streamlines.trk"
                try:
                    _save_patch_streamlines(patch_streamlines, trk_info['trk_obj'], patch_trk_path, 
                                          x_start, y_start, z_start)
                    
                    # Generate visualization PNG
                    patch_viz_path = output_path / f"{patch_filename_base}_visualization.png"
                    _generate_patch_visualization(
                        str(patch_nifti_path), str(patch_trk_path), str(output_path),
                        patch_filename_base, save_masks, contrast_method,
                        background_enhancement, cornucopia_preset, tract_linewidth,
                        mask_thickness, density_threshold, gaussian_sigma,
                        close_gaps, closing_footprint_size, label_bundles, min_bundle_size,
                        False, 0.3  # Default orange blob parameters for legacy function
                    )
                except Exception as e:
                    print(f"  Patch {patch_counter}: TRK generation failed ({e}), saving NIfTI only")
                    # Remove the TRK file if it was partially created
                    if patch_trk_path.exists():
                        patch_trk_path.unlink()
                
                patch_detail = {
                    'patch_id': patch_counter,
                    'center': (center_x, center_y, center_z),
                    'bounds': (x_start, x_end, y_start, y_end, z_start, z_end),
                    'streamlines_count': len(patch_streamlines),
                    'nifti_file': str(patch_nifti_path),
                    'trk_file': str(patch_trk_path) if patch_trk_path.exists() else None,
                    'source_file': trk_file
                }
                
                results['patch_details'].append(patch_detail)
                results['patches_extracted'] += 1
                file_results['patches_extracted'] += 1
                
                print(f"  Patch {patch_counter}: ✓ ({len(patch_streamlines)} streamlines)")
                
            except Exception as e:
                print(f"  Patch {patch_counter}: ✗ Error - {e}")
                results['patches_failed'] += 1
                file_results['patches_failed'] += 1
        
        results['trk_file_stats'].append(file_results)
    
    # Save results summary
    summary_path = output_path / "patch_extraction_summary.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def _convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: _convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert results to JSON-safe format
    json_safe_results = _convert_numpy_types(results)
    
    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    print(f"\n=== Patch Extraction Complete ===")
    print(f"Total patches extracted: {results['patches_extracted']}/{total_patches}")
    print(f"Failed extractions: {results['patches_failed']}")
    print(f"Summary saved: {summary_path}")
    
    return results


def extract_random_patches(nifti_file: str, 
                          trk_files: List[str], 
                          output_dir: str,
                          total_patches: int = 100,
                          patch_size: Tuple[int, int] = (1024, 1024),
                          min_streamlines_per_patch: int = 50,
                          random_state: Optional[int] = None,
                          prefix: str = "patch",
                          save_masks: bool = True,
                          contrast_method: str = 'clahe',
                          background_enhancement: str = 'preserve_edges',
                          cornucopia_preset: str = 'disabled',
                          tract_linewidth: float = 1.0,
                          mask_thickness: int = 1,
                          density_threshold: float = 0.15,
                          gaussian_sigma: float = 2.0,
                          close_gaps: bool = False,
                          closing_footprint_size: int = 5,
                          label_bundles: bool = False,
                          min_bundle_size: int = 20,
                          enable_orange_blobs: bool = False,
                          orange_blob_probability: float = 0.3) -> Dict:
    """
    Main patch extraction interface - automatically chooses best method.
    
    This function maintains backward compatibility while using the improved
    robust patch extraction when available. For 3D patches, uses the robust
    method. For 2D patches, falls back to legacy method.
    
    Args:
        nifti_file: Path to the NIfTI file
        trk_files: List of paths to TRK files
        output_dir: Directory to save patches
        total_patches: Total number of patches to extract
        patch_size: Size of each patch (width, height) for 2D or (w, h, d) for 3D
        ... (other parameters as documented in individual methods)
        
    Returns:
        Dictionary with extraction results and metadata
    """
    
    # Check if 3D patch size provided
    if len(patch_size) == 3 and ROBUST_PATCH_AVAILABLE:
        # Use robust 3D extraction
        return extract_random_patches_robust(
            nifti_file, trk_files, output_dir, total_patches,
            patch_size, min_streamlines_per_patch, random_state, prefix,
            save_masks, contrast_method, background_enhancement, cornucopia_preset,
            tract_linewidth, mask_thickness, density_threshold, gaussian_sigma,
            close_gaps, closing_footprint_size, label_bundles, min_bundle_size,
            max_trials=100, enable_orange_blobs=enable_orange_blobs, 
            orange_blob_probability=orange_blob_probability
        )
    elif len(patch_size) == 2:
        # Use legacy 2D extraction (no orange blob support for legacy)
        return extract_random_patches_legacy(
            nifti_file, trk_files, output_dir, total_patches,
            patch_size, min_streamlines_per_patch, random_state, prefix,
            save_masks, contrast_method, background_enhancement, cornucopia_preset,
            tract_linewidth, mask_thickness, density_threshold, gaussian_sigma,
            close_gaps, closing_footprint_size, label_bundles, min_bundle_size
        )
    else:
        # Try to convert 2D to 3D for robust extraction
        if ROBUST_PATCH_AVAILABLE and len(patch_size) == 2:
            patch_3d = (patch_size[0], patch_size[1], patch_size[0])
            print(f"Converting 2D patch size {patch_size} to 3D {patch_3d} for robust extraction")
            return extract_random_patches_robust(
                nifti_file, trk_files, output_dir, total_patches,
                patch_3d, min_streamlines_per_patch, random_state, prefix,
                save_masks, contrast_method, background_enhancement, cornucopia_preset,
                tract_linewidth, mask_thickness, density_threshold, gaussian_sigma,
                close_gaps, closing_footprint_size, label_bundles, min_bundle_size
            )
        else:
            # Fallback to legacy
            return extract_random_patches_legacy(
                nifti_file, trk_files, output_dir, total_patches,
                tuple(patch_size[:2]), min_streamlines_per_patch, random_state, prefix,
                save_masks, contrast_method, background_enhancement, cornucopia_preset,
                tract_linewidth, mask_thickness, density_threshold, gaussian_sigma,
                close_gaps, closing_footprint_size, label_bundles, min_bundle_size
            )


def _filter_streamlines_for_patch(streamlines, x_start, x_end, y_start, y_end, z_start, z_end):
    """Filter streamlines that actually pass through the patch region with strict bounds checking."""
    patch_streamlines = []
    
    for sl in streamlines:
        if len(sl) < 2:
            continue
            
        # Get coordinates
        x_coords = sl[:, 0]
        y_coords = sl[:, 1]
        z_coords = sl[:, 2]
        
        # Strict approach: check if streamline has at least one point WITHIN the patch bounds
        # No margin - must be actually inside the patch region
        within_bounds = (
            (x_coords >= x_start) & (x_coords < x_end) &
            (y_coords >= y_start) & (y_coords < y_end) &
            (z_coords >= z_start) & (z_coords < z_end)
        )
        
        # Accept only if streamline has at least one point strictly within bounds
        if np.any(within_bounds):
            # Keep only the portion of streamline that's within or close to the patch
            # Create segments that are relevant to this patch
            extended_bounds = (
                (x_coords >= x_start - 5) & (x_coords <= x_end + 5) &
                (y_coords >= y_start - 5) & (y_coords <= y_end + 5) &
                (z_coords >= z_start - 5) & (z_coords <= z_end + 5)
            )
            
            if np.any(extended_bounds):
                # Keep the relevant portion of the streamline
                relevant_indices = np.where(extended_bounds)[0]
                start_idx = max(0, relevant_indices[0] - 2)  # Include a bit before
                end_idx = min(len(sl), relevant_indices[-1] + 3)  # Include a bit after
                patch_streamlines.append(sl[start_idx:end_idx])
    
    return patch_streamlines


def _save_patch_streamlines(streamlines, original_trk_obj, output_path, x_offset, y_offset, z_offset):
    """Save streamlines for a patch with proper coordinate transformation."""
    from nibabel.streamlines import Tractogram, save as save_trk
    
    # Ensure output_path is a string
    output_path_str = str(output_path)
    
    # The key insight: streamlines should remain in WORLD coordinates, not patch-local coordinates
    # The patch NIfTI will have an adjusted affine, and the visualization function will handle 
    # the coordinate transformation properly
    
    # Keep streamlines in their original world coordinate system
    # Do NOT subtract the patch offsets - let the affine handle the transformation
    world_streamlines = []
    for sl in streamlines:
        # Keep streamlines as-is in world coordinates
        # The patch affine will handle the coordinate transformation during visualization
        world_streamlines.append(sl.copy())
    
    # Convert to float32 for TRK compatibility
    clipped_streamlines = [s.astype(np.float32) for s in world_streamlines if len(s) >= 2]
    
    if not clipped_streamlines:
        print(f"    Warning: No valid streamlines after filtering")
        return  # No streamlines to save
    
    # Use the original tractogram's affine - this ensures coordinate consistency
    # The streamlines are in world coordinates, and the affine will transform them correctly
    slice_tractogram = Tractogram(clipped_streamlines, affine_to_rasmm=original_trk_obj.affine)
    
    # Save TRK file
    try:
        save_trk(slice_tractogram, output_path_str)
        print(f"    Saved {len(clipped_streamlines)} streamlines to {output_path_str}")
    except Exception as e:
        print(f"    Error saving streamlines to {output_path_str}: {e}")


def _generate_patch_visualization(nifti_path, trk_path, output_dir, prefix, save_masks,
                                 contrast_method, background_enhancement, cornucopia_preset,
                                 tract_linewidth, mask_thickness, density_threshold,
                                 gaussian_sigma, close_gaps, closing_footprint_size,
                                 label_bundles, min_bundle_size, enable_orange_blobs,
                                 orange_blob_probability=0.3):
    """Generate visualization for a single patch."""
    
    # Randomize cornucopia preset for variation unless explicitly disabled
    import random
    import time
    
    if cornucopia_preset == 'disabled':
        # Use random presets to add variation
        available_presets = ['disabled', 'clean_optical', 'gamma_speckle', 'optical_with_debris', 'subtle_debris', 'clinical_simulation']
        random.seed(int(time.time() * 1000000) % (2**32))  # Truly random seed
        actual_preset = random.choice(available_presets)
    else:
        actual_preset = cornucopia_preset
    
    # Use coronal view for patches
    try:
        # First, create the standard visualization
        output_path = os.path.join(output_dir, f"{prefix}_visualization.png")
        visualize_nifti_with_trk_coronal(
            nifti_file=nifti_path,
            trk_file=trk_path,
            output_file=output_path,
            n_slices=1,
            slice_idx=0,  # Use middle slice
            streamline_percentage=100.0,
            tract_linewidth=tract_linewidth,
            save_masks=save_masks,
            mask_thickness=mask_thickness,
            density_threshold=density_threshold,
            gaussian_sigma=gaussian_sigma,
            close_gaps=close_gaps,
            closing_footprint_size=closing_footprint_size,
            label_bundles=label_bundles,
            min_bundle_size=min_bundle_size,
            contrast_method='equalize',  # Use global histogram equalization for smoother results
            background_enhancement=background_enhancement,
            cornucopia_augmentation=actual_preset,
            truly_random=True  # Enable truly random parameters
        )
        
        # Now add orange injection sites if enabled
        if enable_orange_blobs:
            import random
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            from PIL import Image
            import numpy as np
            
            print(f"🔥 ADDING ORANGE INJECTION SITE to patch visualization")
            
            if random.random() < orange_blob_probability:
                # Load the saved visualization
                if os.path.exists(output_path):
                    # Load image using PIL
                    img = Image.open(output_path)
                    img_array = np.array(img)
                    
                    # Create a new matplotlib figure
                    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)
                    ax.imshow(img_array)
                    ax.set_xlim(0, img_array.shape[1])
                    ax.set_ylim(img_array.shape[0], 0)  # Flip Y axis to match image coordinates
                    ax.axis('off')
                    
                    # Get image dimensions  
                    height, width = img_array.shape[0], img_array.shape[1]
                    
                    # Create dense orange injection site at random location
                    margin = int(min(width, height) * 0.1)  # Keep some margin from edges
                    center_x = np.random.randint(margin, width - margin)
                    center_y = np.random.randint(margin, height - margin)
                    injection_radius = min(width, height) * 0.05  # Slightly larger injection area
                    
                    print(f"🧡 Adding dense orange injection site at ({center_x:.0f}, {center_y:.0f}), radius: {injection_radius:.0f}")
                    
                    # Add fewer orange streamlines for smaller area
                    num_orange_streamlines = 400  # More streamlines for better visibility
                    for i in range(num_orange_streamlines):
                        # Random start point within injection area
                        angle = np.random.uniform(0, 2*np.pi)
                        radius = np.random.uniform(0, injection_radius)
                        start_x = center_x + radius * np.cos(angle)
                        start_y = center_y + radius * np.sin(angle)
                        
                        # Generate curved orange streamline - much shorter for small area
                        streamline_length = np.random.randint(20, 30)  # Slightly longer streamlines
                        x_coords = [start_x]
                        y_coords = [start_y]
                        
                        # Direction radiating outward with some randomness
                        direction_x = np.cos(angle) + np.random.normal(0, 0.3)
                        direction_y = np.sin(angle) + np.random.normal(0, 0.3)
                        
                        current_x, current_y = start_x, start_y
                        
                        # Add curvature parameters
                        curve_amount = np.random.uniform(0.1, 0.4)  # How much to curve
                        curve_frequency = np.random.uniform(0.05, 0.15)  # How often to change direction
                        
                        for step in range(streamline_length):
                            # Add progressive curvature and noise for natural fiber appearance
                            step_size = np.random.uniform(0.5, 1.0)  # Much smaller steps
                            
                            # Add smooth curvature
                            curve_offset_x = curve_amount * np.sin(step * curve_frequency) * np.random.uniform(0.5, 1.5)
                            curve_offset_y = curve_amount * np.cos(step * curve_frequency) * np.random.uniform(0.5, 1.5)
                            
                            # Add random noise for natural variation
                            noise_x = np.random.normal(0, 0.3)
                            noise_y = np.random.normal(0, 0.3)
                            
                            # Update direction with curvature and noise
                            direction_x += (curve_offset_x + noise_x) * 0.1
                            direction_y += (curve_offset_y + noise_y) * 0.1
                            
                            # Normalize to prevent runaway
                            direction_length = np.sqrt(direction_x**2 + direction_y**2)
                            if direction_length > 0:
                                direction_x /= direction_length
                                direction_y /= direction_length
                            
                            current_x += direction_x * step_size
                            current_y += direction_y * step_size
                            
                            if (current_x < 0 or current_x >= width or 
                                current_y < 0 or current_y >= height):
                                break
                                
                            x_coords.append(current_x)
                            y_coords.append(current_y)
                        
                        # Plot orange streamline with natural appearance
                        if len(x_coords) > 5:
                            # Use more natural orange colors with lower brightness
                            orange_colors = ['#CC5500', '#BB4400', '#DD6600', '#AA3300', '#EE7700']
                            color = np.random.choice(orange_colors)
                            ax.plot(x_coords, y_coords, color=color, linewidth=1.5, 
                                   alpha=0.6, solid_capstyle='round', zorder=25)
                    
                    # Add subtle orange center marker
                    ax.scatter([center_x], [center_y], c='#CC5500', s=80, alpha=0.7, zorder=30, marker='o')  # Subtle orange center
                    ax.scatter([center_x], [center_y], c='#AA3300', s=25, alpha=0.8, zorder=31, marker='o')   # Darker orange center
                    
                    # Save the modified image
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    print(f"🧡 Added {num_orange_streamlines} orange streamlines to patch visualization")
                    print(f"🧡 Orange injection site successfully added!")
                else:
                    print(f"⚠️ Visualization file not found: {output_path}")
            else:
                print(f"🔥 Orange injection site skipped due to probability ({orange_blob_probability})")
            
        
    except Exception as e:
        print(f"    Warning: Visualization failed for {prefix}: {e}")
def main():
    """Command line interface for patch extraction."""
    parser = argparse.ArgumentParser(
        description="Extract random patches from NIfTI volume with tractography data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument("--nifti", required=True, help="Path to NIfTI file")
    parser.add_argument("--trk_files", nargs='+', required=True, help="Paths to TRK files")
    parser.add_argument("--output_dir", required=True, help="Output directory for patches")
    
    # Patch parameters
    parser.add_argument("--total_patches", type=int, default=100, help="Total number of patches to extract")
    parser.add_argument("--patch_size", type=int, nargs='+', default=[1024, 15, 1024], help="Patch size (width height depth) - use 3 values for 3D patches, 2 for 2D")
    parser.add_argument("--min_streamlines", type=int, default=5, help="Minimum streamlines per patch")
    parser.add_argument("--random_state", type=int, help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument("--prefix", default="patch", help="Prefix for output files")
    parser.add_argument("--save_masks", action="store_true", help="Save fiber masks")
    
    # Visualization parameters
    parser.add_argument("--contrast_method", default="clahe", choices=["none", "clahe", "equalize"],
                       help="Contrast enhancement method")
    parser.add_argument("--background_enhancement", default="preserve_edges",
                       choices=["none", "preserve_edges", "enhance_contrast"],
                       help="Background enhancement preset")
    parser.add_argument("--cornucopia_preset", default="disabled",
                       choices=["disabled", "low", "medium", "high"],
                       help="Cornucopia augmentation preset")
    parser.add_argument("--tract_linewidth", type=float, default=1.0, help="Tract line width")
    parser.add_argument("--mask_thickness", type=int, default=1, help="Mask line thickness")
    parser.add_argument("--density_threshold", type=float, default=0.15, help="Density threshold")
    parser.add_argument("--gaussian_sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--close_gaps", action="store_true", help="Close gaps in masks")
    parser.add_argument("--closing_footprint_size", type=int, default=5, help="Gap closing footprint size")
    parser.add_argument("--label_bundles", action="store_true", help="Label distinct bundles")
    parser.add_argument("--min_bundle_size", type=int, default=20, help="Minimum bundle size")
    
    args = parser.parse_args()
    
    # Run patch extraction
    results = extract_random_patches(
        nifti_file=args.nifti,
        trk_files=args.trk_files,
        output_dir=args.output_dir,
        total_patches=args.total_patches,
        patch_size=tuple(args.patch_size),
        min_streamlines_per_patch=args.min_streamlines,
        random_state=args.random_state,
        prefix=args.prefix,
        save_masks=args.save_masks,
        contrast_method=args.contrast_method,
        background_enhancement=args.background_enhancement,
        cornucopia_preset=args.cornucopia_preset,
        tract_linewidth=args.tract_linewidth,
        mask_thickness=args.mask_thickness,
        density_threshold=args.density_threshold,
        gaussian_sigma=args.gaussian_sigma,
        close_gaps=args.close_gaps,
        closing_footprint_size=args.closing_footprint_size,
        label_bundles=args.label_bundles,
        min_bundle_size=args.min_bundle_size
    )
    
    print(f"\nPatch extraction completed successfully!")
    print(f"Results: {results['patches_extracted']}/{args.total_patches} patches extracted")


if __name__ == "__main__":
    main()
