#!/usr/bin/env python
"""
Patch Extraction Module for Syntract Viewer

This module provides functionality to extract random patches from NIfTI volumes
with overlaid tractography data. It supports:
- Specified total number of patches
- Fixed patch dimensions
- Distribution across multiple TRK files
- Random sampling from different coronal sections
- Efficient center-point based extraction
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.streamlines import load
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


def extract_random_patches(nifti_file: str, 
                          trk_files: List[str], 
                          output_dir: str,
                          total_patches: int = 100,
                          patch_size: Tuple[int, int] = (1024, 1024),
                          min_streamlines_per_patch: int = 50,  # Increased default from 5 to 50
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
    Extract random patches from NIfTI volume with tractography data.
    
    Args:
        nifti_file: Path to the NIfTI file
        trk_files: List of paths to TRK files
        output_dir: Directory to save patches
        total_patches: Total number of patches to extract
        patch_size: Size of each patch (width, height)
        min_streamlines_per_patch: Minimum streamlines required in a patch
        random_state: Random seed for reproducibility
        prefix: Prefix for output files
        save_masks: Whether to save fiber masks
        contrast_method: Contrast enhancement method
        background_enhancement: Background enhancement preset
        cornucopia_preset: Cornucopia augmentation preset
        tract_linewidth: Line width for tract visualization
        mask_thickness: Thickness of mask lines
        density_threshold: Density threshold for mask creation
        gaussian_sigma: Gaussian smoothing sigma
        close_gaps: Whether to close gaps in masks
        closing_footprint_size: Footprint size for gap closing
        label_bundles: Whether to label distinct bundles
        min_bundle_size: Minimum bundle size for labeling
        
    Returns:
        Dictionary with extraction results and metadata
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
                        close_gaps, closing_footprint_size, label_bundles, min_bundle_size
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
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Patch Extraction Complete ===")
    print(f"Total patches extracted: {results['patches_extracted']}/{total_patches}")
    print(f"Failed extractions: {results['patches_failed']}")
    print(f"Summary saved: {summary_path}")
    
    return results


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
                                 label_bundles, min_bundle_size):
    """Generate visualization for a single patch."""
    
    # Use coronal view for patches
    try:
        visualize_nifti_with_trk_coronal(
            nifti_file=nifti_path,
            trk_file=trk_path,
            output_file=os.path.join(output_dir, f"{prefix}_visualization.png"),
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
            contrast_method=contrast_method,
            background_enhancement=background_enhancement,
            cornucopia_augmentation=cornucopia_preset
        )
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
    parser.add_argument("--patch_size", type=int, nargs=2, default=[1024, 1024], help="Patch size (width height)")
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
