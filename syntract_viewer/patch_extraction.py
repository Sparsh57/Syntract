#!/usr/bin/env python
"""
Simplified Patch Extraction Module for Syntract Viewer

This module provides functionality to extract random patches from NIfTI volumes
with overlaid tractography data and generate visualizations.

Features:
- Uses robust patch extraction from patch_extract.py
- Integrated visualization generation with orange blob support
- Multiple TRK file support
"""
import os
import sys
import json
import random
import time
import argparse
import gc
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

try:
    from .core import visualize_nifti_with_trk_coronal
except ImportError:
    from core import visualize_nifti_with_trk_coronal

# Import robust patch extraction functions
try:
    # Import from the main module directory
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from patch_extract import extract_single_patch, extract_multiple_patches
    ROBUST_PATCH_AVAILABLE = True
except ImportError:
    ROBUST_PATCH_AVAILABLE = False
    print("Error: Robust patch extraction not available")


def extract_random_patches(nifti_file: str, 
                          trk_files: List[str], 
                          output_dir: str,
                          total_patches: int = 100,
                          patch_size: Tuple[int, int] = (1024, 1024),
                          min_streamlines_per_patch: int = 50,
                          random_state: Optional[int] = None,
                          prefix: str = "patch",
                          batch_size: int = 50,
                          save_masks: bool = True,
                          contrast_method: str = 'clahe',
                          background_enhancement: str = 'preserve_edges',
                          cornucopia_preset: str = 'clean_optical',
                          tract_linewidth: float = 1.0,
                          mask_thickness: int = 1,
                          density_threshold: float = 0.15,
                          gaussian_sigma: float = 2.0,
                          close_gaps: bool = False,
                          closing_footprint_size: int = 5,
                          label_bundles: bool = False,
                          min_bundle_size: int = 20,
                          enable_orange_blobs: bool = False,
                          orange_blob_probability: float = 0.3,
                          **kwargs) -> List[Dict]:
    """
    Extract random patches using robust methodology with memory-efficient batch processing.
    
    This function uses the robust patch extraction from patch_extract.py and adds
    visualization capabilities for the syntract viewer pipeline. Optimized for large
    volumes with memory-mapping and batch processing to prevent OOM errors.
    
    Args:
        nifti_file: Path to the NIfTI file
        trk_files: List of paths to TRK files
        output_dir: Directory to save patches
        total_patches: Total number of patches to extract
        patch_size: Size of each patch (width, height) for 2D or (w, h, d) for 3D
        min_streamlines_per_patch: Minimum streamlines required in a patch
        random_state: Random seed for reproducibility
        prefix: Prefix for output files
        batch_size: Number of patches per batch before memory cleanup (default: 50)
        save_masks: Whether to save fiber masks
        enable_orange_blobs: Whether to add orange injection sites
        orange_blob_probability: Probability of adding orange blobs
        ... (other visualization parameters)
        
    Returns:
        Dictionary with extraction results and metadata
    """
    
    if not ROBUST_PATCH_AVAILABLE:
        raise RuntimeError("Robust patch extraction not available. Cannot extract patches.")
    
    print(f"=== Robust Random Patch Extraction ===")
    print(f"NIfTI file: {nifti_file}")
    print(f"TRK files: {len(trk_files)}")
    print(f"Total patches: {total_patches}")
    
    # Determine if 3D patch extraction is requested
    is_3d_patch = len(patch_size) == 3
    if not is_3d_patch and len(patch_size) == 2:
        # Convert 2D to 3D with single slice depth
        patch_size = (patch_size[0], 1, patch_size[1])
        is_3d_patch = True
        print(f"Converting 2D patch size to 3D: {patch_size}")
    
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
                    max_trials=100
                )
                
                # Generate visualization if files were created successfully
                nifti_out = f"{patch_output_prefix}.nii.gz"
                trk_out = f"{patch_output_prefix}.trk"
                
                if os.path.exists(nifti_out) and os.path.exists(trk_out):
                    _generate_patch_visualization(
                        nifti_out, trk_out, str(output_path), patch_prefix, 
                        save_masks, contrast_method, background_enhancement, 
                        cornucopia_preset, tract_linewidth, mask_thickness,
                        density_threshold, gaussian_sigma, close_gaps, 
                        closing_footprint_size, label_bundles, min_bundle_size,
                        enable_orange_blobs, orange_blob_probability
                    )
                
                # CRITICAL: Force garbage collection after each visualization to prevent memory accumulation
                gc.collect()
                
                # Record successful extraction
                results['patches_extracted'] += 1
                file_results['patches_extracted'] += 1
                results['patch_details'].append({
                    'patch_id': patch_counter,
                    'source_trk': trk_file,
                    'streamlines_kept': meta['validations']['streamlines_kept'],
                    'trials': meta['validations']['trials'],
                    'files': {
                        'nifti': nifti_out,
                        'trk': trk_out,
                        'meta': f"{patch_output_prefix}.meta.json"
                    }
                })
                
                print(f"    Patch {patch_counter}: OK ({meta['validations']['streamlines_kept']} streamlines, {meta['validations']['trials']} trials)")
                
            except Exception as e:
                print(f"    Patch {patch_counter}: ERROR: Failed - {e}")
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


def _generate_patch_visualization(nifti_path, trk_path, output_dir, prefix, save_masks,
                                 contrast_method, background_enhancement, cornucopia_preset,
                                 tract_linewidth, mask_thickness, density_threshold,
                                 gaussian_sigma, close_gaps, closing_footprint_size,
                                 label_bundles, min_bundle_size, enable_orange_blobs,
                                 orange_blob_probability=0.3):
    """Generate visualization for a single patch."""
    
    # Randomize cornucopia preset for variation unless explicitly set to clean_optical
    if cornucopia_preset == 'clean_optical':
        # Weighted selection with increased heavy preset probability
        presets = ['clean_optical', 'gamma_speckle', 'optical_with_debris', 
                  'subtle_debris', 'clinical_simulation', 'heavy_speckle']
        # Weights: clean (30%), subtle (30%), moderate (20%), heavy (20%)
        weights = [0.05, 0.25, 0.25, 0.25, 0.10, 0.10]  # heavy_speckle gets 20%
        random.seed(int(time.time() * 1000000) % (2**32))  # Truly random seed
        actual_preset = random.choices(presets, weights=weights, k=1)[0]
    else:
        actual_preset = cornucopia_preset
    
    # Use coronal view for patches
    try:
        # First, create the standard visualization
        output_path = os.path.join(output_dir, f"{prefix}_visualization.png")
        result = visualize_nifti_with_trk_coronal(
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
            contrast_method=contrast_method,  # Use the passed contrast method parameter
            background_enhancement=background_enhancement,
            cornucopia_augmentation=actual_preset,
            truly_random=True  # Enable truly random parameters
        )
        # Now add orange injection sites if enabled
        if enable_orange_blobs:
            from PIL import Image
            import matplotlib.pyplot as plt
            
            print(f"ADDING ORANGE INJECTION SITE to patch visualization")
            
            if random.random() < orange_blob_probability:
                # Load the visualization image
                img = Image.open(output_path)
                img_array = np.array(img)
                
                # Generate random orange injection site parameters
                img_height, img_width = img_array.shape[:2]
                center_x = random.randint(img_width // 4, 3 * img_width // 4)
                center_y = random.randint(img_height // 4, 3 * img_height // 4)
                radius = random.randint(30, 80)
                
                print(f"Adding dense orange injection site at ({center_x}, {center_y}), radius: {radius}")
                
                # Create circular mask for injection site
                y_coords, x_coords = np.ogrid[:img_height, :img_width]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                
                # Apply orange color with varying intensity
                orange_color = np.array([255, 165, 0])  # Orange RGB
                if len(img_array.shape) == 3:
                    for i in range(3):
                        # Create gradient effect - stronger in center
                        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                        intensity = np.exp(-distance_from_center / (radius / 3))
                        img_array[mask, i] = np.clip(
                            img_array[mask, i] * (1 - intensity[mask] * 0.8) + 
                            orange_color[i] * intensity[mask] * 0.8, 0, 255
                        )
                
                # Add some random orange streamlines around the injection site
                num_orange_streamlines = random.randint(200, 500)
                for _ in range(num_orange_streamlines):
                    # Random streamline originating from injection site
                    start_x = center_x + random.randint(-radius//2, radius//2)
                    start_y = center_y + random.randint(-radius//2, radius//2)
                    
                    # Random direction and length
                    angle = random.uniform(0, 2 * np.pi)
                    length = random.randint(20, 60)
                    
                    end_x = int(start_x + length * np.cos(angle))
                    end_y = int(start_y + length * np.sin(angle))
                    
                    # Ensure endpoints are within image bounds
                    end_x = max(0, min(img_width - 1, end_x))
                    end_y = max(0, min(img_height - 1, end_y))
                    
                    # Draw line (simple implementation)
                    # Use Bresenham's line algorithm or simple interpolation
                    x_points = np.linspace(start_x, end_x, abs(end_x - start_x) + abs(end_y - start_y) + 1).astype(int)
                    y_points = np.linspace(start_y, end_y, abs(end_x - start_x) + abs(end_y - start_y) + 1).astype(int)
                    
                    # Filter points within bounds
                    valid_mask = (x_points >= 0) & (x_points < img_width) & (y_points >= 0) & (y_points < img_height)
                    x_points = x_points[valid_mask]
                    y_points = y_points[valid_mask]
                    
                    # Apply orange color to line pixels
                    if len(x_points) > 0 and len(img_array.shape) == 3:
                        for i in range(3):
                            img_array[y_points, x_points, i] = np.clip(
                                img_array[y_points, x_points, i] * 0.3 + orange_color[i] * 0.7, 0, 255
                            )
                
                print(f"Added {num_orange_streamlines} orange streamlines to patch visualization")
                
                # Save the modified image
                modified_img = Image.fromarray(img_array.astype(np.uint8))
                modified_img.save(output_path)
                
                print(f"Orange injection site successfully added!")
            else:
                print(f"Orange injection site skipped due to probability ({orange_blob_probability})")
        
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
    parser.add_argument("--cornucopia_preset", default="clean_optical",
                       choices=["clean_optical", "gamma_speckle", "optical_with_debris", 
                               "subtle_debris", "clinical_simulation", "heavy_speckle"],
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