#!/usr/bin/env python
"""
DEPRECATED: Patch Extraction Module for Syntract Viewer

This module is largely deprecated in favor of the integrated patch-first optimization 
in syntract.py. The visualization function _generate_patch_visualization() is still 
used by the main pipeline.

For new patch extraction, use:
  python syntract.py --input brain.nii.gz --trk fibers.trk --total_patches 100

Features:
- Patch visualization generation with orange blob support  
- Smart Cornucopia preset selection
- Multiple contrast enhancement options
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

# Note: This module no longer imports patch_extract due to deprecation
# All patch extraction should now use the patch-first optimization in syntract.py
ROBUST_PATCH_AVAILABLE = False
print("Notice: patch_extract module deprecated. Use syntract.py patch-first optimization instead.")


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
    DEPRECATED: Use syntract.py patch-first optimization instead.
    
    This function is deprecated. For patch extraction, use:
        python syntract.py --input brain.nii.gz --trk fibers.trk \\
            --total_patches 100 --patch_size 800 1 800
    
    The new patch-first method provides 80-95% performance improvements with better
    curvature preservation and zero-tolerance spatial accuracy.
    """
    
    raise DeprecationWarning(
        "extract_random_patches() is deprecated. "
        "Use the patch-first optimization in syntract.py instead:\n"
        "  python syntract.py --input brain.nii.gz --trk fibers.trk \\\n"
        "    --total_patches 100 --patch_size 800 1 800\n"
        "This provides 80-95% performance improvements with better accuracy."
    )


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
        weights = [0.05, 0.25, 0.35, 0.25, 0.10, 0.10]  # heavy_speckle gets 20%
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
    """DEPRECATED: Command line interface for patch extraction."""
    print("=" * 60)
    print("DEPRECATION WARNING:")
    print("This patch extraction module is deprecated.")
    print("Use the integrated patch-first optimization in syntract.py instead:")
    print("")
    print("  python syntract.py --input brain.nii.gz --trk fibers.trk \\")
    print("    --total_patches 100 --patch_size 800 1 800")
    print("")
    print("The new method provides 80-95% performance improvements")
    print("with better curvature preservation and spatial accuracy.")
    print("=" * 60)
    
    response = input("Continue with deprecated method anyway? [y/N]: ")
    if response.lower() != 'y':
        print("Exiting. Please use syntract.py for patch extraction.")
        return
    
    # Original argument parsing (kept for reference but will fail due to deprecated function)
    parser = argparse.ArgumentParser(
        description="DEPRECATED: Extract random patches from NIfTI volume with tractography data",
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