#!/usr/bin/env python
"""
Simple Batch Processing for MRI Synthesis

Process multiple TRK files with a common NIfTI file efficiently.
Automatically optimizes processing and provides clean output.

Usage:
    python cumulative.py --nifti brain.nii.gz --trk-dir /path/to/trk/files
    
    # Or from Python:
    from cumulative import process_batch
    results = process_batch('brain.nii.gz', '/path/to/trk/files')
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Import the syntract function
from syntract import process_syntract


def process_batch(nifti_file, trk_directory, output_dir="results", patches=30, 
                  patch_size=None, min_streamlines_per_patch=20, patch_prefix="patch",
                  use_ants=False, ants_warp=None, ants_iwarp=None, ants_aff=None,
                  voxel_size=0.05, new_dim=None, skip_synthesis=False,
                  n_examples=10, viz_prefix="synthetic_", enable_orange_blobs=False,
                  orange_blob_probability=0.3, save_masks=True, use_high_density_masks=True,
                  mask_thickness=1, density_threshold=0.15, min_bundle_size=20,
                  label_bundles=False, disable_patch_processing=False, cleanup_intermediate=True):
    """
    Process multiple TRK files with a common NIfTI file.
    
    This is the main function for batch processing - simple and efficient.
    
    Parameters:
    -----------
    nifti_file : str
        Path to the NIfTI file
    trk_directory : str
        Directory containing TRK files
    output_dir : str, optional
        Output directory (default: "results")
    patches : int, optional
        Total number of patches to extract across all files (default: 30)
    use_ants : bool, optional
        Use ANTs transformation (default: False)
    ants_warp : str, optional
        ANTs warp file path
    ants_iwarp : str, optional
        ANTs inverse warp file path  
    ants_aff : str, optional
        ANTs affine file path
        
    Returns:
    --------
    dict
        Results with successful/failed files and timing
        
    Example:
    --------
    >>> from cumulative import process_batch
    >>> results = process_batch('brain.nii.gz', './trk_files/', patches=50)
    >>> print(f"Processed {len(results['successful'])} files successfully")
    """
    
    # Validate inputs
    if not os.path.exists(nifti_file):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_file}")
    
    if not os.path.exists(trk_directory):
        raise FileNotFoundError(f"TRK directory not found: {trk_directory}")
    
    # Find TRK files
    trk_files = []
    for file in os.listdir(trk_directory):
        if file.endswith('.trk'):
            trk_files.append(os.path.join(trk_directory, file))
    
    if not trk_files:
        raise ValueError(f"No TRK files found in {trk_directory}")
    
    # Auto-calculate dimensions if not provided
    if new_dim is None:
        print("Auto-calculating target dimensions...")
        try:
            import nibabel as nib
            import numpy as np
            
            nifti_img = nib.load(nifti_file)
            original_shape = nifti_img.shape[:3]
            original_voxel_sizes = nifti_img.header.get_zooms()[:3]
            
            # Calculate physical size and target dimensions
            physical_size_mm = np.array(original_shape) * np.array(original_voxel_sizes)
            target_dimensions = np.round(physical_size_mm / voxel_size).astype(int)
            target_dimensions = np.clip(target_dimensions, 32, 4000)
            new_dim = tuple(target_dimensions)
            
            print(f"  Original shape: {original_shape}")
            print(f"  Target dimensions: {new_dim}")
        except Exception as e:
            print(f"  Warning: Could not auto-calculate ({e}), using default")
            new_dim = (116, 140, 96)
    
    # Set default patch size if not provided (optimized for thin dimensions)
    if patch_size is None:
        
        patch_size = [600, 1, 600]
        print(f"  Auto-patch size (high-resolution thin): {patch_size}")
    else:
        # Validate user-provided patch size
        if isinstance(patch_size, list) and len(patch_size) == 3:
            if patch_size[1] == 1 and new_dim[1] > 1:
                print(f"  Warning: Ultra-thin patch Y dimension ({patch_size[1]}) may miss fiber details.")
                print(f"  Consider increasing to at least {min(8, new_dim[1])} for better fiber capture.")
        print(f"  Using patch size: {patch_size}")
    
    # Determine output image size based on patch processing mode
    if disable_patch_processing:
        # Default to 1024x1024 when patch processing is disabled
        output_image_size = (1024, 1024)
        print(f"  Output image size (patch processing disabled): {output_image_size}")
    else:
        # Use patch size to determine output image size when patch processing is enabled
        if isinstance(patch_size, list) and len(patch_size) >= 2:
            # For 3D patch_size like [600, 1, 600], use the first and last dimensions for 2D output
            if len(patch_size) == 3:
                output_image_size = (patch_size[0], patch_size[2])
            else:
                output_image_size = (patch_size[0], patch_size[1])
        else:
            # Fallback if patch_size format is unexpected
            output_image_size = (1024, 1024)
        print(f"  Output image size (from patch size): {output_image_size}")
    
    print(f"Processing {len(trk_files)} TRK files")
    print(f"NIfTI: {nifti_file}")
    print(f"Total patches: {patches}")
    print(f"Target dimensions: {new_dim}")
    print(f"Patch size: {patch_size}")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'successful': [], 'failed': [], 'total_time': 0}
    start_time = time.time()
    
    # Calculate patches per file with better distribution
    patches_per_file = max(1, patches // len(trk_files))
    remaining_patches = patches - (patches_per_file * len(trk_files))
    
    # Calculate examples per file more intelligently
    total_expected_patches = sum([
        patches_per_file + (1 if j < remaining_patches else 0) 
        for j in range(len(trk_files))
    ])
    examples_per_patch = max(1, n_examples // total_expected_patches) if total_expected_patches > 0 else 1
    
    print(f"Distribution: {examples_per_patch} examples per patch")
    
    for i, trk_path in enumerate(trk_files, 1):
        file_start = time.time()
        trk_name = os.path.basename(trk_path)
        base_name = os.path.splitext(trk_name)[0]
        
        # Calculate patches for this file
        extra_patch = 1 if (i - 1) < remaining_patches else 0
        file_patches = patches_per_file + extra_patch
        
        print(f"[{i}/{len(trk_files)}] {trk_name} ({file_patches} patches)...", end=' ')
        
        try:
            # Quick streamline count check for memory optimization
            try:
                import nibabel as nib
                from dipy.io.streamline import load_tractogram
                
                # Quick streamline count check
                tractogram = load_tractogram(trk_path, reference='same', bbox_valid_check=False)
                streamline_count = len(tractogram.streamlines)
                
                # Adjust processing strategy for large files
                if streamline_count > 100000:
                    print(f"({streamline_count:,} streamlines - large file) ", end='')
                    # For very large files, increase patch count to distribute load
                    if file_patches == 1 and patches < len(trk_files) * 2:
                        file_patches = min(3, patches)  # Use up to 3 patches for large files
                        print(f"[auto-increased to {file_patches} patches] ", end='')
                elif streamline_count < 10:
                    print(f"({streamline_count} streamlines - sparse file) ", end='')
                else:
                    print(f"({streamline_count:,} streamlines) ", end='')
                    
            except Exception as e:
                print(f"[streamline check failed: {e}] ", end='')
            
            # Set up configuration with all syntract options
            config = {
                # Core processing
                'new_dim': new_dim,
                'voxel_size': voxel_size,
                'skip_synthesis': skip_synthesis,
                'disable_patch_processing': disable_patch_processing,
                
                # Patch processing
                'total_patches': file_patches,
                'patch_size': patch_size,
                'patch_output_dir': os.path.join(output_dir, "patches", base_name),
                'patch_prefix': f"{base_name}_{patch_prefix}",
                'min_streamlines_per_patch': min_streamlines_per_patch,
                
                # Visualization
                'n_examples': examples_per_patch,
                'viz_prefix': viz_prefix,
                'enable_orange_blobs': enable_orange_blobs,
                'orange_blob_probability': orange_blob_probability,
                'output_image_size': output_image_size,  # Pass the calculated output image size
                
                # Masks and bundles
                'save_masks': save_masks,
                'use_high_density_masks': use_high_density_masks,
                'mask_thickness': mask_thickness,
                'density_threshold': density_threshold,
                'min_bundle_size': min_bundle_size,
                'label_bundles': label_bundles,
                
                # Cleanup
                'cleanup_intermediate': cleanup_intermediate
            }
            
            # Add ANTs if specified
            if use_ants:
                if not all([ants_warp, ants_iwarp, ants_aff]):
                    raise ValueError("ANTs transformation requires warp, iwarp, and affine files")
                config.update({
                    'use_ants': True,
                    'ants_warp_path': ants_warp,
                    'ants_iwarp_path': ants_iwarp,
                    'ants_aff_path': ants_aff
                })
            
            # Process the file
            result = process_syntract(
                input_nifti=nifti_file,
                input_trk=trk_path,
                output_base=os.path.join(output_dir, "processed", f"processed_{base_name}"),
                **config
            )
            
            file_time = time.time() - file_start
            
            if result.get('success', False):
                # Check if any patches were actually extracted
                patches_extracted = 0
                if result.get('stage') == 'patch_extraction_optimized':
                    patch_result = result.get('result', {})
                    patches_extracted = patch_result.get('patches_extracted', 0)
                else:
                    # For other stages, assume some data was processed if successful
                    patches_extracted = file_patches
                
                if patches_extracted == 0:
                    print(f"SUCCESS ({file_time:.1f}s) - No patches extracted (ANTs transformation issue)")
                else:
                    print(f"SUCCESS ({file_time:.1f}s)")
                
                results['successful'].append({
                    'file': trk_name,
                    'time': file_time,
                    'patches': file_patches,
                    'patches_extracted': patches_extracted
                })
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"FAILED ({file_time:.1f}s) - {error_msg}")
                results['failed'].append({
                    'file': trk_name,
                    'error': error_msg,
                    'time': file_time
                })
                
        except Exception as e:
            file_time = time.time() - file_start
            print(f"ERROR ({file_time:.1f}s) - {str(e)}")
            results['failed'].append({
                'file': trk_name,
                'error': str(e),
                'time': file_time
            })
    
    # Summary
    results['total_time'] = time.time() - start_time
    
    print(f"\n" + "="*50)
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Total time: {results['total_time']:.1f}s")
    
    if results['successful']:
        total_patches_extracted = sum(r.get('patches_extracted', r['patches']) for r in results['successful'])
        avg_time = sum(r['time'] for r in results['successful']) / len(results['successful'])
        files_with_data = sum(1 for r in results['successful'] if r.get('patches_extracted', r['patches']) > 0)
        print(f"Total patches: {total_patches_extracted}")
        print(f"Files with extracted data: {files_with_data}/{len(results['successful'])}")
        print(f"Avg time/file: {avg_time:.1f}s")
    
    print(f"Results: {output_dir}/")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main CLI interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch processing for MRI synthesis with all syntract options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/
  
  # With better patch distribution for 200 visualizations  
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \\
    --total-patches 50 --n-examples 200 --enable-orange-blobs
  
  # For thin slice data (Y dimension ~1) - recommended settings
  # Output images will be 256x256 pixels (from patch size)
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \\
    --total-patches 30 --patch-size 256 8 256 --n-examples 200 --voxel-size 0.05
  
  # With larger output images (800x800)
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \\
    --patch-size 800 1 800 --n-examples 200
  
  # With ANTs transformation
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \\
    --use-ants --ants-warp warp.nii.gz --ants-iwarp iwarp.nii.gz --ants-aff affine.mat
  
  # With custom dimensions and voxel size
  python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \\
    --new-dim 800 20 800 --voxel-size 0.05
        """
    )
    
    # Required arguments
    parser.add_argument('--nifti', required=True, help='Path to NIfTI file')
    parser.add_argument('--trk-dir', required=True, help='Directory containing TRK files')
    parser.add_argument('--output', default='results', help='Output directory (default: results)')
    
    # Synthesis parameters
    synthesis_group = parser.add_argument_group("Synthesis Parameters")
    synthesis_group.add_argument("--skip-synthesis", action="store_true",
                                help="Skip synthesis step and use input files directly")
    synthesis_group.add_argument("--new-dim", nargs=3, type=int, default=None,
                                help="Target dimensions (X Y Z). Auto-calculated if not provided")
    synthesis_group.add_argument("--voxel-size", type=float, default=0.05,
                                help="Target voxel size in mm (default: 0.05)")
    
    # ANTs transformation
    ants_group = parser.add_argument_group("ANTs Transformation")
    ants_group.add_argument('--use-ants', action='store_true', help='Use ANTs transformation')
    ants_group.add_argument('--ants-warp', help='ANTs warp file')
    ants_group.add_argument('--ants-iwarp', help='ANTs inverse warp file')
    ants_group.add_argument('--ants-aff', help='ANTs affine file')
    
    # Patch Processing
    patch_group = parser.add_argument_group("Patch Processing")
    patch_group.add_argument("--total-patches", type=int, default=30,
                            help="Total patches to extract across all files (default: 30)")
    patch_group.add_argument("--patch-size", type=int, nargs='+', default=None,
                            help="Patch dimensions [width, height, depth]. Also determines output image size when patch processing is enabled. Auto-calculated if not provided")
    patch_group.add_argument("--min-streamlines-per-patch", type=int, default=20,
                            help="Minimum streamlines required per patch (default: 20)")
    patch_group.add_argument("--patch-prefix", default="patch",
                            help="Prefix for patch files (default: 'patch')")
    patch_group.add_argument("--disable-patch-processing", action="store_true",
                            help="Disable patch processing and use traditional full-volume synthesis. Output images default to 1024x1024")
    
    # Visualization parameters
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--n-examples", type=int, default=10,
                          help="Number of visualization examples to generate (default: 10)")
    viz_group.add_argument("--viz-prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files (default: 'synthetic_')")
    viz_group.add_argument("--enable-orange-blobs", action="store_true",
                          help="Enable orange blob generation to simulate injection site artifacts")
    viz_group.add_argument("--orange-blob-probability", type=float, default=0.3,
                          help="Probability of applying orange blobs (0.0-1.0, default: 0.3)")
    
    # Mask and Bundle parameters
    mask_group = parser.add_argument_group("Mask & Bundle Detection")
    mask_group.add_argument("--save-masks", action="store_true", default=True,
                           help="Save binary masks alongside visualizations (default: True)")
    mask_group.add_argument("--use-high-density-masks", action="store_true",
                           help="Use high-density mask generation (default: True)")
    mask_group.add_argument("--no-high-density-masks", action="store_true",
                           help="Disable high-density mask generation")
    mask_group.add_argument("--mask-thickness", type=int, default=1,
                           help="Thickness of generated masks (default: 1)")
    mask_group.add_argument("--density-threshold", type=float, default=0.15,
                           help="Fiber density threshold for masking (default: 0.15)")
    mask_group.add_argument("--min-bundle-size", type=int, default=20,
                           help="Minimum size for bundle detection (default: 20)")
    mask_group.add_argument("--label-bundles", action="store_true",
                           help="Label individual fiber bundles (default: False)")
    
    # Cleanup parameters
    cleanup_group = parser.add_argument_group("Cleanup")
    cleanup_group.add_argument("--cleanup-intermediate", action="store_true", default=True,
                              help="Remove intermediate NIfTI and TRK files after processing to save disk space (default: True)")
    cleanup_group.add_argument("--no-cleanup-intermediate", action="store_true",
                              help="Keep intermediate NIfTI and TRK files after processing")
    
    args = parser.parse_args()
    
    # Handle high density masks default (True unless explicitly disabled)
    use_high_density_masks = not args.no_high_density_masks if hasattr(args, 'no_high_density_masks') else True
    if hasattr(args, 'use_high_density_masks') and args.use_high_density_masks:
        use_high_density_masks = True
    
    # Handle cleanup parameter (default True unless explicitly disabled)
    cleanup_intermediate = not getattr(args, 'no_cleanup_intermediate', False)
    if hasattr(args, 'cleanup_intermediate') and not args.cleanup_intermediate:
        cleanup_intermediate = False
    
    try:
        results = process_batch(
            nifti_file=args.nifti,
            trk_directory=args.trk_dir,
            output_dir=args.output,
            patches=args.total_patches,
            patch_size=args.patch_size,
            min_streamlines_per_patch=args.min_streamlines_per_patch,
            patch_prefix=args.patch_prefix,
            use_ants=args.use_ants,
            ants_warp=args.ants_warp,
            ants_iwarp=args.ants_iwarp,
            ants_aff=args.ants_aff,
            voxel_size=args.voxel_size,
            new_dim=tuple(args.new_dim) if args.new_dim else None,
            skip_synthesis=args.skip_synthesis,
            n_examples=args.n_examples,
            viz_prefix=args.viz_prefix,
            enable_orange_blobs=args.enable_orange_blobs,
            orange_blob_probability=args.orange_blob_probability,
            save_masks=args.save_masks,
            use_high_density_masks=use_high_density_masks,
            mask_thickness=args.mask_thickness,
            density_threshold=args.density_threshold,
            min_bundle_size=args.min_bundle_size,
            label_bundles=args.label_bundles,
            disable_patch_processing=args.disable_patch_processing,
            cleanup_intermediate=cleanup_intermediate
        )
        
        if results['failed']:
            print(f"\nWarning: Some files failed. Check {args.output}/summary.json for details.")
            sys.exit(1)
        else:
            print(f"\nAll files processed successfully!")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()