#!/usr/bin/env python
"""
Combined MRI Synthesis and Visualization Pipeline

This script provides a unified interface for both processing NIfTI/TRK data
and generating visualizations. It combines the synthesis pipeline for data
processing with the syntract viewer for visualization generation.

Usage:
    python syntract.py --input brain.nii.gz --trk fibers.trk [options]
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Import synthesis functions
try:
    from synthesis.main import process_and_save
    SYNTHESIS_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'synthesis'))
        from main import process_and_save
        SYNTHESIS_AVAILABLE = True
    except ImportError:
        SYNTHESIS_AVAILABLE = False
        print("Warning: Synthesis module not available")

# Import syntract viewer functions  
try:
    from syntract_viewer.generate_fiber_examples import generate_examples_original_mode, generate_examples_with_spatial_subdivisions
    SYNTRACT_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'syntract_viewer'))
        from generate_fiber_examples import generate_examples_original_mode, generate_examples_with_spatial_subdivisions
        SYNTRACT_AVAILABLE = True
    except ImportError:
        SYNTRACT_AVAILABLE = False
        print("Warning: Syntract viewer module not available")


def run_synthesis_stage(args, temp_dir):
    """Run the synthesis processing stage."""
    if not SYNTHESIS_AVAILABLE:
        raise RuntimeError("Synthesis module not available. Cannot run processing stage.")
    
    print("\n" + "="*60)
    print("STAGE 1: SYNTHESIS PROCESSING")
    print("="*60)
    
    # Set up synthesis output paths
    synthesis_output = os.path.join(temp_dir, "processed")
    
    # Prepare synthesis parameters
    synthesis_args = {
        'original_nifti_path': args.input,
        'original_trk_path': args.trk,
        'target_voxel_size': args.voxel_size[0] if len(args.voxel_size) == 1 else tuple(args.voxel_size),
        'target_dimensions': tuple(args.new_dim),
        'output_prefix': synthesis_output,
        'num_jobs': args.jobs,
        'patch_center': tuple(args.patch_center) if args.patch_center else None,
        'reduction_method': args.reduction,
        'use_gpu': not args.cpu and args.use_gpu,
        'interpolation_method': args.interp,
        'step_size': args.step_size,
        'max_output_gb': args.max_gb,
        'use_ants': args.use_ants,
        'ants_warp_path': args.ants_warp,
        'ants_iwarp_path': args.ants_iwarp,
        'ants_aff_path': args.ants_aff,
        'force_dimensions': args.force_dimensions,
        'transform_mri_with_ants': args.transform_mri_with_ants
    }
    
    # Run synthesis
    process_and_save(**synthesis_args)
    
    # Return paths to processed files
    processed_nifti = synthesis_output + ".nii.gz"
    processed_trk = synthesis_output + ".trk"
    
    if not os.path.exists(processed_nifti) or not os.path.exists(processed_trk):
        raise RuntimeError("Synthesis stage failed to produce output files")
    
    print(f"Synthesis completed successfully!")
    print(f"Processed NIfTI: {processed_nifti}")
    print(f"Processed TRK: {processed_trk}")
    
    return processed_nifti, processed_trk


def run_visualization_stage(nifti_file, trk_file, args):
    """Run the visualization generation stage."""
    if not SYNTRACT_AVAILABLE:
        raise RuntimeError("Syntract viewer module not available. Cannot run visualization stage.")
    
    print("\n" + "="*60)
    print("STAGE 2: VISUALIZATION GENERATION")
    print("="*60)
    
    # Create output directory for visualizations
    viz_output_dir = args.viz_output_dir
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Prepare visualization arguments
    viz_args = argparse.Namespace()
    
    # Set all the visualization parameters  
    viz_args.nifti = nifti_file
    viz_args.trk = trk_file
    viz_args.output_dir = viz_output_dir
    viz_args.examples = args.n_examples  # Note: visualization code expects 'examples' not 'n_examples'
    viz_args.prefix = args.viz_prefix
    viz_args.view = args.slice_mode
    viz_args.specific_slice = args.specific_slice
    viz_args.streamline_percentage = args.streamline_percentage
    viz_args.roi_sphere = None  # Set to None as in original code
    viz_args.tract_linewidth = args.tract_linewidth
    viz_args.save_masks = args.save_masks
    viz_args.use_high_density_masks = args.use_high_density_masks
    viz_args.label_bundles = args.label_bundles
    viz_args.mask_thickness = args.mask_thickness
    viz_args.min_fiber_pct = args.min_fiber_percentage
    viz_args.max_fiber_pct = args.max_fiber_percentage
    viz_args.min_bundle_size = args.min_bundle_size
    viz_args.density_threshold = args.density_threshold
    viz_args.contrast_method = args.contrast_method
    viz_args.background_preset = args.background_preset
    viz_args.cornucopia_preset = args.cornucopia_preset
    viz_args.enable_sharpening = args.enable_sharpening
    viz_args.sharpening_strength = args.sharpening_strength
    viz_args.close_gaps = args.close_gaps
    viz_args.closing_footprint_size = args.closing_footprint_size
    viz_args.randomize = args.randomize_viz
    viz_args.random_state = args.random_state
    viz_args.spatial_subdivisions = args.use_spatial_subdivisions
    viz_args.n_subdivisions = args.n_subdivisions
    viz_args.max_streamlines_per_region = args.max_streamlines_per_subdivision
    
    # Import background enhancement availability
    try:
        from syntract_viewer.background_enhancement import enhance_slice_background
        viz_args.background_enhancement_available = True
    except ImportError:
        viz_args.background_enhancement_available = False
    
    # Run visualization generation
    if args.use_spatial_subdivisions:
        print("Using spatial subdivisions mode")
        generate_examples_with_spatial_subdivisions(viz_args, viz_args.background_enhancement_available)
    else:
        print("Using original mode")
        generate_examples_original_mode(viz_args, viz_args.background_enhancement_available)
    
    print(f"Visualization generation completed!")
    print(f"Output directory: {viz_output_dir}")


def copy_final_outputs(temp_dir, args):
    """Copy final outputs to the specified locations."""
    if args.keep_processed:
        # Copy processed files to final location
        synthesis_output = os.path.join(temp_dir, "processed")
        processed_nifti = synthesis_output + ".nii.gz"
        processed_trk = synthesis_output + ".trk"
        
        final_nifti = args.output + "_processed.nii.gz"
        final_trk = args.output + "_processed.trk"
        
        if os.path.exists(processed_nifti):
            shutil.copy2(processed_nifti, final_nifti)
            print(f"Saved processed NIfTI: {final_nifti}")
        
        if os.path.exists(processed_trk):
            shutil.copy2(processed_trk, final_trk)
            print(f"Saved processed TRK: {final_trk}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined MRI synthesis and visualization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - process and visualize
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results

  # Skip synthesis, only visualize existing files
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results --skip_synthesis

  # Use ANTs transforms with high-quality visualizations
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results \\
    --use_ants --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat \\
    --background_preset high_quality --cornucopia_preset realistic_optical

  # Generate spatial subdivisions with custom parameters
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results \\
    --use_spatial_subdivisions --n_subdivisions 10 --n_examples 3
        """
    )

    # Input/Output arguments
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to input NIfTI (.nii or .nii.gz) file")
    parser.add_argument("--trk", type=str, required=True, 
                       help="Path to input TRK (.trk) file")
    parser.add_argument("--output", type=str, default="combined_output", 
                       help="Base name for output files")
    
    # Pipeline control
    parser.add_argument("--skip_synthesis", action="store_true", 
                       help="Skip synthesis stage and use input files directly for visualization")
    parser.add_argument("--skip_visualization", action="store_true", 
                       help="Skip visualization stage, only run synthesis")
    parser.add_argument("--keep_processed", action="store_true",
                       help="Keep the processed NIfTI and TRK files from synthesis stage")
    
    # Synthesis parameters
    synthesis_group = parser.add_argument_group('Synthesis Parameters')
    synthesis_group.add_argument("--voxel_size", type=float, nargs='+', default=[0.5],
                                help="New voxel size: single value for isotropic or three values for anisotropic")
    synthesis_group.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], 
                                help="New image dimensions (x, y, z)")
    synthesis_group.add_argument("--jobs", type=int, default=8, 
                                help="Number of parallel jobs (-1 for all CPUs)")
    synthesis_group.add_argument("--patch_center", type=float, nargs=3, default=None, 
                                help="Optional patch center in mm")
    synthesis_group.add_argument("--reduction", type=str, choices=["mip", "mean"], default=None,
                                help="Optional reduction along z-axis")
    synthesis_group.add_argument("--use_gpu", type=lambda x: str(x).lower() != 'false', 
                                nargs='?', const=True, default=True,
                                help="Use GPU acceleration (default: True)")
    synthesis_group.add_argument("--cpu", action="store_true", 
                                help="Force CPU processing (disables GPU)")
    synthesis_group.add_argument("--interp", type=str, choices=["hermite", "linear", "rbf"], default="hermite",
                                help="Interpolation method for streamlines")
    synthesis_group.add_argument("--step_size", type=float, default=0.5, 
                                help="Step size for streamline densification")
    synthesis_group.add_argument("--max_gb", type=float, default=64.0,
                                help="Maximum output size in GB")
    
    # ANTs parameters
    ants_group = parser.add_argument_group('ANTs Transform Parameters')
    ants_group.add_argument("--use_ants", action="store_true", 
                           help="Use ANTs transforms for processing")
    ants_group.add_argument("--ants_warp", type=str, default=None, 
                           help="Path to ANTs warp file")
    ants_group.add_argument("--ants_iwarp", type=str, default=None, 
                           help="Path to ANTs inverse warp file")
    ants_group.add_argument("--ants_aff", type=str, default=None, 
                           help="Path to ANTs affine file")
    ants_group.add_argument("--force_dimensions", action="store_true", 
                           help="Force using specified new_dim even when using ANTs")
    ants_group.add_argument("--transform_mri_with_ants", action="store_true", 
                           help="Also transform MRI with ANTs (default: only transforms streamlines)")
    
    # Visualization parameters
    viz_group = parser.add_argument_group('Visualization Parameters')
    viz_group.add_argument("--viz_output_dir", type=str, default=None,
                          help="Output directory for visualizations (default: {output}_visualizations)")
    viz_group.add_argument("--n_examples", type=int, default=5, 
                          help="Number of visualization examples to generate")
    viz_group.add_argument("--viz_prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files")
    viz_group.add_argument("--slice_mode", type=str, choices=["coronal", "axial", "sagittal"], default="coronal",
                          help="Slice orientation for visualization")
    viz_group.add_argument("--specific_slice", type=int, default=None, 
                          help="Specific slice number to visualize")
    viz_group.add_argument("--streamline_percentage", type=float, default=100.0, 
                          help="Percentage of streamlines to include")
    viz_group.add_argument("--tract_linewidth", type=float, default=1.0, 
                          help="Linewidth for tract visualization")
    viz_group.add_argument("--save_masks", action="store_true", 
                          help="Save fiber masks along with visualizations")
    viz_group.add_argument("--use_high_density_masks", action="store_true",
                          help="Use masks from high-density fibers for all density variations")
    viz_group.add_argument("--label_bundles", action="store_true",
                          help="Label distinct fiber bundles in the masks")
    viz_group.add_argument("--mask_thickness", type=int, default=1,
                          help="Thickness of the mask lines in pixels")
    viz_group.add_argument("--min_fiber_percentage", type=float, default=10.0, 
                          help="Minimum fiber percentage for visualization")
    viz_group.add_argument("--max_fiber_percentage", type=float, default=100.0, 
                          help="Maximum fiber percentage for visualization")
    viz_group.add_argument("--min_bundle_size", type=int, default=20,
                          help="Minimum bundle size for fiber labeling")
    viz_group.add_argument("--density_threshold", type=float, default=0.15,
                          help="Density threshold for fiber visualization")
    
    # Enhancement parameters
    enhancement_group = parser.add_argument_group('Enhancement Parameters')
    enhancement_group.add_argument("--contrast_method", type=str, default="clahe", 
                                  help="Contrast enhancement method")
    enhancement_group.add_argument("--background_preset", type=str, 
                                  choices=["smooth_realistic", "high_quality", "clinical_appearance", 
                                          "preserve_edges", "subtle_enhancement", "lpsvd_denoising", 
                                          "lpsvd_aggressive", "lpsvd_conservative", "hybrid_lpsvd"],
                                  default="preserve_edges",
                                  help="Background enhancement preset")
    enhancement_group.add_argument("--cornucopia_preset", type=str,
                                  choices=["clean_optical", "realistic_optical", "noisy_optical", 
                                          "artifact_simulation", "clinical_simulation"],
                                  default=None,
                                  help="Cornucopia augmentation preset")
    enhancement_group.add_argument("--enable_sharpening", action="store_true", default=True,
                                  help="Enable sharpening in background enhancement")
    enhancement_group.add_argument("--sharpening_strength", type=float, default=0.5,
                                  help="Strength of sharpening effect")
    enhancement_group.add_argument("--close_gaps", action="store_true", 
                                  help="Close gaps in fiber visualizations")
    enhancement_group.add_argument("--closing_footprint_size", type=int, default=5, 
                                  help="Size of morphological closing footprint")
    
    # Spatial subdivisions parameters
    subdivision_group = parser.add_argument_group('Spatial Subdivision Parameters')
    subdivision_group.add_argument("--use_spatial_subdivisions", action="store_true",
                                  help="Use spatial subdivisions for visualization generation")
    subdivision_group.add_argument("--n_subdivisions", type=int, default=8,
                                  help="Number of spatial subdivisions")
    subdivision_group.add_argument("--max_streamlines_per_subdivision", type=int, default=50000,
                                  help="Maximum streamlines per subdivision")
    
    # Miscellaneous parameters
    misc_group = parser.add_argument_group('Miscellaneous Parameters')
    misc_group.add_argument("--randomize_viz", action="store_true", 
                           help="Randomize visualization parameters")
    misc_group.add_argument("--random_state", type=int, default=None, 
                           help="Random seed for reproducible results")
    misc_group.add_argument("--temp_dir", type=str, default=None,
                           help="Temporary directory for intermediate files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_synthesis and args.skip_visualization:
        parser.error("Cannot skip both synthesis and visualization stages")
    
    if args.use_ants and not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
        parser.error("When --use_ants is specified, --ants_warp, --ants_iwarp, and --ants_aff must be provided")
    
    # Set up output directory for visualizations
    if args.viz_output_dir is None:
        args.viz_output_dir = f"{args.output}_visualizations"
    
    # Set up temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="mri_pipeline_")
        cleanup_temp = True
    
    try:
        print(f"Combined MRI Processing and Visualization Pipeline")
        print(f"Input NIfTI: {args.input}")
        print(f"Input TRK: {args.trk}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Final output base: {args.output}")
        print(f"Visualization output: {args.viz_output_dir}")
        
        # Determine which files to use for visualization
        if args.skip_synthesis:
            print("\nSkipping synthesis stage - using original files")
            viz_nifti = args.input
            viz_trk = args.trk
        else:
            # Run synthesis stage
            viz_nifti, viz_trk = run_synthesis_stage(args, temp_dir)
        
        # Run visualization stage
        if not args.skip_visualization:
            run_visualization_stage(viz_nifti, viz_trk, args)
        
        # Copy final outputs if requested
        if not args.skip_synthesis and args.keep_processed:
            copy_final_outputs(temp_dir, args)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if not args.skip_visualization:
            print(f"Visualizations saved to: {args.viz_output_dir}")
        if not args.skip_synthesis and args.keep_processed:
            print(f"Processed files saved with prefix: {args.output}_processed")
    
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        sys.exit(1)
    
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main() 