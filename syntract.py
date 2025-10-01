#!/usr/bin/env python\n\"\"\"\nStreamlined MRI Synthesis and Visualization Pipeline\n\nThis script provides a unified interface for processing NIfTI/TRK data\nand generating visualizations with minimal parameters.\n\nENHANCED CORNUCOPIA FEATURES:\n- Expanded cornucopia preset options from 6 to 16+ different background styles\n- Added new creative presets: high_contrast, minimal_noise, speckle_heavy, \n  debris_field, smooth_gradients, mixed_effects, clean_gradients, textured_background\n- Implemented weighted selection system for balanced distribution of preset types\n- Categorized presets into clean (35%), subtle (35%), moderate (20%), heavy (10%)\n- This ensures diverse background appearances while maintaining medical realism\n\nUsage:\n    python syntract.py --input brain.nii.gz --trk fibers.trk [options]\n\"\"\"

import argparse
import os
import sys
import tempfile
import shutil
import psutil
import signal
import time
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
    from syntract_viewer.generate_fiber_examples import generate_examples_original_mode
    SYNTRACT_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'syntract_viewer'))
        from generate_fiber_examples import generate_examples_original_mode
        SYNTRACT_AVAILABLE = True
    except ImportError:
        SYNTRACT_AVAILABLE = False
        print("Warning: Syntract viewer module not available")


def monitor_memory():
    """Monitor current memory usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        print(f"Current memory usage: {memory_mb:.1f} MB")
        return memory_mb
    except Exception as e:
        print(f"Could not monitor memory: {e}")
        return 0


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    print("⚠️  Process timed out - this might be due to HPC resource limits")
    raise TimeoutError("Process exceeded time limit")


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
    
    # Set essential visualization parameters  
    viz_args.nifti = nifti_file
    viz_args.trk = trk_file
    viz_args.output_dir = viz_output_dir
    viz_args.examples = args.n_examples
    viz_args.prefix = args.viz_prefix
    viz_args.view = "coronal"  # Fixed to coronal
    viz_args.specific_slice = None
    viz_args.streamline_percentage = 100.0
    viz_args.roi_sphere = None
    viz_args.tract_linewidth = 1.0
    viz_args.save_masks = args.save_masks
    viz_args.use_high_density_masks = args.use_high_density_masks
    viz_args.label_bundles = args.label_bundles
    viz_args.enable_orange_blobs = args.enable_orange_blobs
    viz_args.orange_blob_probability = args.orange_blob_probability
    viz_args.mask_thickness = args.mask_thickness
    viz_args.min_fiber_pct = 10.0
    viz_args.max_fiber_pct = 100.0
    viz_args.min_bundle_size = args.min_bundle_size
    viz_args.density_threshold = args.density_threshold
    viz_args.contrast_method = "clahe"
    viz_args.background_preset = "preserve_edges"
    viz_args.cornucopia_preset = "clean_optical"
    viz_args.enable_sharpening = False
    viz_args.sharpening_strength = 1.0
    viz_args.close_gaps = False
    viz_args.closing_footprint_size = 3
    viz_args.randomize = False
    viz_args.random_state = 42
    
    # Run visualization
    generate_examples_original_mode(viz_args, True)  # background_enhancement_available=True
    
    print(f"Visualization generation completed!")
    print(f"Output directory: {viz_output_dir}")


def batch_process_slice_folders(slice_output_dir, args):
    """
    Automatically batch process extracted slice folders through the visualization pipeline.
    
    Args:
        slice_output_dir (str): Directory containing slice folders (slice_XXX/)
        args: Arguments object with visualization parameters
        
    Returns:
        dict: Summary of batch processing results
    """
    print(f"\n=== Starting Automated Batch Processing ===")
    print(f"Processing slices from: {slice_output_dir}")
    
    if not os.path.exists(slice_output_dir):
        print(f"Error: Slice output directory not found: {slice_output_dir}")
        return {'success': False, 'error': 'Directory not found'}
    
    # Find all slice folders
    slice_folders = []
    for item in os.listdir(slice_output_dir):
        item_path = os.path.join(slice_output_dir, item)
        if os.path.isdir(item_path) and item.startswith('slice_'):
            # Check if folder contains both required files
            nifti_file = None
            trk_file = None
            
            for file in os.listdir(item_path):
                if file.endswith('.nii.gz'):
                    nifti_file = os.path.join(item_path, file)
                elif file.endswith('.trk'):
                    trk_file = os.path.join(item_path, file)
            
            if nifti_file and trk_file:
                slice_folders.append({
                    'folder': item,
                    'path': item_path,
                    'nifti': nifti_file,
                    'trk': trk_file
                })
            else:
                print(f"Warning: Incomplete slice folder {item} (missing NIfTI or TRK file)")
    
    if not slice_folders:
        print("No valid slice folders found for processing")
        return {'success': False, 'error': 'No valid slice folders found'}
    
    print(f"Found {len(slice_folders)} valid slice folders to process")
    
    # Create batch output directory
    batch_viz_dir = os.path.join(slice_output_dir, "batch_visualizations")
    os.makedirs(batch_viz_dir, exist_ok=True)
    
    results = {
        'success': True,
        'total_slices': len(slice_folders),
        'processed_slices': 0,
        'failed_slices': 0,
        'slice_results': [],
        'output_dir': batch_viz_dir
    }
    
    for i, slice_info in enumerate(slice_folders, 1):
        print(f"\nProcessing slice {i}/{len(slice_folders)}: {slice_info['folder']}")
        
        try:
            # Create individual output directory for this slice
            slice_viz_dir = os.path.join(batch_viz_dir, slice_info['folder'])
            os.makedirs(slice_viz_dir, exist_ok=True)
            
            # Update args for this slice
            slice_args = argparse.Namespace(**vars(args))
            slice_args.viz_output_dir = slice_viz_dir
            slice_args.prefix = f"{slice_info['folder']}_"
            
            # Run visualization for this slice
            run_visualization_stage(slice_info['nifti'], slice_info['trk'], slice_args)
            
            results['processed_slices'] += 1
            results['slice_results'].append({
                'slice': slice_info['folder'],
                'status': 'success',
                'output_dir': slice_viz_dir
            })
            
            print(f"✓ Successfully processed {slice_info['folder']}")
            
        except Exception as e:
            print(f"✗ Error processing {slice_info['folder']}: {e}")
            results['failed_slices'] += 1
            results['slice_results'].append({
                'slice': slice_info['folder'],
                'status': 'failed',
                'error': str(e)
            })
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"Successfully processed: {results['processed_slices']}/{results['total_slices']} slices")
    print(f"Failed: {results['failed_slices']} slices")
    print(f"Output directory: {batch_viz_dir}")
    
    # Save summary
    import json
    summary_file = os.path.join(batch_viz_dir, "batch_processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved: {summary_file}")
    
    return results


def run_patch_extraction_stage(nifti_file, trk_files, args):
    """Run the patch extraction stage using robust methodology."""
    if not SYNTRACT_AVAILABLE:
        raise RuntimeError("Syntract viewer module not available. Cannot run patch extraction stage.")
    
    print("\n" + "="*60)
    print("STAGE 3: ROBUST PATCH EXTRACTION")
    print("="*60)
    
    # Import the robust patch extraction module
    try:
        from syntract_viewer.patch_extraction import extract_random_patches
    except ImportError:
        raise RuntimeError("Could not import patch extraction module.")
    
    # Create output directory for patches
    patch_output_dir = args.patch_output_dir
    if not patch_output_dir:
        patch_output_dir = "patches"
    os.makedirs(patch_output_dir, exist_ok=True)
    
    print(f"Input NIfTI: {nifti_file}")
    print(f"Input TRK files: {trk_files}")
    print(f"Output directory: {patch_output_dir}")
    print(f"Generating {args.total_patches} patches")
    
    # Determine patch size - convert from args format to 3D tuple
    if hasattr(args, 'patch_size') and args.patch_size:
        if len(args.patch_size) == 2:
            patch_dimensions = (args.patch_size[0], args.patch_size[1], args.patch_size[0])
        elif len(args.patch_size) == 3:
            patch_dimensions = tuple(args.patch_size)
        else:
            raise ValueError(f"Invalid patch_size format: {args.patch_size}")
    else:
        # Default 3D patch size
        patch_dimensions = (128, 128, 128)
    
    print(f"Using patch dimensions: {patch_dimensions}")
    
    # Run robust patch extraction
    try:
        results = extract_random_patches(
            nifti_file=nifti_file,
            trk_files=trk_files,
            output_dir=patch_output_dir,
            total_patches=args.total_patches,
            patch_size=patch_dimensions,
            min_streamlines_per_patch=getattr(args, 'min_streamlines_per_patch', 30),
            random_state=getattr(args, 'random_state', None),
            prefix=getattr(args, 'patch_prefix', 'patch').rstrip('_'),
            save_masks=getattr(args, 'save_masks', True),
            contrast_method='clahe',
            background_enhancement='preserve_edges',
            cornucopia_preset='clean_optical',
            tract_linewidth=1.0,
            mask_thickness=getattr(args, 'mask_thickness', 1),
            density_threshold=getattr(args, 'density_threshold', 0.15),
            gaussian_sigma=2.0,
            close_gaps=False,
            closing_footprint_size=3,
            label_bundles=getattr(args, 'label_bundles', False),
            min_bundle_size=getattr(args, 'min_bundle_size', 20),
            enable_orange_blobs=getattr(args, 'enable_orange_blobs', False),
            orange_blob_probability=getattr(args, 'orange_blob_probability', 0.3)
        )
        
        print(f"\n=== Patch Extraction Results ===")
        print(f"Patches requested: {results['total_patches_requested']}")
        print(f"Patches extracted: {results['patches_extracted']}")
        print(f"Patches failed: {results['patches_failed']}")
        print(f"Success rate: {results['patches_extracted']/results['total_patches_requested']*100:.1f}%")
        print(f"Output directory: {patch_output_dir}")
        
        # Visualizations are already generated by the patch extraction process
        print(f"\n✅ Patch extraction completed with integrated visualizations!")
        print(f"    - {results['patches_extracted']} patches extracted")
        print(f"    - Visualizations saved as patch_XXXX_visualization.png")
        print(f"    - Masks saved as patch_XXXX_visualization_mask_slice0.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Patch extraction failed: {e}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()
        
        return {
            'patches_extracted': 0,
            'patches_failed': args.total_patches,
            'error': str(e)
        }

def process_syntract(input_nifti, input_trk, output_base, new_dim, voxel_size, 
                    use_ants=False, ants_warp_path=None, ants_iwarp_path=None, ants_aff_path=None,
                    slice_count=None, enable_slice_extraction=False, slice_output_dir=None,
                    use_simplified_slicing=True, force_full_slicing=False, auto_batch_process=False,
                    enable_patch_extraction=False, patch_output_dir=None, total_patches=None,
                    patch_size=None, min_streamlines_per_patch=5, patch_prefix="patch_",
                    n_examples=10, viz_output_dir=None, viz_prefix="viz_",
                    enable_orange_blobs=False, orange_blob_probability=0.3,
                    save_masks=True, use_high_density_masks=False, mask_thickness=1,
                    density_threshold=0.15, min_bundle_size=20, label_bundles=False):
    """Main processing function"""
    import signal
    import sys
    
    def signal_handler(signum, frame):
        """Handle various signals that might kill the process"""
        signal_names = {
            signal.SIGTERM: "SIGTERM (terminated)",
            signal.SIGINT: "SIGINT (interrupted)", 
            signal.SIGKILL: "SIGKILL (killed)",
            signal.SIGSEGV: "SIGSEGV (segmentation fault)",
            signal.SIGABRT: "SIGABRT (aborted)"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        print(f"🚨 Process received {signal_name}")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    # Note: SIGKILL cannot be caught
    if hasattr(signal, 'SIGSEGV'):
        signal.signal(signal.SIGSEGV, signal_handler)
    if hasattr(signal, 'SIGABRT'):
        signal.signal(signal.SIGABRT, signal_handler)
        
    try:
        print("🚀 Starting syntract processing pipeline...")
        print(f"Input NIfTI: {input_nifti}")
        print(f"Input TRK: {input_trk}")
        print(f"Output base: {output_base}")
        
        # Always run base synthesis first to create files at target dimensions
        print("\n🧠 Running base synthesis...")
        from synthesis.main import process_and_save
        
        synthesis_result = process_and_save(
            original_nifti_path=input_nifti,
            original_trk_path=input_trk,
            target_voxel_size=voxel_size,
            target_dimensions=new_dim,
            output_prefix=output_base,
            use_ants=use_ants,
            ants_warp_path=ants_warp_path,
            ants_iwarp_path=ants_iwarp_path,
            ants_aff_path=ants_aff_path,
            step_size=voxel_size * 0.5,  # Use fine step size (0.5x voxel) for good curvature preservation
            interpolation_method='hermite'  # Use Hermite for base synthesis too
        )
        
        # Check if patch extraction is enabled (after base synthesis)
        if enable_patch_extraction:
            print("\n Patch extraction enabled - starting patch processing...")
            
            # Use the synthesized files as input for patch extraction
            base_nifti = f"{output_base}.nii.gz"
            base_trk = f"{output_base}.trk"
            
            if not os.path.exists(base_nifti) or not os.path.exists(base_trk):
                print(f"⚠️  Base synthesis files not found:")
                print(f"    Expected NIfTI: {base_nifti}")
                print(f"    Expected TRK: {base_trk}")
                print(f"    Using original files instead")
                base_nifti = input_nifti
                base_trk = input_trk
            else:
                print(f"✅ Using synthesized base files:")
                print(f"    Base NIfTI: {base_nifti}")
                print(f"    Base TRK: {base_trk}")
            
            patch_result = run_patch_extraction_stage(base_nifti, [base_trk], 
                argparse.Namespace(**{
                    'patch_size': patch_size,
                    'total_patches': total_patches,
                    'min_streamlines_per_patch': min_streamlines_per_patch,
                    'patch_prefix': patch_prefix,
                    'patch_output_dir': patch_output_dir or 'patches',
                    'voxel_size': voxel_size,
                    'new_dim': new_dim,
                    'use_ants': use_ants,
                    'ants_warp_path': ants_warp_path,
                    'ants_iwarp_path': ants_iwarp_path,
                    'ants_aff_path': ants_aff_path,
                    'n_examples': n_examples,
                    'viz_prefix': viz_prefix,
                    'enable_orange_blobs': enable_orange_blobs,
                    'orange_blob_probability': orange_blob_probability
                }))
            
            return {'success': True, 'stage': 'patch_extraction', 'result': patch_result}
        
        return {'success': True, 'stage': 'synthesis', 'result': synthesis_result}
        
    except Exception as e:
        print(f"❌ Error in syntract processing: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main entry point for the syntract console script."""
    parser = argparse.ArgumentParser(
        description="Streamlined MRI Synthesis and Visualization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python syntract.py --input brain.nii.gz --trk fibers.trk
  
  # With ANTs transformation
  python syntract.py --input brain.nii.gz --trk fibers.trk --use_ants \\
    --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat
  
  # With slice extraction
  python syntract.py --input brain.nii.gz --trk fibers.trk --slice_count 10
  
  # With auto batch processing (processes all slices automatically)
  python syntract.py --input brain.nii.gz --trk fibers.trk --slice_count 10 --auto_batch_process
  
  # With patch extraction (extracts random patches from different brain regions)
  python syntract.py --input brain.nii.gz --trk fibers.trk --enable_patch_extraction \\
    --total_patches 100 --patch_size 1024 1024 --min_streamlines_per_patch 5
        """
    )
    
    # Essential input arguments
    parser.add_argument("--input", required=True, help="Input NIfTI file path")
    parser.add_argument("--trk", required=True, help="Input TRK file path")
    parser.add_argument("--output", default="output", help="Output base name")
    
    # Synthesis parameters
    synthesis_group = parser.add_argument_group("Synthesis Parameters")
    synthesis_group.add_argument("--new_dim", nargs=3, type=int, default=[116, 140, 96],
                                help="Target dimensions (X Y Z)")
    synthesis_group.add_argument("--voxel_size", type=float, default=0.5,
                                help="Target voxel size in mm")
    
    # ANTs parameters
    ants_group = parser.add_argument_group("ANTs Transformation")
    ants_group.add_argument("--use_ants", action="store_true", 
                           help="Use ANTs transformation")
    ants_group.add_argument("--ants_warp", help="ANTs warp field file")
    ants_group.add_argument("--ants_iwarp", help="ANTs inverse warp field file")
    ants_group.add_argument("--ants_aff", help="ANTs affine transformation file")
    
    # Slice extraction parameters
    slice_group = parser.add_argument_group("Slice Extraction")
    slice_group.add_argument("--slice_count", type=int,
                            help="Number of coronal slices to extract")
    slice_group.add_argument("--slice_output_dir", 
                            help="Directory for slice outputs")
    slice_group.add_argument("--auto_batch_process", action="store_true",
                            help="Automatically process all extracted slices through visualization")
    
    # Patch extraction parameters
    patch_group = parser.add_argument_group("Robust Patch Extraction")
    patch_group.add_argument("--enable_patch_extraction", action="store_true",
                            help="Enable robust 3D patch extraction with proper coordinate transformations")
    patch_group.add_argument("--patch_output_dir", 
                            help="Directory for patch outputs (default: 'patches')")
    patch_group.add_argument("--total_patches", type=int, default=10,
                            help="Total number of patches to extract (default: 10)")
    patch_group.add_argument("--patch_size", type=int, nargs='+', default=[300, 15, 300],
                            help="Patch size - 3D: [width, height, depth] or 2D: [width, height] (default: 300x15x300 for good resolution with reasonable thickness)")
    patch_group.add_argument("--min_streamlines_per_patch", type=int, default=30,
                            help="Minimum streamlines required per patch (default: 30)")
    patch_group.add_argument("--max_patch_trials", type=int, default=100,
                            help="Maximum trials per patch to find adequate streamlines (default: 100)")
    patch_group.add_argument("--random_state", type=int, 
                            help="Random seed for reproducible patch extraction")
    patch_group.add_argument("--patch_prefix", default="patch",
                            help="Prefix for patch files")
    
    # Visualization parameters
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--n_examples", type=int, default=3,
                          help="Number of visualization examples to generate")
    viz_group.add_argument("--viz_prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files")
    viz_group.add_argument("--enable_orange_blobs", action="store_true",
                          help="Enable orange blob generation to simulate injection site artifacts")
    viz_group.add_argument("--orange_blob_probability", type=float, default=0.3,
                          help="Probability of applying orange blobs to each visualization (0.0-1.0, default: 0.3)")
    
    # Mask and Bundle parameters
    mask_group = parser.add_argument_group("Mask & Bundle Detection")
    mask_group.add_argument("--save_masks", action="store_true", default=True,
                           help="Save binary masks alongside visualizations (default: True)")
    mask_group.add_argument("--use_high_density_masks", action="store_true",
                           help="Use high-density mask generation (default: False)")
    mask_group.add_argument("--mask_thickness", type=int, default=1,
                           help="Thickness of generated masks (default: 1)")
    mask_group.add_argument("--density_threshold", type=float, default=0.15,
                           help="Fiber density threshold for masking (default: 0.15)")
    mask_group.add_argument("--min_bundle_size", type=int, default=20,
                           help="Minimum size for bundle detection (default: 20)")
    mask_group.add_argument("--label_bundles", action="store_true",
                           help="Label individual fiber bundles (default: False)")
    
    
    args = parser.parse_args()
    
    # Validate ANTs parameters
    if args.use_ants:
        if not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
            print("❌ Error: --use_ants requires --ants_warp, --ants_iwarp, and --ants_aff")
            sys.exit(1)
    
    # Set up extraction mode
    enable_slice_extraction = args.slice_count is not None
    slice_output_dir = args.slice_output_dir or f"{args.output}_slices"
    
    # Run the pipeline
    result = process_syntract(
        input_nifti=args.input,
        input_trk=args.trk,
        output_base=args.output,
        new_dim=tuple(args.new_dim),
        voxel_size=args.voxel_size,
        use_ants=args.use_ants,
        ants_warp_path=args.ants_warp,
        ants_iwarp_path=args.ants_iwarp,
        ants_aff_path=args.ants_aff,
        slice_count=args.slice_count,
        enable_slice_extraction=enable_slice_extraction,
        slice_output_dir=slice_output_dir,
        use_simplified_slicing=True,
        force_full_slicing=False,
        auto_batch_process=args.auto_batch_process,
        enable_patch_extraction=args.enable_patch_extraction,
        patch_output_dir=args.patch_output_dir,
        total_patches=args.total_patches,
        patch_size=args.patch_size,
        min_streamlines_per_patch=args.min_streamlines_per_patch,
        patch_prefix=args.patch_prefix,
        n_examples=args.n_examples,
        viz_prefix=args.viz_prefix,
        enable_orange_blobs=args.enable_orange_blobs,
        orange_blob_probability=args.orange_blob_probability,
        save_masks=args.save_masks,
        use_high_density_masks=args.use_high_density_masks,
        mask_thickness=args.mask_thickness,
        density_threshold=args.density_threshold,
        min_bundle_size=args.min_bundle_size,
        label_bundles=args.label_bundles
    )
    
    if not result['success']:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main() 