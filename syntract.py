#!/usr/bin/env python\n\"\"\"\nStreamlined MRI Synthesis and Visualization Pipeline\n\nThis script provides a unified interface for processing NIfTI/TRK data\nand generating visualizations with minimal parameters.\n\nENHANCED CORNUCOPIA FEATURES:\n- Expanded cornucopia preset options from 6 to 16+ different background styles\n- Added new creative presets: high_contrast, minimal_noise, speckle_heavy, \n  debris_field, smooth_gradients, mixed_effects, clean_gradients, textured_background\n- Implemented weighted selection system for balanced distribution of preset types\n- Categorized presets into clean (30%), subtle (30%), moderate (20%), heavy (20%)\n- This ensures diverse background appearances while maintaining medical realism\n\nUsage:\n    python syntract.py --input brain.nii.gz --trk fibers.trk [options]\n\"\"\"

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

# Import patch-first optimization (separate from main synthesis)
try:
    from synthesis.patch_first_processing import process_patch_first_extraction
    PATCH_FIRST_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'synthesis'))
        from patch_first_processing import process_patch_first_extraction
        PATCH_FIRST_AVAILABLE = True
    except ImportError:
        PATCH_FIRST_AVAILABLE = False
        print("Warning: Patch-first optimization module not available")

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
    print("WARNING: Process timed out - this might be due to HPC resource limits")
    raise TimeoutError("Process exceeded time limit")


def run_visualization_stage(nifti_file, trk_file, args, output_image_size=(1024, 1024)):
    """Run the visualization generation stage."""
    if not SYNTRACT_AVAILABLE:
        raise RuntimeError("Syntract viewer module not available. Cannot run visualization stage.")
    
    print("\n" + "="*60)
    print("STAGE 2: VISUALIZATION GENERATION")
    print("="*60)
    print(f"Output image size: {output_image_size}")
    
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
    viz_args.output_image_size = output_image_size  # Pass the output image size
    
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
                if file.endswith('.nii.gz') or file.endswith('.nii'):
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
            
            print(f"Successfully processed {slice_info['folder']}")
            
        except Exception as e:
            print(f"ERROR: Error processing {slice_info['folder']}: {e}")
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


def calculate_target_dimensions(input_nifti, target_voxel_size=0.05):
    """
    Automatically calculate target dimensions based on input data characteristics.
    
    This function analyzes the input NIfTI file to determine appropriate target
    dimensions that:
    1. Maintain the original aspect ratio
    2. Provide the desired spatial resolution (voxel size)
    3. Stay within reasonable memory limits
    
    Algorithm:
    - Calculates physical size of original volume (shape × voxel_sizes)
    - Determines target dimensions as physical_size / target_voxel_size
    - Applies constraints: minimum 32 voxels, maximum 4000 voxels per dimension
    - Rounds to integer dimensions
    
    Parameters
    ----------
    input_nifti : str
        Path to input NIfTI file
    target_voxel_size : float
        Target voxel size in mm (default: 0.05mm for high-resolution)
        
    Returns
    -------
    tuple
        Target dimensions (x, y, z) that maintain aspect ratio and provide good resolution
        
    Examples
    --------
    >>> # For a 400×50×400 volume with 0.05mm voxels, targeting 0.05mm:
    >>> calculate_target_dimensions('brain.nii.gz', 0.05)
    (400, 50, 400)  # Same size since voxel sizes match
    
    >>> # For the same volume, targeting 0.1mm voxels:
    >>> calculate_target_dimensions('brain.nii.gz', 0.1) 
    (200, 25, 200)  # Half the size since voxels are 2x larger
    """
    import nibabel as nib
    import numpy as np
    
    try:
        # Load the input NIfTI to get its characteristics
        nifti_img = nib.load(input_nifti)
        original_shape = nifti_img.shape[:3]  # Get spatial dimensions only
        original_voxel_sizes = nifti_img.header.get_zooms()[:3]
        
        # Calculate physical size of the original volume
        physical_size_mm = np.array(original_shape) * np.array(original_voxel_sizes)
        
        # Calculate target dimensions that maintain aspect ratio
        target_dimensions = np.round(physical_size_mm / target_voxel_size).astype(int)
        
        # Apply reasonable constraints
        min_dim = 32   # Minimum dimension for meaningful processing
        max_dim = 4000  # Maximum dimension to prevent excessive memory usage
        
        target_dimensions = np.clip(target_dimensions, min_dim, max_dim)
        
        print(f"Auto-calculating target dimensions:")
        print(f"  Original shape: {original_shape}")
        print(f"  Original voxel sizes: {[f'{v:.3f}' for v in original_voxel_sizes]} mm")
        print(f"  Physical size: {[f'{s:.1f}' for s in physical_size_mm]} mm")
        print(f"  Target voxel size: {target_voxel_size} mm")
        print(f"  Calculated target dimensions: {tuple(target_dimensions.tolist())}")
        
        return tuple(target_dimensions.tolist())
        
    except Exception as e:
        print(f"Warning: Could not auto-calculate dimensions ({e})")
        print(f"  Using default dimensions: (116, 140, 96)")
        return (116, 140, 96)

def process_syntract(input_nifti, input_trk, output_base, new_dim, voxel_size, 
                    use_ants=False, ants_warp_path=None, ants_iwarp_path=None, ants_aff_path=None,
                    slice_count=None, enable_slice_extraction=False, slice_output_dir=None,
                    use_simplified_slicing=True, force_full_slicing=False, auto_batch_process=False,
                    total_patches=50, patch_size=[600, 1, 600], min_streamlines_per_patch=20,
                    patch_prefix="patch", patch_output_dir="patches", patch_batch_size=50,
                    skip_synthesis=False, disable_patch_processing=False,
                    n_examples=10, viz_prefix="synthetic_",
                    enable_orange_blobs=False, orange_blob_probability=0.3,
                    save_masks=True, use_high_density_masks=True, mask_thickness=1,
                    density_threshold=0.6, min_bundle_size=2000, label_bundles=False,
                    output_image_size=None, cleanup_intermediate=True):
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
        print(f"WARNING: Process received {signal_name}")
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
        print("Starting syntract processing pipeline...")
        print(f"Input NIfTI: {input_nifti}")
        print(f"Input TRK: {input_trk}")
        print(f"Output base: {output_base}")
        
        # Determine output image size based on patch processing mode and user preference
        if output_image_size is None:
            if disable_patch_processing:
                # Default to 1024x1024 when patch processing is disabled
                output_image_size = (1024, 1024)
                print(f"Output image size (patch processing disabled): {output_image_size}")
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
                print(f"Output image size (from patch size): {output_image_size}")
        else:
            print(f"Output image size (user specified): {output_image_size}")
        
        # CRITICAL: Check for patch processing mode (enabled by default unless disabled)
        if not disable_patch_processing and PATCH_FIRST_AVAILABLE:
            print("\nPATCH-FIRST PROCESSING ENABLED (DEFAULT)")
            print("   Skipping full volume synthesis and proceeding directly to optimized patch extraction!")
            print("   This will dramatically reduce memory usage and processing time!")
            
            # Convert patch_size to target dimensions
            if len(patch_size) == 2:
                target_patch_size = (patch_size[0], 1, patch_size[1])
            elif len(patch_size) == 3:
                target_patch_size = tuple(patch_size)
            else:
                raise ValueError(f"Invalid patch_size: {patch_size}")
            
            # Create output directory for patches
            patch_output_path = patch_output_dir or 'patches'
            os.makedirs(patch_output_path, exist_ok=True)
            
            # Run optimized patch-first extraction directly on original files
            patch_result = process_patch_first_extraction(
                original_nifti_path=input_nifti,
                original_trk_path=input_trk,
                target_voxel_size=voxel_size,
                target_patch_size=target_patch_size,
                target_dimensions=new_dim,
                num_patches=total_patches,
                output_prefix=os.path.join(patch_output_path, patch_prefix.rstrip('_')),
                min_streamlines_per_patch=min_streamlines_per_patch,
                use_ants=use_ants,
                ants_warp_path=ants_warp_path,
                ants_iwarp_path=ants_iwarp_path,
                ants_aff_path=ants_aff_path,
                random_state=None,  # Could be parameterized
                use_gpu=True
            )
            
            # Add visualization generation for patches if successful
            if patch_result['success'] and patch_result['patches_extracted'] > 0:
                print(f"\nGenerating visualizations for {patch_result['patches_extracted']} patches...")
                
                try:
                    from syntract_viewer.patch_extraction import _generate_patch_visualization
                    
                    for patch_detail in patch_result['patch_details']:
                        patch_id = patch_detail['patch_id']
                        nifti_file = patch_detail['files']['nifti']
                        trk_file = patch_detail['files']['trk']
                        
                        if os.path.exists(nifti_file) and os.path.exists(trk_file):
                            patch_viz_prefix = f"patch_{patch_id:04d}"
                            _generate_patch_visualization(
                                nifti_file, trk_file, 
                                patch_output_path, 
                                patch_viz_prefix,
                                save_masks=save_masks,
                                contrast_method='clahe',
                                background_enhancement='preserve_edges',
                                cornucopia_preset='clean_optical',
                                tract_linewidth=1.0,
                                mask_thickness=mask_thickness,
                                density_threshold=density_threshold,
                                gaussian_sigma=2.0,
                                close_gaps=False,
                                closing_footprint_size=3,
                                label_bundles=label_bundles,
                                min_bundle_size=min_bundle_size,
                                enable_orange_blobs=enable_orange_blobs,
                                orange_blob_probability=orange_blob_probability,
                                output_image_size=output_image_size
                            )
                    
                    print(f"Patch-first optimization complete!")
                    print(f"   Processing time: {patch_result['processing_time']:.2f}s")
                    print(f"   Memory usage: Dramatically reduced vs. traditional method")
                    print(f"   Output location: {patch_output_path}")
                    
                except ImportError as e:
                    print(f"Warning: Could not import visualization module for patches: {e}")
                    print("Patches extracted successfully but visualization skipped.")
            
            # Cleanup patch files if requested
            if cleanup_intermediate and patch_result['success'] and patch_result['patches_extracted'] > 0:
                print(f"\nCleaning up {patch_result['patches_extracted']} intermediate patch files...")
                patch_files_to_cleanup = []
                
                # Find all patch files created
                for i in range(1, patch_result['patches_extracted'] + 1):
                    patch_nifti = os.path.join(patch_output_path, f"{patch_prefix.rstrip('_')}_{i:04d}.nii.gz")
                    patch_trk = os.path.join(patch_output_path, f"{patch_prefix.rstrip('_')}_{i:04d}.trk")
                    patch_files_to_cleanup.extend([patch_nifti, patch_trk])
                
                # Clean up the files
                cleaned_count = 0
                for file_path in patch_files_to_cleanup:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            cleaned_count += 1
                            print(f"  Removed: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {file_path}: {e}")
                
                print(f"Successfully cleaned up {cleaned_count} patch files")
            
            return {'success': True, 'stage': 'patch_extraction_optimized', 'result': patch_result}
        
        # Determine which files to use for downstream processing
        base_nifti = input_nifti
        base_trk = input_trk
        synthesis_result = None
        intermediate_files = []  # Track files created for potential cleanup
        
        if skip_synthesis:
            print("\nSkipping synthesis step (--skip_synthesis enabled)")
            print(f"Using input files directly:")
            print(f"    NIfTI: {base_nifti}")
            print(f"    TRK: {base_trk}")
        else:
            # Run base synthesis first to create files at target dimensions
            print("\nRunning base synthesis...")
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
            
            # Use synthesized files for downstream processing
            base_nifti = f"{output_base}.nii"
            base_trk = f"{output_base}.trk"
            
            # Track intermediate files for cleanup
            intermediate_files.extend([base_nifti, base_trk])
            
            if not os.path.exists(base_nifti) or not os.path.exists(base_trk):
                print(f"WARNING: Synthesis output files not found:")
                print(f"    Expected NIfTI: {base_nifti}")
                print(f"    Expected TRK: {base_trk}")
                print(f"    Falling back to original files")
                base_nifti = input_nifti
                base_trk = input_trk
                # Don't track original files for cleanup
                intermediate_files = []
            else:
                print(f"Using synthesized files:")
                print(f"    NIfTI: {base_nifti}")
                print(f"    TRK: {base_trk}")
        
        # Cleanup intermediate files if requested
        if cleanup_intermediate and intermediate_files:
            print(f"\nCleaning up {len(intermediate_files)} intermediate files...")
            for file_path in intermediate_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"  Removed: {file_path}")
                    else:
                        print(f"  Skipped (not found): {file_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_path}: {e}")
        
        # Return successful completion for traditional synthesis
        if disable_patch_processing:
            return {'success': True, 'stage': 'traditional_synthesis', 'result': synthesis_result}
        
        # Proceed with slice extraction or visualization as needed
        if enable_slice_extraction:
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
            
            return {'success': True, 'stage': 'synthesis', 'result': synthesis_result}
        
    except Exception as e:
        print(f"ERROR: Error in syntract processing: {e}")
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
  
  # With patch processing (default mode - extracts random patches from different brain regions)
  # Output images will be 1024x1024 pixels (from patch size dimensions)
  python syntract.py --input brain.nii.gz --trk fibers.trk \\
    --total_patches 100 --patch_size 1024 1 1024 --min_streamlines_per_patch 5
  
  # With smaller output images (512x512)
  python syntract.py --input brain.nii.gz --trk fibers.trk \\
    --patch_size 512 1 512
  
  # Disable patch processing and use traditional full-volume synthesis (slower, more memory-intensive)
  # Output images will be 1024x1024 pixels by default
  python syntract.py --input brain.nii.gz --trk fibers.trk --disable_patch_processing
        """
    )
    
    # Essential input arguments
    parser.add_argument("--input", required=True, help="Input NIfTI file path")
    parser.add_argument("--trk", required=True, help="Input TRK file path")
    parser.add_argument("--output", default="output", help="Output base name")
    
    # Synthesis parameters
    synthesis_group = parser.add_argument_group("Synthesis Parameters")
    synthesis_group.add_argument("--skip_synthesis", action="store_true",
                                help="Skip synthesis step and use input files directly (useful for pre-processed data)")
    synthesis_group.add_argument("--new_dim", nargs=3, type=int, default=None,
                                help="Target dimensions (X Y Z). If not provided, will auto-calculate based on input data and voxel size")
    synthesis_group.add_argument("--voxel_size", type=float, default=0.05,
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
    
    # Patch Processing (Default Method)
    patch_group = parser.add_argument_group("Patch Processing")
    patch_group.add_argument("--total_patches", type=int, default=50,
                            help="Number of patches to extract and process (default: 50)")
    patch_group.add_argument("--patch_size", type=int, nargs='+', default=[600, 1, 600],
                            help="Patch dimensions [width, height, depth] for processing. Also determines output image size when patch processing is enabled (default: 600x1x600 -> 600x600 output images)")
    patch_group.add_argument("--patch_output_dir", default="patches",
                            help="Directory for patch outputs (default: 'patches')")
    patch_group.add_argument("--min_streamlines_per_patch", type=int, default=20,
                            help="Minimum streamlines required per patch (default: 20)")
    patch_group.add_argument("--patch_batch_size", type=int, default=50,
                            help="Number of patches to process before memory cleanup (default: 50)")
    patch_group.add_argument("--patch_prefix", default="patch",
                            help="Prefix for patch files (default: 'patch')")
    patch_group.add_argument("--random_state", type=int, 
                            help="Random seed for reproducible patch extraction")
    patch_group.add_argument("--disable_patch_processing", action="store_true",
                            help="Disable patch processing and use traditional full-volume synthesis (slower, more memory-intensive). Output images default to 1024x1024")
    patch_group.add_argument("--cleanup_intermediate", action="store_true", default=True,
                            help="Remove intermediate NIfTI and TRK files after processing to save disk space (default: True)")
    patch_group.add_argument("--no_cleanup_intermediate", action="store_true",
                            help="Keep intermediate NIfTI and TRK files after processing")
    
    # Visualization parameters
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--n_examples", type=int, default=10,
                          help="Number of visualization examples to generate")
    viz_group.add_argument("--viz_prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files")
    viz_group.add_argument("--enable_orange_blobs", action="store_true",
                          help="Enable orange blob generation to simulate injection site artifacts")
    viz_group.add_argument("--orange_blob_probability", type=float, default=0.3,
                          help="Probability of applying orange blobs to each visualization (0.0-1.0, default: 0.3)")
    
    # Mask and Bundle parameters (unified defaults)
    mask_group = parser.add_argument_group("Mask & Bundle Detection")
    mask_group.add_argument("--save_masks", action="store_true", default=True,
                           help="Save binary masks alongside visualizations (default: True)")
    mask_group.add_argument("--use_high_density_masks", action="store_true", 
                           help="Use high-density mask generation with prominent bundles (default: True)")
    mask_group.add_argument("--no_high_density_masks", action="store_true",
                           help="Disable high-density mask generation and use regular masks")
    mask_group.add_argument("--mask_thickness", type=int, default=1,
                           help="Base thickness for mask lines (default: 1, auto-scaled by output size)")
    mask_group.add_argument("--density_threshold", type=float, default=0.6,
                           help="Fiber density threshold for masking (default: 0.6, extremely aggressive filtering)")
    mask_group.add_argument("--min_bundle_size", type=int, default=2000,
                           help="Minimum size for bundle detection (default: 2000, only keeps very large prominent bundles)")
    mask_group.add_argument("--label_bundles", action="store_true",
                           help="Label individual fiber bundles with distinct colors (default: False)")
    
    
    args = parser.parse_args()
    
    # Validate ANTs parameters
    if args.use_ants:
        if not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
            print("ERROR: --use_ants requires --ants_warp, --ants_iwarp, and --ants_aff")
            sys.exit(1)
    
    # Set up extraction mode
    enable_slice_extraction = args.slice_count is not None
    slice_output_dir = args.slice_output_dir or f"{args.output}_slices"
    
    # Auto-calculate target dimensions if not provided
    if args.new_dim is None:
        print("No target dimensions specified, auto-calculating...")
        target_dimensions = calculate_target_dimensions(args.input, args.voxel_size)
    else:
        target_dimensions = tuple(args.new_dim)
        print(f"Using specified target dimensions: {target_dimensions}")
    
    # Handle high density masks default (True unless explicitly disabled)
    use_high_density_masks = not args.no_high_density_masks if hasattr(args, 'no_high_density_masks') else True
    if hasattr(args, 'use_high_density_masks') and args.use_high_density_masks:
        use_high_density_masks = True
    
    # Handle cleanup parameter (default True unless explicitly disabled)
    cleanup_intermediate = not getattr(args, 'no_cleanup_intermediate', False)
    if hasattr(args, 'cleanup_intermediate') and not args.cleanup_intermediate:
        cleanup_intermediate = False
    
    # Run the pipeline
    result = process_syntract(
        input_nifti=args.input,
        input_trk=args.trk,
        output_base=args.output,
        new_dim=target_dimensions,
        voxel_size=args.voxel_size,
        skip_synthesis=args.skip_synthesis,
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
        disable_patch_processing=getattr(args, 'disable_patch_processing', False),
        patch_output_dir=args.patch_output_dir,
        total_patches=args.total_patches,
        patch_size=args.patch_size,
        min_streamlines_per_patch=args.min_streamlines_per_patch,
        patch_batch_size=args.patch_batch_size,
        patch_prefix=args.patch_prefix,
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
        output_image_size=None,  # Let process_syntract determine it automatically
        cleanup_intermediate=cleanup_intermediate
    )
    
    if not result['success']:
        print(f"ERROR: Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main() 