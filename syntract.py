#!/usr/bin/env python
"""
Streamlined MRI Synthesis and Visualization Pipeline

This script provides a unified interface for processing NIfTI/TRK data
and generating visualizations with minimal parameters.

Usage:
    python syntract.py --input brain.nii.gz --trk fibers.trk [options]
"""

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
    viz_args.save_masks = True
    viz_args.use_high_density_masks = False
    viz_args.label_bundles = False
    viz_args.mask_thickness = 1
    viz_args.min_fiber_pct = 10.0
    viz_args.max_fiber_pct = 100.0
    viz_args.min_bundle_size = 20
    viz_args.density_threshold = 0.15
    viz_args.contrast_method = "clahe"
    viz_args.background_preset = "preserve_edges"
    viz_args.cornucopia_preset = "disabled"
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
    """Run the patch extraction stage with re-synthesis per patch center."""
    if not SYNTRACT_AVAILABLE:
        raise RuntimeError("Syntract viewer module not available. Cannot run patch extraction stage.")
    
    print("\n" + "="*60)
    print("STAGE 3: PATCH EXTRACTION WITH RE-SYNTHESIS")
    print("="*60)
    
    # Create output directory for patches
    patch_output_dir = args.patch_output_dir
    if not patch_output_dir:
        patch_output_dir = "patches"
    os.makedirs(patch_output_dir, exist_ok=True)
    
    # Load the base files to get dimensions and understand coordinate space
    import numpy as np
    import nibabel as nib
    from synthesis.main import process_and_save
    
    base_img = nib.load(nifti_file)
    base_data = base_img.get_fdata()
    base_shape = base_data.shape
    base_trk_file = trk_files[0] if trk_files else None
    
    print(f"Base image shape: {base_shape}")
    print(f"Generating {args.total_patches} patches with size {args.patch_size}")
    
    # Use patch_size for dimensions, not patch_size_3d unless explicitly patch_mode
    if args.patch_mode and hasattr(args, 'patch_size_3d') and args.patch_size_3d:
        patch_dimensions = tuple(args.patch_size_3d)
    else:
        # Use the user-specified patch_size
        if len(args.patch_size) == 2:
            patch_dimensions = (args.patch_size[0], args.patch_size[1], args.patch_size[0])
        elif len(args.patch_size) == 3:
            patch_dimensions = tuple(args.patch_size)
        else:
            raise ValueError(f"Invalid patch_size format: {args.patch_size}")
    
    print(f"Using patch dimensions for re-synthesis: {patch_dimensions}")
    print(f"Original patch_size argument: {args.patch_size}")
    
    # Calculate valid patch centers (ensuring patch fits within base image)
    half_patch = [dim // 2 for dim in patch_dimensions]
    min_centers = [half_patch[i] for i in range(3)]
    max_centers = [base_shape[i] - half_patch[i] for i in range(3)]
    
    print(f"Valid patch center ranges:")
    print(f"  X: {min_centers[0]} to {max_centers[0]}")
    print(f"  Y: {min_centers[1]} to {max_centers[1]}")
    print(f"  Z: {min_centers[2]} to {max_centers[2]}")
    
    # Verify we can generate patches
    for i in range(3):
        if min_centers[i] >= max_centers[i]:
            raise ValueError(f"Patch size {patch_dimensions} too large for base image shape {base_shape}")
    
    # Set random seed for reproducible results
    if hasattr(args, 'random_state') and args.random_state:
        np.random.seed(args.random_state)
    
    # Generate random patch centers
    patch_centers = []
    for patch_idx in range(args.total_patches):
        center = [
            np.random.randint(min_centers[0], max_centers[0]),
            np.random.randint(min_centers[1], max_centers[1]),
            np.random.randint(min_centers[2], max_centers[2])
        ]
        patch_centers.append(center)
    
    print(f"Generated {len(patch_centers)} patch centers")
    
    # Process each patch
    results = {
        'patches_extracted': 0,
        'patches_failed': 0,
        'patch_details': []
    }
    
    for patch_idx, patch_center in enumerate(patch_centers, 1):
        print(f"\n--- Processing Patch {patch_idx}/{args.total_patches} ---")
        print(f"Patch center: {patch_center}")
        print(f"Patch dimensions: {patch_dimensions}")
        
        try:
            # Create subfolder for this patch
            patch_subfolder = os.path.join(patch_output_dir, f"patch_{patch_idx:04d}")
            os.makedirs(patch_subfolder, exist_ok=True)
            
            # Convert patch center from voxel coordinates to world coordinates
            base_affine = base_img.affine
            patch_center_voxel = np.array(patch_center + [1])  # Add homogeneous coordinate
            patch_center_world = (base_affine @ patch_center_voxel)[:3]
            
            print(f"  Patch center (voxel): {patch_center}")
            print(f"  Patch center (world mm): {patch_center_world}")
            
            # Create temporary directory for intermediate files
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix=f"patch_{patch_idx:04d}_")
            try:
                temp_prefix = f"patch_{patch_idx:04d}_temp"
                temp_output = os.path.join(temp_dir, temp_prefix)
                
                print(f"  Using temp directory: {temp_dir}")
                print(f"  Running re-synthesis with center {patch_center} and dimensions {patch_dimensions}")
                
                # Run synthesis with patch-specific parameters
                print(f"  📊 Starting synthesis process...")
                print(f"  🔧 Parameters:")
                print(f"      Input NIfTI: {nifti_file}")
                print(f"      Input TRK: {base_trk_file}")
                print(f"      Target dimensions: {patch_dimensions}")
                print(f"      Patch center (world): {patch_center_world.tolist()}")
                print(f"      Temp output prefix: {temp_output}")
                
                # Add detailed process tracking
                import subprocess
                import threading
                import time
                
                def monitor_process():
                    """Monitor the synthesis process"""
                    start_time = time.time()
                    while True:
                        elapsed = time.time() - start_time
                        if elapsed > 60:  # Log every minute
                            # Get memory usage
                            try:
                                import psutil
                                process = psutil.Process()
                                memory_mb = process.memory_info().rss / 1024 / 1024
                                print(f"  ⏱️  Synthesis still running... {elapsed:.0f}s elapsed, Memory: {memory_mb:.1f} MB")
                            except:
                                print(f"  ⏱️  Synthesis still running... {elapsed:.0f}s elapsed")
                            start_time = time.time()
                        time.sleep(10)  # Check every 10 seconds
                
                # Start monitoring thread
                monitor_thread = threading.Thread(target=monitor_process, daemon=True)
                monitor_thread.start()
                
                try:
                    # Add timeout to prevent hanging
                    import signal
                    def timeout_handler(signum, frame):
                        print(f"  ⚠️  Synthesis process exceeded timeout!")
                        raise TimeoutError("Synthesis process timed out")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(600)  # 10 minute timeout per patch
                    
                    print(f"  🚀 Calling process_and_save...")
                    
                    # Force CPU usage on HPC login nodes
                    import socket
                    hostname = socket.gethostname().lower()
                    force_cpu_usage = any(pattern in hostname for pattern in ['login', 'head', 'master'])
                    
                    if force_cpu_usage:
                        print(f"  🖥️  Detected HPC login node ({hostname}) - forcing CPU processing")
                        use_gpu_for_synthesis = False
                    else:
                        use_gpu_for_synthesis = getattr(args, 'use_gpu', True)
                    
                    # Debug voxel size
                    actual_voxel_size = args.voxel_size if hasattr(args, 'voxel_size') else 0.5
                    actual_step_size = args.voxel_size if hasattr(args, 'voxel_size') else 0.5
                    print(f"  📏 Debug voxel size parameters:")
                    print(f"      args.voxel_size: {getattr(args, 'voxel_size', 'NOT_SET')}")
                    print(f"      actual_voxel_size: {actual_voxel_size}")
                    print(f"      actual_step_size: {actual_step_size}")
                    
                    synthesis_result = process_and_save(
                        original_nifti_path=nifti_file,
                        original_trk_path=base_trk_file,
                        target_voxel_size=actual_voxel_size,
                        target_dimensions=patch_dimensions,
                        output_prefix=temp_output,
                        num_jobs=getattr(args, 'num_jobs', 8),
                        patch_center=patch_center_world.tolist(),  # Use world coordinates for centering
                        use_gpu=use_gpu_for_synthesis,
                        interpolation_method='rbf',  # Use RBF for better curvature preservation
                        step_size=actual_step_size,
                        max_output_gb=getattr(args, 'max_output_gb', 64.0),
                        use_ants=False,  # Skip ANTs since already applied to base files
                        force_dimensions=True
                    )
                    
                    # Cancel timeout
                    signal.alarm(0)
                    print(f"  ✅ process_and_save returned successfully!")
                    
                    print(f"  📊 Synthesis process completed. Result type: {type(synthesis_result)}")
                    if isinstance(synthesis_result, dict):
                        print(f"      Success: {synthesis_result.get('success', 'unknown')}")
                        print(f"      Keys: {list(synthesis_result.keys())}")
                    else:
                        print(f"      Result: {synthesis_result}")
                        
                except TimeoutError as timeout_error:
                    signal.alarm(0)  # Cancel timeout
                    print(f"  ⏰ Synthesis timed out after 10 minutes for patch {patch_idx}")
                    print(f"  📋 This suggests the process is hanging during streamline processing")
                    results['patches_failed'] += 1
                    continue
                except KeyboardInterrupt as kb_error:
                    signal.alarm(0)  # Cancel timeout
                    print(f"  ⛔ Process interrupted by user (Ctrl+C)")
                    results['patches_failed'] += 1
                    break
                except SystemExit as sys_error:
                    signal.alarm(0)  # Cancel timeout
                    print(f"  🚪 Process called sys.exit(): {sys_error}")
                    results['patches_failed'] += 1
                    continue
                except Exception as synthesis_error:
                    signal.alarm(0)  # Cancel timeout
                    print(f"  ❌ Synthesis failed with exception: {synthesis_error}")
                    print(f"  🔍 Exception type: {type(synthesis_error).__name__}")
                    import traceback
                    print(f"  📋 Full traceback:")
                    traceback.print_exc()
                    results['patches_failed'] += 1
                    continue
                
                # Check if synthesis was successful
                # Success is determined by having synthesis outputs, not a 'success' flag
                success = False
                if synthesis_result is not None:
                    if isinstance(synthesis_result, dict):
                        # Check if we have synthesis outputs
                        if 'synthesis_outputs' in synthesis_result:
                            success = True
                        elif synthesis_result.get('success', False):
                            success = True
                    else:
                        # Non-None result might indicate success
                        success = True
                
                if not success:
                    print(f"  ⚠️  Synthesis failed for patch {patch_idx} (no valid output)")
                    results['patches_failed'] += 1
                    continue
                else:
                    print(f"  ✅ Synthesis succeeded for patch {patch_idx}!")
                
                # Move the synthesized files to the patch subfolder
                temp_nifti = f"{temp_output}.nii.gz"
                temp_trk = f"{temp_output}.trk"
                
                final_nifti = os.path.join(patch_subfolder, f"patch_{patch_idx:04d}.nii.gz")
                final_trk = os.path.join(patch_subfolder, f"patch_{patch_idx:04d}_streamlines.trk")
                
                print(f"  🔍 Checking for temp files:")
                print(f"    Expected NIfTI: {temp_nifti}")
                print(f"    Expected TRK: {temp_trk}")
                print(f"    NIfTI exists: {os.path.exists(temp_nifti)}")
                print(f"    TRK exists: {os.path.exists(temp_trk)}")
                
                # List all files in temp directory for debugging
                if os.path.exists(temp_dir):
                    temp_files = os.listdir(temp_dir)
                    print(f"    Files in temp dir: {temp_files}")
                else:
                    print(f"    ⚠️  Temp directory doesn't exist: {temp_dir}")
                
                files_moved = 0
                
                # Use shutil.move instead of os.rename to handle cross-device links
                if os.path.exists(temp_nifti):
                    shutil.move(temp_nifti, final_nifti)
                    print(f"  ✅ Moved NIfTI: {temp_nifti} → {final_nifti}")
                    files_moved += 1
                else:
                    print(f"  ❌ NIfTI file not found: {temp_nifti}")
                
                if os.path.exists(temp_trk):
                    shutil.move(temp_trk, final_trk)
                    print(f"  ✅ Moved TRK: {temp_trk} → {final_trk}")
                    files_moved += 1
                else:
                    print(f"  ❌ TRK file not found: {temp_trk}")
                    # Create placeholder TRK file
                    print(f"  📝 Creating placeholder TRK file: {final_trk}")
                    with open(final_trk, 'w') as f:
                        f.write("# Empty TRK file - no streamlines processed\n")
                
                print(f"  📊 Files moved: {files_moved}/2")
                
            finally:
                # Clean up temporary directory manually
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"  Cleaned up temp dir: {temp_dir}")
            
            # Generate visualization if requested (outside temp directory handling)
            if SYNTRACT_AVAILABLE and not getattr(args, 'skip_visualization', False):
                try:
                    from syntract_viewer.core import visualize_nifti_with_trk_coronal
                    viz_path = os.path.join(patch_subfolder, f"patch_{patch_idx:04d}_visualization.png")
                    visualize_nifti_with_trk_coronal(
                        nifti_file=final_nifti,
                        trk_file=final_trk,
                        output_file=viz_path,
                        n_slices=1,
                        slice_idx=patch_dimensions[1] // 2,  # Middle slice
                        streamline_percentage=100.0,
                        save_masks=getattr(args, 'save_masks', False)
                    )
                    print(f"  Generated visualization: {viz_path}")
                except Exception as viz_e:
                    print(f"  Warning: Visualization failed: {viz_e}")
            
            patch_result = {
                'patch_idx': patch_idx,
                'center': patch_center,
                'center_world_mm': patch_center_world.tolist(),
                'dimensions': patch_dimensions,
                'nifti_file': final_nifti,
                'trk_file': final_trk,
                'output_folder': patch_subfolder
            }
            
            results['patches_extracted'] += 1
            results['patch_details'].append(patch_result)
            print(f"✓ Patch {patch_idx} completed successfully")
                
        except Exception as e:
            results['patches_failed'] += 1
            print(f"✗ Patch {patch_idx} failed with exception: {e}")
    
    # Save summary
    import json
    summary = {
        'total_patches_requested': args.total_patches,
        'patches_extracted': results['patches_extracted'],
        'patches_failed': results['patches_failed'],
        'patch_details': results['patch_details'],
        'extraction_params': {
            'patch_dimensions': patch_dimensions,
            'base_image_shape': base_shape,
            'random_state': getattr(args, 'random_state', None)
        }
    }
    
    summary_path = os.path.join(patch_output_dir, "patch_extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"Patch extraction with re-synthesis completed!")
    print(f"Output directory: {patch_output_dir}")
    
    return results


def process_syntract(input_nifti, input_trk, output_base, new_dim, voxel_size, 
                    use_ants=False, ants_warp_path=None, ants_iwarp_path=None, ants_aff_path=None,
                    slice_count=None, enable_slice_extraction=False, slice_output_dir=None,
                    use_simplified_slicing=True, force_full_slicing=False, auto_batch_process=False,
                    patch_mode=False, patch_size_3d=None, num_patches=None,
                    enable_patch_extraction=False, patch_output_dir=None, total_patches=None,
                    patch_size=None, min_streamlines_per_patch=5, patch_prefix="patch_",
                    n_examples=10, viz_output_dir=None, viz_prefix="viz_"):
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
            step_size=voxel_size,  # CRITICAL: Match step size to voxel size for curvature preservation
            interpolation_method='hermite'  # Use Hermite for base synthesis too
        )
        
        # Check if patch extraction is enabled (after base synthesis)
        if enable_patch_extraction:
            print("\n📦 Patch extraction enabled - starting patch processing...")
            
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
                    'patch_mode': patch_mode,
                    'patch_size_3d': patch_size_3d,
                    'n_examples': n_examples,
                    'viz_prefix': viz_prefix
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
    patch_group = parser.add_argument_group("Patch Extraction")
    patch_group.add_argument("--patch_mode", action="store_true",
                            help="Enable 3D patch extraction mode (replaces slice extraction and disables auto batch)")
    patch_group.add_argument("--patch_size_3d", type=int, nargs=3, default=[64, 64, 64],
                            help="3D patch dimensions (x y z) in voxels")
    patch_group.add_argument("--num_patches", type=int, default=10,
                            help="Number of patches to extract in patch mode")
    
    # Legacy patch extraction (for backward compatibility)
    patch_group.add_argument("--enable_patch_extraction", action="store_true",
                            help="Enable legacy patch extraction from processed data")
    patch_group.add_argument("--patch_output_dir", 
                            help="Directory for patch outputs")
    patch_group.add_argument("--total_patches", type=int, default=100,
                            help="Total number of patches to extract (legacy)")
    patch_group.add_argument("--patch_size", type=int, nargs='+', default=[128, 128],
                            help="Patch size - 2D: [width, height] or 3D: [width, height, depth]. "
                                 "For re-synthesis mode, 3D dimensions become target synthesis size.")
    patch_group.add_argument("--min_streamlines_per_patch", type=int, default=50,
                            help="Minimum streamlines per patch (legacy)")
    patch_group.add_argument("--patch_prefix", default="patch",
                            help="Prefix for patch files")
    
    # Visualization parameters
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--n_examples", type=int, default=3,
                          help="Number of visualization examples to generate")
    viz_group.add_argument("--viz_prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files")
    
    args = parser.parse_args()
    
    # Validate ANTs parameters
    if args.use_ants:
        if not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
            print("❌ Error: --use_ants requires --ants_warp, --ants_iwarp, and --ants_aff")
            sys.exit(1)
    
    # Set up extraction mode
    enable_slice_extraction = args.slice_count is not None and not args.patch_mode
    enable_patch_mode = args.patch_mode
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
        patch_mode=args.patch_mode,
        patch_size_3d=args.patch_size_3d,
        num_patches=args.num_patches,
        enable_patch_extraction=args.enable_patch_extraction,
        patch_output_dir=args.patch_output_dir,
        total_patches=args.total_patches,
        patch_size=args.patch_size,
        min_streamlines_per_patch=args.min_streamlines_per_patch,
        patch_prefix=args.patch_prefix,
        n_examples=args.n_examples,
        viz_prefix=args.viz_prefix
    )
    
    if not result['success']:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main() 