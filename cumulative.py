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
import tempfile
import shutil
import gc
from typing import Tuple, List
import numpy as np

# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the syntract function
from syntract import process_syntract


def process_batch(nifti_file, trk_directory, output_dir="results", patches=30, 
                  patch_size=None, min_streamlines_per_patch=20, patch_prefix="patch",
                  use_ants=False, ants_warp=None, ants_iwarp=None, ants_aff=None,
                  voxel_size=0.05, new_dim=None, skip_synthesis=False,
                  n_examples=10, viz_prefix="synthetic_", enable_orange_blobs=False,
                  orange_blob_probability=0.3, save_masks=True, use_high_density_masks=True,
                  mask_thickness=1, density_threshold=0.6, min_bundle_size=2000,
                  label_bundles=False, disable_patch_processing=False, cleanup_intermediate=True):
    """
    Process multiple TRK files with a common NIfTI file.
    
    This is the main function for batch processing - simple and efficient.
    
    Mask Parameters (Unified Defaults)
    ----------------------------------
    All mask parameters use consistent defaults across the entire codebase:
    - mask_thickness: 1 (auto-scaled by output image size)
    - density_threshold: 0.6 (extremely aggressive filtering, only largest bundles)
    - min_bundle_size: 2000 (only keeps very large, prominent fiber bundles)
    - use_high_density_masks: True (creates prominent, well-connected bundles)
    
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


def _capture_figure_as_array(fig, target_size=(1024, 1024)):
    """
    Capture matplotlib figure as numpy array.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to capture
    target_size : tuple
        Target size (height, width) for the output array
        
    Returns
    -------
    np.ndarray
        RGB image array with shape (H, W, 3) in uint8 format
    """
    # Draw the canvas to ensure it's rendered
    fig.canvas.draw()
    
    # Get the RGBA buffer from the canvas
    # Use buffer_rgba() for newer matplotlib, with fallback to tostring_rgb() for older versions
    try:
        # Try newer API first (matplotlib >= 3.3)
        buf = fig.canvas.buffer_rgba()
        ncols, nrows = fig.canvas.get_width_height()
        # buffer_rgba() returns RGBA, so we need to handle 4 channels
        array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
        # Convert RGBA to RGB by dropping the alpha channel
        array = array[:, :, :3]
    except AttributeError:
        # Fallback for older matplotlib versions
        try:
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        except AttributeError:
            # Alternative: use print_to_buffer()
            buf, (width, height) = fig.canvas.print_to_buffer()
            array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            array = array[:, :, :3]  # Drop alpha channel
    
    # Resize if needed
    from syntract_viewer.utils import resize_image_to_size
    if array.shape[:2] != target_size:
        array = resize_image_to_size(array, target_size, is_mask=False)
    
    return array


def _load_and_resize_mask(mask_path, target_size=(1024, 1024)):
    """
    Load mask from file and resize.
    
    Parameters
    ----------
    mask_path : str
        Path to mask file
    target_size : tuple
        Target size (height, width) for the output array
        
    Returns
    -------
    np.ndarray
        Binary mask array with shape (H, W) in uint8 format (0-255)
    """
    from PIL import Image
    mask_img = Image.open(mask_path).convert('L')
    mask_array = np.array(mask_img)
    
    from syntract_viewer.utils import resize_image_to_size
    if mask_array.shape[:2] != target_size:
        mask_array = resize_image_to_size(mask_array, target_size, is_mask=True)
    
    # Ensure binary format (0 or 255)
    mask_array = (mask_array > 127).astype(np.uint8) * 255
    return mask_array


def process_patches_inmemory(
    input_nifti: str,
    trk_file: str,
    num_patches: int = 50,
    patch_size: list = None,
    min_streamlines_per_patch: int = 0,
    voxel_size: float = 0.05,
    new_dim: tuple = None,
    use_ants: bool = False,
    ants_warp: str = None,
    ants_iwarp: str = None,
    ants_aff: str = None,
    enable_orange_blobs: bool = True,
    orange_blob_probability: float = 0.3,
    output_image_size: tuple = None,
    random_state: int = None,
    **kwargs
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Process NIfTI and TRK files to generate visualization patches and masks in-memory.
    
    This function extracts patches from tractography data and generates visualizations
    and high-density masks without saving intermediate PNG files. All processing is done
    in-memory and results are returned as numpy arrays ready for use with matplotlib.
    
    Parameters
    ----------
    input_nifti : str
        Path to input NIfTI file
    trk_file : str
        Path to TRK tractography file or directory containing TRK files.
        If directory, will randomly select TRK files for each patch.
    num_patches : int, optional
        Total number of patches to generate (default: 50)
    patch_size : list, optional
        Patch dimensions [width, height, depth] (default: [600, 1, 600])
    min_streamlines_per_patch : int, optional
        Minimum streamlines required per patch (default: 20)
    voxel_size : float, optional
        Target voxel size in mm (default: 0.05)
    new_dim : tuple, optional
        Target dimensions (X, Y, Z). Auto-calculated if not provided
    use_ants : bool, optional
        Use ANTs transformation (default: False)
    ants_warp : str, optional
        ANTs warp file path
    ants_iwarp : str, optional
        ANTs inverse warp file path
    ants_aff : str, optional
        ANTs affine file path
    enable_orange_blobs : bool, optional
        Enable orange blob injection site simulation (default: False)
    orange_blob_probability : float, optional
        Probability of applying orange blobs (0.0-1.0, default: 0.3)
    output_image_size : tuple, optional
        Output image size (height, width). Derived from patch_size if not provided
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        - images: List of RGB visualization arrays, shape (H, W, 3), dtype uint8
        - masks: List of binary mask arrays, shape (H, W), dtype uint8 (0-255)
        
    Examples
    --------
    >>> # Single TRK file
    >>> images, masks = process_patches_inmemory(
    ...     'brain.nii.gz', 
    ...     'fibers.trk',
    ...     num_patches=10,
    ...     patch_size=[512, 1, 512]
    ... )
    >>> 
    >>> # TRK directory (randomly selects files)
    >>> images, masks = process_patches_inmemory(
    ...     'brain.nii.gz',
    ...     './trk_files/',
    ...     num_patches=10,
    ...     patch_size=[512, 1, 512],
    ...     enable_orange_blobs=True
    ... )
    >>> plt.imshow(images[0])
    >>> plt.show()
    """
    
    print("="*60)
    print("IN-MEMORY PATCH PROCESSING")
    print("="*60)
    print(f"Input NIfTI: {input_nifti}")
    print(f"Input TRK: {trk_file}")
    print(f"Num patches: {num_patches}")
    
    # Import numpy at the beginning
    import numpy as np
    
    # Validate inputs
    if not os.path.exists(input_nifti):
        raise FileNotFoundError(f"NIfTI file not found: {input_nifti}")
    if not os.path.exists(trk_file):
        raise FileNotFoundError(f"TRK file/directory not found: {trk_file}")
    
    # Check if trk_file is a directory and find TRK files
    trk_files = []
    is_trk_directory = False
    if os.path.isdir(trk_file):
        is_trk_directory = True
        # Find all TRK files in the directory
        for file in os.listdir(trk_file):
            if file.endswith('.trk'):
                trk_files.append(os.path.join(trk_file, file))
        
        if not trk_files:
            raise ValueError(f"No TRK files found in directory: {trk_file}")
        
        print(f"Found {len(trk_files)} TRK files in directory")
        print(f"Will randomly select from these files for each patch")
    else:
        # Single TRK file
        trk_files = [trk_file]
        print(f"Using single TRK file")
    
    # Step 1: Setup and Validation
    # Set default patch size if not provided
    if patch_size is None:
        patch_size = [600, 1, 600]
        print(f"Using default patch size: {patch_size}")
    else:
        print(f"Using patch size: {patch_size}")
    
    # Auto-calculate dimensions if not provided
    if new_dim is None:
        print("Auto-calculating target dimensions...")
        try:
            import nibabel as nib
            
            nifti_img = nib.load(input_nifti)
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
    
    # Derive output image size from patch_size if not provided
    if output_image_size is None:
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
    
    # Convert patch_size to 3D tuple format
    if len(patch_size) == 2:
        target_patch_size = (patch_size[0], 1, patch_size[1])
    elif len(patch_size) == 3:
        target_patch_size = tuple(patch_size)
    else:
        raise ValueError(f"Invalid patch_size: {patch_size}")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="inmemory_patches_")
    print(f"Created temporary directory: {temp_dir}")
    
    images = []
    masks = []
    
    try:
        # Step 2: Extract Patches Using Patch-First Optimization
        print("\nExtracting patches using patch-first optimization...")
        from synthesis.patch_first_processing import process_patch_first_extraction
        
        # Set up random number generator for reproducible TRK selection
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState()
        
        # For directory mode, extract patches iteratively until we have exactly num_patches WITH streamlines
        if is_trk_directory:
            all_patch_details = []
            total_extracted = 0
            file_idx = 0
            patch_counter = 0
            max_attempts = num_patches * 20  # Allow up to 20x attempts to find valid patches
            attempts = 0
            
            print(f"\nExtracting {num_patches} patches with streamlines from {len(trk_files)} TRK files...")
            print("Will iteratively generate patches until we have exactly the target count of patches WITH streamlines")
            
            # Keep generating patches until we have exactly num_patches WITH streamlines
            while total_extracted < num_patches and attempts < max_attempts:
                trk_path = trk_files[file_idx % len(trk_files)]
                current_file_idx = file_idx % len(trk_files)
                
                # Calculate how many patches we still need
                patches_needed = num_patches - total_extracted
                
                # Try to extract a small batch of patches from current TRK file
                batch_size = min(patches_needed, 10)  # Process in small batches
                
                print(f"\nProcessing TRK file {current_file_idx + 1}/{len(trk_files)}: {os.path.basename(trk_path)}")
                print(f"  Need {patches_needed} more patches, trying to extract {batch_size} patches from this file...")
                
                try:
                    patch_result = process_patch_first_extraction(
                        original_nifti_path=input_nifti,
                        original_trk_path=trk_path,
                        target_voxel_size=voxel_size,
                        target_patch_size=target_patch_size,
                        target_dimensions=new_dim,
                        num_patches=batch_size,
                        output_prefix=os.path.join(temp_dir, f"patch_{patch_counter:03d}"),
                        min_streamlines_per_patch=min_streamlines_per_patch,
                        use_ants=use_ants,
                        ants_warp_path=ants_warp,
                        ants_iwarp_path=ants_iwarp,
                        ants_aff_path=ants_aff,
                        random_state=random_state + attempts if random_state else None,
                        use_gpu=True,
                    )
                    
                    if patch_result['success'] and patch_result['patches_extracted'] > 0:
                        # Only count patches that actually have streamlines
                        valid_patches = 0
                        for patch_detail in patch_result['patch_details']:
                            # Check if this patch has streamlines by looking at the TRK file
                            trk_file_path = patch_detail['files']['trk']
                            try:
                                from dipy.io.streamline import load_tractogram
                                tractogram = load_tractogram(trk_file_path, reference='same', bbox_valid_check=False)
                                if len(tractogram.streamlines) > 0:
                                    # Update patch_id to be globally unique
                                    patch_detail['patch_id'] = f"{patch_counter:03d}_{patch_detail['patch_id']}"
                                    all_patch_details.append(patch_detail)
                                    valid_patches += 1
                            except:
                                # If we can't load the TRK file, skip this patch
                                continue
                        
                        total_extracted += valid_patches
                        patch_counter += 1
                        print(f"  Successfully extracted {valid_patches} patches with streamlines (total: {total_extracted}/{num_patches})")
                        
                        # If we've reached our target, break out of the loop
                        if total_extracted >= num_patches:
                            print(f"  Target reached! Stopping extraction.")
                            break
                    else:
                        print(f"  WARNING: No patches extracted from this TRK file")
                        
                except Exception as e:
                    print(f"  ERROR: Failed to process TRK file: {e}")
                
                file_idx += 1
                attempts += 1
                
                # If we've cycled through all files and still don't have enough, continue with random selection
                if file_idx >= len(trk_files):
                    file_idx = 0  # Reset to start cycling through files again
                    print(f"  Cycled through all files, continuing with random selection...")
            
            # Create a combined result
            patch_result = {
                'success': total_extracted > 0,
                'patches_extracted': total_extracted,
                'patch_details': all_patch_details
            }
            
            if total_extracted < num_patches:
                print(f"\nWARNING: Only extracted {total_extracted} patches out of {num_patches} requested")
                print("This may be due to very sparse fiber data in the TRK files")
            else:
                print(f"\nSUCCESS: Extracted exactly {total_extracted} patches as requested!")
            
        else:
            # Single TRK file mode (original behavior)
            patch_result = process_patch_first_extraction(
                original_nifti_path=input_nifti,
                original_trk_path=trk_files[0],
                target_voxel_size=voxel_size,
                target_patch_size=target_patch_size,
                target_dimensions=new_dim,
                num_patches=num_patches,
                output_prefix=os.path.join(temp_dir, "patch"),
                min_streamlines_per_patch=min_streamlines_per_patch,
                use_ants=use_ants,
                ants_warp_path=ants_warp,
                ants_iwarp_path=ants_iwarp,
                ants_aff_path=ants_aff,
                random_state=random_state,
                use_gpu=True,
            )
        
        if not patch_result['success'] or patch_result['patches_extracted'] == 0:
            print("ERROR: No patches were extracted successfully")
            return (images, masks)
        
        print(f"\nSuccessfully extracted {patch_result['patches_extracted']} patches")
        
        # Step 3 & 4: Generate Visualizations and Masks In-Memory
        print("\nGenerating visualizations and masks...")
        
        # Ensure matplotlib is in non-interactive mode
        plt.ioff()
        
        for i, patch_detail in enumerate(patch_result['patch_details']):
            patch_id = patch_detail['patch_id']
            nifti_file = patch_detail['files']['nifti']
            trk_file_path = patch_detail['files']['trk']
            
            print(f"\nProcessing patch {patch_id}/{patch_result['patches_extracted']}...")
            
            try:
                # Verify files exist
                if not os.path.exists(nifti_file) or not os.path.exists(trk_file_path):
                    print(f"  WARNING: Patch files not found, skipping")
                    continue
                
                # Generate visualization
                print(f"  Generating visualization...")
                from syntract_viewer.core import visualize_nifti_with_trk_coronal
                
                # Random fiber percentage for high-density masks (70-100%)
                max_fiber_pct = np.random.uniform(70, 100) if random_state is None else np.random.RandomState(random_state + i).uniform(70, 100)
                
                # Generate visualization without saving to disk
                fig, axes, _ = visualize_nifti_with_trk_coronal(
                    nifti_file=nifti_file,
                    trk_file=trk_file_path,
                    output_file=None,  # Don't save to disk
                    n_slices=1,
                    save_masks=False,  # We'll generate high-density masks separately
                    use_high_density_masks=False,
                    contrast_method='clahe',
                    background_enhancement='preserve_edges',
                    cornucopia_augmentation='clean_optical',
                    tract_linewidth=1.0,
                    output_image_size=output_image_size,
                    random_state=random_state + i if random_state else None
                )
                
                # Apply custom orange blobs if enabled (your implementation)
                if enable_orange_blobs and fig is not None:
                    import random as rnd
                    
                    # Set random state for reproducibility
                    if random_state is not None:
                        rnd.seed(random_state + i)
                        np.random.seed(random_state + i)
                    
                    # Randomly decide whether to apply orange blobs
                    apply_blobs = rnd.random() < orange_blob_probability
                    
                    if apply_blobs:
                        print(f"Adding  orange injection site...")
                        
                        # Get the first (and only) axis
                        ax = axes[0] if isinstance(axes, list) else axes
                        
                        # Get image dimensions from output_image_size
                        height, width = output_image_size
                        
                        # Create dense orange injection site at random location
                        margin = int(min(width, height) * 0.1)  # Keep some margin from edges
                        center_x = np.random.randint(margin, width - margin)
                        center_y = np.random.randint(margin, height - margin)
                        injection_radius = min(width, height) * 0.05  # Slightly larger injection area
                        
                        print(f"Adding  orange injection site at ({center_x:.0f}, {center_y:.0f}), radius: {injection_radius:.0f}")
                        
                        # Add more orange streamlines for better visibility
                        num_orange_streamlines = 400  # More streamlines for better visibility
                        for stream_idx in range(num_orange_streamlines):
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
                        
                        # Add subtle orange center markers
                        ax.scatter([center_x], [center_y], c='#CC5500', s=80, alpha=0.7, zorder=30, marker='o')  # Subtle orange center
                        ax.scatter([center_x], [center_y], c='#AA3300', s=25, alpha=0.8, zorder=31, marker='o')   # Darker orange center
                        
                        print(f"ADDED  INJECTION SITE with {num_orange_streamlines} orange streamlines")
                
                # Capture figure as numpy array
                if fig is not None:
                    image_array = _capture_figure_as_array(fig, target_size=output_image_size)
                    plt.close(fig)
                else:
                    print(f"  WARNING: Figure not generated, skipping")
                    continue
                
                # Step 4: Generate High-Density Mask
                print(f"  Generating high-density mask (fiber pct: {max_fiber_pct:.1f}%)...")
                
                # Create a temporary output file path for the mask
                temp_viz_file = os.path.join(temp_dir, f"temp_viz_{patch_id}.png")
                
                # Generate high-density mask using the existing function
                # This will save the mask to a file, which we'll load
                from syntract_viewer.core import _generate_and_apply_high_density_mask_coronal
                
                # Get slice index (for coronal view, typically the middle slice)
                import nibabel as nib
                patch_img = nib.load(nifti_file)
                slice_idx = patch_img.shape[1] // 2
                
                # Generate mask (this saves to disk)
                _generate_and_apply_high_density_mask_coronal(
                    nifti_file=nifti_file,
                    trk_file=trk_file_path,
                    output_file=temp_viz_file,
                    slice_idx=slice_idx,
                    max_fiber_percentage=max_fiber_pct,
                    tract_linewidth=1.0,
                    mask_thickness=1,
                    density_threshold=0.6,
                    gaussian_sigma=2.0,
                    close_gaps=False,
                    closing_footprint_size=5,
                    label_bundles=False,
                    min_bundle_size=2000,
                    output_image_size=output_image_size,
                    static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel for high-density masks
                )
                
                # Load the generated mask
                mask_dir = os.path.dirname(temp_viz_file)
                mask_basename = os.path.splitext(os.path.basename(temp_viz_file))[0]
                mask_file = os.path.join(mask_dir, f"{mask_basename}_high_density_mask_slice{slice_idx}.png")
                
                if os.path.exists(mask_file):
                    mask_array = _load_and_resize_mask(mask_file, target_size=output_image_size)
                    
                    # Clean up temporary mask file
                    try:
                        os.remove(mask_file)
                        if os.path.exists(temp_viz_file):
                            os.remove(temp_viz_file)
                    except:
                        pass
                else:
                    print(f"  WARNING: Mask file not generated: {mask_file}")
                    continue
                
                # Add to results
                images.append(image_array)
                masks.append(mask_array)
                
                print(f"  Successfully processed patch {patch_id}")
                
                # Periodic garbage collection
                if (i + 1) % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"  ERROR: Failed to process patch {patch_id}: {e}")
                continue
        
        print(f"\n" + "="*60)
        print(f"IN-MEMORY PROCESSING COMPLETE")
        print(f"="*60)
        print(f"Successfully generated: {len(images)} images and {len(masks)} masks")
        print(f"Image shape: {images[0].shape if images else 'N/A'}")
        print(f"Mask shape: {masks[0].shape if masks else 'N/A'}")
        
    finally:
        # Step 7: Cleanup
        print(f"\nCleaning up temporary directory...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")
        
        # Force garbage collection
        gc.collect()
    
    # Step 8: Return Results
    return (images, masks)


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
    
    # Mask and Bundle parameters (unified defaults)
    mask_group = parser.add_argument_group("Mask & Bundle Detection")
    mask_group.add_argument("--save-masks", action="store_true", default=True,
                           help="Save binary masks alongside visualizations (default: True)")
    mask_group.add_argument("--use-high-density-masks", action="store_true",
                           help="Use high-density mask generation with prominent bundles (default: True)")
    mask_group.add_argument("--no-high-density-masks", action="store_true",
                           help="Disable high-density mask generation and use regular masks")
    mask_group.add_argument("--mask-thickness", type=int, default=1,
                           help="Base thickness for mask lines (default: 1, auto-scaled by output size)")
    mask_group.add_argument("--density-threshold", type=float, default=0.6,
                           help="Fiber density threshold for masking (default: 0.6, extremely aggressive filtering)")
    mask_group.add_argument("--min-bundle-size", type=int, default=2000,
                           help="Minimum size for bundle detection (default: 2000, only keeps very large prominent bundles)")
    mask_group.add_argument("--label-bundles", action="store_true",
                           help="Label individual fiber bundles with distinct colors (default: False)")
    
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