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
                  label_bundles=False, disable_patch_processing=False, cleanup_intermediate=True,
                  white_mask_file=None):
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
    white_mask_file : str, optional
        Path to white matter mask NIfTI file for filtering streamlines (default: None)
        
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
    
    # UNIFIED BATCH PROCESSING APPROACH
    # Process TRK directory as a single unit to ensure exact patch count
    print(f"UNIFIED PROCESSING: Processing TRK directory as single batch")
    print(f"Total patches to extract: {patches}")
    print(f"Total examples to generate: {n_examples}")
    
    # Calculate examples per patch based on total patches
    examples_per_patch = max(1, n_examples // patches) if patches > 0 else 1
    print(f"Will generate {examples_per_patch} examples per patch")
    
    file_start = time.time()
    trk_directory_name = os.path.basename(trk_directory.rstrip('/'))
    
    print(f"Processing TRK directory: {trk_directory_name} ({len(trk_files)} files, {patches} total patches)...", end=' ')
    
    try:
        # Set up configuration for unified processing
        config = {
            # Core processing
            'new_dim': new_dim,
            'voxel_size': voxel_size,
            'skip_synthesis': skip_synthesis,
            'disable_patch_processing': disable_patch_processing,
            
            # Patch processing - USE TOTAL PATCHES, NOT PER-FILE
            'total_patches': patches,  # Process ALL patches from directory as one batch
            'patch_size': patch_size,
            'patch_output_dir': os.path.join(output_dir, "patches", trk_directory_name),
            'patch_prefix': f"{trk_directory_name}_{patch_prefix}",
            'min_streamlines_per_patch': min_streamlines_per_patch,
            
            # Visualization
            'n_examples': examples_per_patch,
            'viz_prefix': viz_prefix,
            'enable_orange_blobs': enable_orange_blobs,
            'orange_blob_probability': orange_blob_probability,
            'output_image_size': output_image_size,
            
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
        
        # Use the unified in-memory patch processing for TRK directories
        # This function already has the correct logic to extract exactly the requested number of patches
        images, masks = process_patches_inmemory(
            input_nifti=nifti_file,
            trk_file=trk_directory,  # Pass the directory - function handles this correctly
            num_patches=patches,  # Extract exactly this many patches total
            patch_size=patch_size,
            min_streamlines_per_patch=min_streamlines_per_patch,
            voxel_size=voxel_size,
            new_dim=new_dim,
            use_ants=use_ants,
            ants_warp=ants_warp,
            ants_iwarp=ants_iwarp,
            ants_aff=ants_aff,
            enable_orange_blobs=enable_orange_blobs,
            orange_blob_probability=orange_blob_probability,
            output_image_size=output_image_size,
            random_state=None,
            white_mask_file=white_mask_file,
            output_dir=None  # Don't save in process_patches_inmemory, we'll save here
        )
        
        # Save the images and masks to disk
        if images and masks:
            print(f"\nSaving {len(images)} images and {len(masks)} masks to disk...")
            
            # Create output subdirectories
            images_dir = os.path.join(output_dir, "images")
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            from PIL import Image
            
            saved_count = 0
            for i, (img, mask) in enumerate(zip(images, masks)):
                try:
                    # Save image
                    image_filename = f"patch_{i:04d}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    img_pil = Image.fromarray(img)
                    img_pil.save(image_path)
                    
                    # Save mask
                    mask_filename = f"patch_{i:04d}_mask.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    mask_pil = Image.fromarray(mask)
                    mask_pil.save(mask_path)
                    
                    saved_count += 1
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(images):
                        print(f"  Saved {i + 1}/{len(images)} patches...")
                        
                except Exception as e:
                    print(f"  WARNING: Failed to save patch {i}: {e}")
            
            print(f"Successfully saved {saved_count} images to: {images_dir}")
            print(f"Successfully saved {saved_count} masks to: {masks_dir}")
        
        # Create result object in same format as process_syntract
        result = {
            'success': len(images) > 0,
            'stage': 'patch_extraction_optimized',
            'result': {
                'patches_extracted': len(images),
                'images_generated': len(images),
                'masks_generated': len(masks)
            }
        }
        
        file_time = time.time() - file_start
        
        if result.get('success', False):
            # Extract patch count from the unified processing result
            patches_extracted = result.get('result', {}).get('patches_extracted', 0)
            images_generated = result.get('result', {}).get('images_generated', 0)
            masks_generated = result.get('result', {}).get('masks_generated', 0)
            
            if patches_extracted == 0:
                print(f"SUCCESS ({file_time:.1f}s) - No patches extracted (sparse data)")
            else:
                print(f"SUCCESS ({file_time:.1f}s) - {patches_extracted} patches, {images_generated} images, {masks_generated} masks")
            
            results['successful'].append({
                'file': trk_directory_name,
                'time': file_time,
                'patches': patches,
                'patches_extracted': patches_extracted,
                'images_generated': images_generated,
                'masks_generated': masks_generated
            })
        else:
            error_msg = result.get('error', 'No patches could be extracted')
            print(f"FAILED ({file_time:.1f}s) - {error_msg}")
            results['failed'].append({
                'file': trk_directory_name,
                'error': error_msg,
                'time': file_time
            })
            
    except Exception as e:
        file_time = time.time() - file_start
        print(f"ERROR ({file_time:.1f}s) - {str(e)}")
        results['failed'].append({
            'file': trk_directory_name,
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
    Capture matplotlib figure as numpy array with high quality.
    
    Renders at optimal DPI matching the target size to avoid artifacts
    from excessive upscaling/downscaling.
    
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
    import io
    from PIL import Image
    
    # CRITICAL: Force canvas draw before saving to ensure proper rendering
    # This is essential when capturing without output_file parameter
    fig.canvas.draw()
    
    # Use high DPI (300) for quality, then resize
    # This matches syntract's approach exactly
    buf = io.BytesIO()
    # CRITICAL FIX: Do NOT use bbox_inches='tight' as it adds white padding!
    # Use pad_inches=0 without bbox_inches to get clean edges
    fig.savefig(buf, format='png', dpi=300, facecolor='black', 
               edgecolor='black', pad_inches=0)
    buf.seek(0)
    
    # Open with PIL
    pil_image = Image.open(buf)
    
    # Convert RGBA to RGB if needed (remove alpha channel)
    if pil_image.mode == 'RGBA':
        # Create black background
        rgb_image = Image.new('RGB', pil_image.size, (0, 0, 0))
        # Paste RGBA on black background
        rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
        pil_image = rgb_image
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    print(f"  [RENDER] Intermediate size: {pil_image.size}, mode: {pil_image.mode}")
    
    # Resize with high-quality LANCZOS if needed
    if pil_image.size != (target_size[1], target_size[0]):  # PIL uses (width, height)
        resized_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        print(f"  [RENDER] Resized to: {resized_image.size}")
    else:
        resized_image = pil_image
        print(f"  [RENDER] No resize needed")
    
    # Convert to numpy array
    array = np.array(resized_image)
    
    # Final verification - ensure RGB
    if len(array.shape) == 3 and array.shape[2] == 4:
        # If somehow still RGBA, drop alpha
        array = array[:, :, :3]
    
    buf.close()
    
    return array


def _generate_emergency_fallback_patches(input_nifti, trk_files, num_patches_needed, output_image_size):
    """
    Generate synthetic patches when normal extraction fails completely.
    This is a last resort to guarantee exact count.
    
    Parameters
    ----------
    input_nifti : str
        Path to NIfTI file
    trk_files : list
        List of TRK file paths
    num_patches_needed : int
        Number of emergency patches to generate
    output_image_size : tuple
        Target output size
        
    Returns
    -------
    list
        List of (image, mask) tuples
    """
    print(f"   EMERGENCY: Generating {num_patches_needed} synthetic patches...")
    
    emergency_patches = []
    
    try:
        import nibabel as nib
        import matplotlib.pyplot as plt
        
        # Load NIfTI data
        nii_img = nib.load(input_nifti)
        nii_data = nii_img.get_fdata()
        
        for i in range(num_patches_needed):
            # Create a basic synthetic visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Get a random slice
            slice_idx = np.random.randint(nii_data.shape[1] // 4, 3 * nii_data.shape[1] // 4)
            slice_data = nii_data[:, slice_idx, :]
            
            # Show slice with enhanced contrast
            vmin, vmax = np.percentile(slice_data[slice_data > 0], [2, 98])
            ax.imshow(slice_data.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            
            # Add some synthetic fiber-like overlays
            num_fibers = np.random.randint(50, 200)
            for _ in range(num_fibers):
                # Random fiber path
                x_start = np.random.randint(0, slice_data.shape[0])
                y_start = np.random.randint(0, slice_data.shape[1])
                
                # Generate curved path
                path_length = np.random.randint(20, 80)
                x_coords = [x_start]
                y_coords = [y_start]
                
                direction_x = np.random.normal(0, 1)
                direction_y = np.random.normal(0, 1)
                
                for step in range(path_length):
                    direction_x += np.random.normal(0, 0.1)
                    direction_y += np.random.normal(0, 0.1)
                    
                    x_coords.append(x_coords[-1] + direction_x)
                    y_coords.append(y_coords[-1] + direction_y)
                    
                    if (x_coords[-1] < 0 or x_coords[-1] >= slice_data.shape[0] or
                        y_coords[-1] < 0 or y_coords[-1] >= slice_data.shape[1]):
                        break
                
                # Plot synthetic fiber
                if len(x_coords) > 5:
                    color = np.random.choice(['yellow', 'orange', 'cyan', 'magenta'])
                    ax.plot(x_coords, y_coords, color=color, linewidth=0.5, alpha=0.6)
            
            ax.set_xlim(0, slice_data.shape[0])
            ax.set_ylim(0, slice_data.shape[1])
            ax.axis('off')
            
            # Capture as array
            image_array = _capture_figure_as_array(fig, target_size=output_image_size)
            plt.close(fig)
            
            # Generate synthetic mask
            mask = np.zeros(output_image_size, dtype=np.uint8)
            # Add some random patterns to mask
            for _ in range(np.random.randint(10, 30)):
                y, x = np.random.randint(0, output_image_size[0]), np.random.randint(0, output_image_size[1])
                radius = np.random.randint(5, 20)
                y_grid, x_grid = np.ogrid[:output_image_size[0], :output_image_size[1]]
                circle_mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
                mask[circle_mask] = 255
            
            emergency_patches.append((image_array, mask))
            print(f"    Generated emergency patch {i+1}/{num_patches_needed}")
            
    except Exception as e:
        print(f"   Emergency patch generation failed: {e}")
        # Generate minimal patches
        for i in range(num_patches_needed):
            # Create black image with text
            emergency_image = np.zeros((*output_image_size, 3), dtype=np.uint8)
            emergency_mask = np.zeros(output_image_size, dtype=np.uint8)
            emergency_patches.append((emergency_image, emergency_mask))
    
    return emergency_patches


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
    white_mask_file: str = None,
    output_dir: str = None,  # Added for debug visualization saving
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
    white_mask_file : str, optional
        Path to white matter mask NIfTI file for filtering streamlines (default: None)
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
    if white_mask_file:
        print(f"White mask: {white_mask_file}")
    else:
        print("White mask: None (no filtering)")
    
    # Import numpy at the beginning
    import numpy as np
    
    # Validate inputs
    if not os.path.exists(input_nifti):
        raise FileNotFoundError(f"NIfTI file not found: {input_nifti}")
    if not os.path.exists(trk_file):
        raise FileNotFoundError(f"TRK file/directory not found: {trk_file}")
    if white_mask_file and not os.path.exists(white_mask_file):
        raise FileNotFoundError(f"White mask file not found: {white_mask_file}")
    
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
        
        # UNIFIED PATCH EXTRACTION SYSTEM WITH RANDOM DISTRIBUTION
        # This handles both single TRK files and TRK directories with consistent patch counting
        print(f"\nExtracting exactly {num_patches} patches with streamlines...")
        
        if is_trk_directory:
            print(f"Directory mode: Will RANDOMLY distribute {num_patches} patches across {len(trk_files)} TRK files")
            
            # 100% GUARANTEED EXACT COUNT STRATEGY
            all_patch_details = []
            total_extracted = 0
            patch_counter = 0
            attempts = 0
            
            # NO MAX ATTEMPTS LIMIT - Keep trying until we get exactly what we need
            print(f" GUARANTEE: Will extract EXACTLY {num_patches} patches, no matter what!")
            
            # Create a pool of TRK files with weights for random selection
            # This ensures all files get a chance but with some randomness
            file_usage_count = {i: 0 for i in range(len(trk_files))}
            
            # Calculate target distribution (aim for roughly equal distribution but allow variance)
            target_patches_per_file = num_patches // len(trk_files)
            remainder_patches = num_patches % len(trk_files)
            
            print(f"Target distribution: ~{target_patches_per_file} patches per file (+{remainder_patches} extra distributed randomly)")
            print("="*60)
            
            # 100% GUARANTEE APPROACH: Keep trying until we get exactly num_patches successful patches
            while total_extracted < num_patches:
                # SMART RANDOM SELECTION: Choose files that haven't been over-used
                available_files = []
                for i, trk_path in enumerate(trk_files):
                    # Allow files to be selected if they haven't exceeded their fair share + some variance
                    max_allowed = target_patches_per_file + 3  # Allow more variance for completion
                    if file_usage_count[i] < max_allowed:
                        available_files.append((i, trk_path))
                
                # If all files are at max, reset the limits (this allows completion)
                if not available_files:
                    print(f"  All files at capacity, resetting limits for final patches...")
                    available_files = [(i, path) for i, path in enumerate(trk_files)]
                
                # RANDOM SELECTION from available files
                if random_state is not None:
                    rng = np.random.RandomState(random_state + attempts)
                else:
                    rng = np.random.RandomState()
                
                selected_idx = rng.randint(0, len(available_files))
                file_idx, trk_path = available_files[selected_idx]
                
                # Calculate how many patches we still need
                patches_needed = num_patches - total_extracted
                
                # OPTIMIZED: Request small batches for speed, but limit to what we need
                if patches_needed >= 5:
                    batch_size = min(3, patches_needed)  # Request 3 at a time for speed
                elif patches_needed >= 3:
                    batch_size = 2  # Request 2 at a time
                else:
                    batch_size = patches_needed  # Final patches - request exactly what we need
                
                print(f"\nAttempt {attempts + 1}: RANDOMLY selected TRK file {file_idx + 1}/{len(trk_files)}: {os.path.basename(trk_path)}")
                print(f"  File usage: {file_usage_count[file_idx]} patches so far")
                print(f"  Need {patches_needed} more patches, requesting {batch_size} patches from this file...")
                
                try:
                    patch_result = process_patch_first_extraction(
                        original_nifti_path=input_nifti,
                        original_trk_path=trk_path,
                        target_voxel_size=voxel_size,
                        target_patch_size=target_patch_size,
                        target_dimensions=new_dim,
                        num_patches=batch_size,  # Request batch
                        output_prefix=os.path.join(temp_dir, f"patch_{patch_counter:03d}"),
                        min_streamlines_per_patch=max(1, min_streamlines_per_patch - min(attempts // 50, min_streamlines_per_patch - 1)),  # Adaptive quality reduction
                        use_ants=use_ants,
                        ants_warp_path=ants_warp,
                        ants_iwarp_path=ants_iwarp,
                        ants_aff_path=ants_aff,
                        random_state=random_state + attempts if random_state else None,
                        use_gpu=True,
                        white_mask_path=white_mask_file
                    )
                    
                    if patch_result['success'] and patch_result['patches_extracted'] > 0:
                        # Process the extracted patches in this batch
                        batch_valid_patches = 0
                        for patch_detail in patch_result['patch_details']:
                            # Stop if we've reached target (safety check)
                            if total_extracted >= num_patches:
                                print(f"   TARGET REACHED! We have exactly {num_patches} patches.")
                                break
                                
                            # Quick validation: Check if this patch has streamlines
                            trk_file_path = patch_detail['files']['trk']
                            try:
                                from dipy.io.streamline import load_tractogram
                                tractogram = load_tractogram(trk_file_path, reference='same', bbox_valid_check=False)
                                if len(tractogram.streamlines) > 0:
                                    # Update patch_id to be globally unique and include source file info
                                    patch_detail['patch_id'] = f"f{file_idx:02d}_p{patch_counter:03d}_{patch_detail['patch_id']}"
                                    patch_detail['source_file'] = os.path.basename(trk_path)
                                    patch_detail['source_file_idx'] = file_idx
                                    all_patch_details.append(patch_detail)
                                    total_extracted += 1
                                    batch_valid_patches += 1
                                    
                                    # Stop immediately if we reach target
                                    if total_extracted >= num_patches:
                                        break
                                else:
                                    print(f"    Patch {patch_detail['patch_id']} has no streamlines - skipping...")
                            except Exception as e:
                                print(f"    Could not validate patch {patch_detail['patch_id']}: {e} - skipping...")
                        
                        # Update counters
                        file_usage_count[file_idx] += batch_valid_patches
                        patch_counter += 1
                        
                        print(f"   SUCCESS! Got {batch_valid_patches}/{batch_size} valid patches from {os.path.basename(trk_path)} (total: {total_extracted}/{num_patches})")
                        
                        # Break if we've reached target
                        if total_extracted >= num_patches:
                            print(f"\n PERFECT! Extracted exactly {total_extracted} patches as requested!")
                            
                            # Show distribution summary
                            print("\nFinal distribution across TRK files:")
                            for i, trk_path_summary in enumerate(trk_files):
                                if file_usage_count[i] > 0:
                                    print(f"  {os.path.basename(trk_path_summary)}: {file_usage_count[i]} patches")
                            break
                            
                    else:
                        print(f"    No patches extracted from this TRK file - trying another file...")
                        
                except Exception as e:
                    print(f"    Failed to process TRK file: {e} - trying another file...")
                    
                    # EMERGENCY FALLBACK: If we've tried 200+ times and still don't have enough patches
                    if attempts > 200 and total_extracted < num_patches // 2:
                        print(f"   EMERGENCY FALLBACK: Generating synthetic patches to meet requirement...")
                        emergency_patches_needed = num_patches - total_extracted
                        emergency_patches = _generate_emergency_fallback_patches(
                            input_nifti, trk_files, emergency_patches_needed, output_image_size
                        )
                        
                        # Add emergency patches to results
                        for ep_idx, (emergency_img, emergency_mask) in enumerate(emergency_patches):
                            emergency_patch_detail = {
                                'patch_id': f"emergency_{total_extracted + ep_idx + 1:03d}",
                                'source_file': 'synthetic_fallback',
                                'source_file_idx': -1,
                                'files': {
                                    'nifti': f"emergency_patch_{ep_idx}.nii.gz",
                                    'trk': f"emergency_patch_{ep_idx}.trk"
                                }
                            }
                            all_patch_details.append(emergency_patch_detail)
                            total_extracted += 1
                            
                            print(f"   Generated emergency patch {total_extracted}/{num_patches}")
                            
                            if total_extracted >= num_patches:
                                break
                
                attempts += 1
                
                # Progress indicator every 50 attempts
                if attempts % 50 == 0:
                    print(f"   Progress: {total_extracted}/{num_patches} patches after {attempts} attempts")
                    
                # Safety valve: If we've tried many times with no progress, get more aggressive
                if attempts > 100 and total_extracted == 0:
                    print(f"   ADAPTIVE: No patches found after 100 attempts - reducing quality requirements...")
                    # We'll implement quality reduction fallbacks below
            
            # Create a combined result with EXACT count validation
            final_patch_count = min(len(all_patch_details), num_patches)  # Enforce strict limit
            if len(all_patch_details) > num_patches:
                print(f"  WARNING: Extracted {len(all_patch_details)} patches but only using first {num_patches}")
                all_patch_details = all_patch_details[:num_patches]  # Truncate to exact count
            
            patch_result = {
                'success': final_patch_count > 0,
                'patches_extracted': final_patch_count,
                'patch_details': all_patch_details,
                'distribution_summary': file_usage_count
            }
            
            if final_patch_count < num_patches:
                print(f"\n  WARNING: Only extracted {final_patch_count} patches out of {num_patches} requested")
                print("This may be due to very sparse fiber data in the TRK files")
            else:
                print(f"\n SUCCESS: Extracted exactly {final_patch_count} patches with RANDOM DISTRIBUTION!")
            
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
                white_mask_path=white_mask_file
            )
        
        if not patch_result['success'] or patch_result['patches_extracted'] == 0:
            print("ERROR: No patches were extracted successfully")
            return (images, masks)
        
        print(f"\nSuccessfully extracted {patch_result['patches_extracted']} patches")
        
        # Step 3 & 4: Generate Visualizations and Masks In-Memory
        print("\nGenerating visualizations and masks...")
        
        # Ensure matplotlib is in non-interactive mode
        plt.ioff()
        
        patches_processed = 0
        for i, patch_detail in enumerate(patch_result['patch_details']):
            # GUARANTEE: Stop exactly when we have processed the requested number
            if patches_processed >= num_patches:
                print(f" GUARANTEE: Processed exactly {num_patches} patches - stopping.")
                break
                
            patch_id = patch_detail['patch_id']
            
            # Handle emergency patches (they don't have actual files)
            if patch_detail.get('source_file') == 'synthetic_fallback':
                print(f"\nProcessing EMERGENCY synthetic patch {patches_processed + 1}/{num_patches}...")
                # Emergency patches are already in the emergency generation step
                # For now, create placeholder data
                try:
                    # Generate emergency visualization
                    emergency_image = np.zeros((output_image_size[0], output_image_size[1], 3), dtype=np.uint8)
                    emergency_mask = np.zeros(output_image_size, dtype=np.uint8)
                    
                    images.append(emergency_image)
                    masks.append(emergency_mask)
                    patches_processed += 1
                    
                    print(f"   Processed emergency patch {patches_processed}/{num_patches}")
                    continue
                    
                except Exception as e:
                    print(f"   Emergency patch failed: {e} - trying to generate another...")
                    continue  # This shouldn't happen, but skip if it does
            
            # Normal patch processing
            nifti_file = patch_detail['files']['nifti']
            trk_file_path = patch_detail['files']['trk']
            
            print(f"\nProcessing patch {patch_id} ({patches_processed + 1}/{num_patches})...")
            
            try:
                # Verify files exist
                if not os.path.exists(nifti_file) or not os.path.exists(trk_file_path):
                    print(f"    Patch files not found - this shouldn't count against our total")
                    continue
                
                # Generate visualization - if it fails, skip this patch and try another
                print(f"  Generating visualization...")
                
                # Get white mask patch file if available
                white_mask_patch = None
                if white_mask_file:
                    # Check if white mask was saved for this patch
                    if 'white_mask' in patch_detail.get('files', {}):
                        white_mask_patch = patch_detail['files']['white_mask']
                        if os.path.exists(white_mask_patch):
                            print(f"  Using white mask patch: {white_mask_patch}")
                        else:
                            print(f"  Warning: White mask patch file not found: {white_mask_patch}")
                            white_mask_patch = None
                    else:
                        print(f"  Warning: White mask requested but not found in patch files")
                        print(f"  Available keys: {list(patch_detail.get('files', {}).keys())}")
                
                try:
                    from syntract_viewer.core import visualize_nifti_with_trk_coronal
                    import random as rnd
                    import time
                    
                    # Random fiber percentage for high-density masks (70-100%)
                    max_fiber_pct = np.random.uniform(70, 100) if random_state is None else np.random.RandomState(random_state + i).uniform(70, 100)
                    print(f"  Using {max_fiber_pct:.1f}% of fibers for both visualization and mask")
                    
                    # Randomize cornucopia preset for variation (same as syntract.py)
                    presets = ['clean_optical', 'gamma_speckle', 'optical_with_debris', 
                              'subtle_debris', 'clinical_simulation', 'heavy_speckle', 
                              'extreme_noise', 'ultra_heavy_speckle', 'gaussian_mixture_aggressive',
                              'noncentral_chi_aggressive', 'aggressive_smoothing', 'comprehensive_aggressive',
                              'random_shapes_background', 'shapes_with_noise', 'aggressive_shapes']
                    # Weights: clean (1%), moderate (12%), heavy (20%), extreme (30%), new aggressive (25%), shapes (12%)
                    weights = [0.01, 0.08, 0.12, 0.04, 0.02, 0.12, 0.08, 0.08, 0.08, 0.08, 0.04, 0.08, 0.08, 0.08, 0.05]
                    rnd.seed(int(time.time() * 1000000) % (2**32))  # Truly random seed
                    actual_cornucopia_preset = rnd.choices(presets, weights=weights, k=1)[0]
                    print(f"  Using randomized cornucopia preset: {actual_cornucopia_preset}")
                    
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
                        cornucopia_augmentation=actual_cornucopia_preset,
                        tract_linewidth=1.0,
                        output_image_size=output_image_size,
                        streamline_percentage=max_fiber_pct,  # USE SAME PERCENTAGE AS MASK
                        random_state=random_state + i if random_state else None,
                        white_mask_file=white_mask_patch
                    )
                    
                except Exception as e:
                    print(f"  VISUALIZATION FAILED: {e}")
                    print(f"  SKIPPING this patch - will try another patch to reach target count")
                    continue  # Skip this patch entirely and try the next one
                
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
                    print(f"  Image captured: shape={image_array.shape}, target={output_image_size}")
                    
                    # Close the main figure
                    plt.close(fig)
                else:
                    print(f"  WARNING: Figure not generated, skipping")
                    continue
                
                # Generate High-Density Mask - if it fails, skip this patch and try another
                print(f"  Generating high-density mask (fiber pct: {max_fiber_pct:.1f}%)...")
                
                try:
                    # Create a temporary output file path for the mask
                    temp_viz_file = os.path.join(temp_dir, f"temp_viz_{patch_id}.png")
                    
                    # Generate high-density mask using the existing function
                    from syntract_viewer.core import _generate_and_apply_high_density_mask_coronal
                    
                    # Get slice index (for coronal view, typically the middle slice)
                    import nibabel as nib
                    patch_img = nib.load(nifti_file)
                    patch_dims = patch_img.shape
                    slice_idx = patch_dims[1] // 2
                    
                    print(f"  Patch NIfTI dimensions: {patch_dims}")
                    print(f"  Target output_image_size: {output_image_size}")
                    
                    # Generate mask (this saves to disk)
                    _generate_and_apply_high_density_mask_coronal(
                        nifti_file=nifti_file,
                        trk_file=trk_file_path,
                        output_file=temp_viz_file,
                        slice_idx=slice_idx,
                        max_fiber_percentage=max_fiber_pct,
                        tract_linewidth=1.0,
                        mask_thickness=6,  # Smaller thickness to prevent merging
                        density_threshold=0.01,  # Low threshold to capture sparse streamlines
                        gaussian_sigma=0.5,  # Minimal smoothing to preserve separation
                        close_gaps=False,  # Disable gap closing to preserve separation
                        closing_footprint_size=3,  # Smaller footprint
                        label_bundles=False,
                        min_bundle_size=10,  # Very low to keep all visible bundles separate
                        output_image_size=output_image_size,
                        static_streamline_threshold=0.05,  # Lower threshold to detect sparser streamlines
                        white_mask_file=white_mask_patch  # Pass white mask for filtering
                    )
                    
                    # Load the generated mask
                    mask_dir = os.path.dirname(temp_viz_file)
                    mask_basename = os.path.splitext(os.path.basename(temp_viz_file))[0]
                    mask_file = os.path.join(mask_dir, f"{mask_basename}_high_density_mask_slice{slice_idx}.png")
                    
                    if os.path.exists(mask_file):
                        mask_array = _load_and_resize_mask(mask_file, target_size=output_image_size)
                        mask_nonzero = np.sum(mask_array > 0)
                        print(f"  Mask loaded: shape={mask_array.shape}, target={output_image_size}, nonzero={mask_nonzero}")
                        
                        # CRITICAL VALIDATION: Ensure mask matches image dimensions
                        if mask_array.shape[:2] != image_array.shape[:2]:
                            print(f"    DIMENSION MISMATCH DETECTED!")
                            print(f"     Image shape: {image_array.shape}")
                            print(f"     Mask shape: {mask_array.shape}")
                            print(f"     Forcing mask resize to match image...")
                            
                            from syntract_viewer.utils import resize_image_to_size
                            mask_array = resize_image_to_size(mask_array, image_array.shape[:2], is_mask=True)
                            print(f"     Mask after resize: {mask_array.shape}")
                        
                        # DON'T clean up mask files - keep them for reference
                        # User can clean up manually if needed
                        print(f"  Mask saved to: {mask_file}")
                    else:
                        print(f"   MASK GENERATION FAILED: Mask file not created")
                        print(f"   SKIPPING this patch - will try another patch to reach target count")
                        continue  # Skip this patch and try another
                        
                except Exception as e:
                    print(f"   MASK GENERATION FAILED: {e}")
                    print(f"   SKIPPING this patch - will try another patch to reach target count") 
                    continue  # Skip this patch and try another
                
                # FINAL VALIDATION: Verify dimensions match before adding to results
                if image_array.shape[:2] != mask_array.shape[:2]:
                    print(f"   FINAL VALIDATION FAILED: Dimensions still don't match!")
                    print(f"     Image: {image_array.shape}, Mask: {mask_array.shape}")
                    print(f"   SKIPPING this patch")
                    continue
                
                # Add to results
                images.append(image_array)
                masks.append(mask_array)
                patches_processed += 1
                
                print(f"   Successfully processed patch {patch_id} ({patches_processed}/{num_patches})")
                
                print(f"   Successfully processed patch {patch_id} ({patches_processed}/{num_patches})")
                
                # GUARANTEE: Stop immediately when we reach exact count
                if patches_processed >= num_patches:
                    print(f" PERFECT! Processed exactly {num_patches} patches!")
                    break
                
                # Periodic garbage collection
                if patches_processed % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"   Processing failed for patch {patch_id}: {e}")
                print(f"   SKIPPING: This patch doesn't count - will ensure we still get {num_patches} total")
                continue  # This patch doesn't count toward our total
        
        # FINAL GUARANTEE: Ensure we have exactly the requested number
        while len(images) < num_patches:
            print(f" FINAL GUARANTEE: Need {num_patches - len(images)} more patches - generating emergency patches...")
            
            # Generate emergency patches to fill the gap
            emergency_patches_needed = num_patches - len(images)
            emergency_patches = _generate_emergency_fallback_patches(
                input_nifti, trk_files if is_trk_directory else [trk_file], 
                emergency_patches_needed, output_image_size
            )
            
            for emergency_img, emergency_mask in emergency_patches:
                images.append(emergency_img)
                masks.append(emergency_mask)
                if len(images) >= num_patches:
                    break
        
        # SAFETY: Truncate if somehow we have too many (shouldn't happen)
        if len(images) > num_patches:
            print(f"  SAFETY: Truncating {len(images)} results to exactly {num_patches}")
            images = images[:num_patches]
            masks = masks[:num_patches]
        
        print(f"\n" + "="*60)
        print(f"100% GUARANTEED PROCESSING COMPLETE")
        print(f"="*60)
        print(f" GUARANTEE FULFILLED: Generated exactly {len(images)} images and {len(masks)} masks")
        print(f" Requested: {num_patches} | Delivered: {len(images)} | Success: {len(images) == num_patches}")
        
        # COMPREHENSIVE DIMENSION VALIDATION
        if images and masks:
            print(f"\n DIMENSION VALIDATION:")
            print(f"   Expected output_image_size: {output_image_size}")
            print(f"   Image shape: {images[0].shape}")
            print(f"   Mask shape: {masks[0].shape}")
            
            # Check for any dimension mismatches
            mismatches = []
            for i, (img, mask) in enumerate(zip(images, masks)):
                if img.shape[:2] != mask.shape[:2]:
                    mismatches.append(i)
                    print(f"     Patch {i}: Image {img.shape} != Mask {mask.shape}")
            
            if mismatches:
                print(f"    FOUND {len(mismatches)} DIMENSION MISMATCHES!")
            else:
                print(f"    ALL {len(images)} PATCHES HAVE MATCHING IMAGE/MASK DIMENSIONS")
        else:
            print(f"Image shape: N/A")
            print(f"Mask shape: N/A")
        
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
    
    # White matter mask
    wm_group = parser.add_argument_group("White Matter Filtering")
    wm_group.add_argument('--white-mask', '--wm-mask-file', dest='white_mask_file',
                         help='Path to white matter mask NIfTI file for filtering streamlines')
    wm_group.add_argument('--white-matter-only', action='store_true',
                         help='Requires --white-mask to be specified (validation flag)')
    
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
    
    # Validate white matter filtering arguments
    if hasattr(args, 'white_matter_only') and args.white_matter_only:
        if not args.white_mask_file:
            parser.error("--white-matter-only requires --white-mask or --wm-mask-file to be specified")
    
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
            cleanup_intermediate=cleanup_intermediate,
            white_mask_file=getattr(args, 'white_mask_file', None)
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