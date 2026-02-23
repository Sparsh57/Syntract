#!/usr/bin/env python
"""
Generate 3D NIfTI volumes with streamlines overlaid.
Applies dark field microscopy-style rendering with subtle fiber visualization.
"""

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk
from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage import zoom
from tqdm import tqdm
import os
import sys
import gc

# GPU support with fallback
try:
    from synthesis.gpu_utils import try_gpu_import
    gpu_result = try_gpu_import()
    xp = gpu_result['xp']
    use_gpu = gpu_result['cupy_available']
    if use_gpu:
        import cupyx.scipy.ndimage as gpu_ndimage
        print(f"GPU acceleration enabled for 3D rendering ({gpu_result['gpu_name']})")
    else:
        from scipy.ndimage import gaussian_filter
        print("GPU not available, using CPU for 3D rendering")
except ImportError:
    xp = np
    use_gpu = False
    from scipy.ndimage import gaussian_filter
    print("GPU utils not found, using CPU for 3D rendering")

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from syntract_viewer.volumetric_3d import process_volume_full_3d
except ImportError:
    from volumetric_3d import process_volume_full_3d


def create_3d_volume_with_streamlines(nifti_file, trk_file, output_file,
                                       slice_range=None, orientation='coronal',
                                       white_mask_path=None, contrast_method='clahe',
                                       gamma=2.2, scaling_factor=40.0,
                                       fiber_intensity_min=15.0, fiber_intensity_max=25.0,
                                       tissue_threshold=2.0,
                                       min_bundle_size=None,
                                       use_cornucopia_3d=False,
                                       cornucopia_allowed_presets=None,
                                       cornucopia_prob=0.9,
                                       random_state=None,
                                       save_mask=True):
    """
    Create 3D NIfTI volume with streamlines rendered as subtle fibers on dark tissue.
    
    Uses TRUE 3D CLAHE - processes entire volume as single unit, no tiling artifacts.
    Mask generation creates THIN, ACCURATE streamline representations (ground truth),
    not thick segmentation training masks like the 2D pipeline.
    
    Parameters
    ----------
    nifti_file : str
        Path to input NIfTI file
    trk_file : str
        Path to TRK streamline file
    output_file : str
        Path for output 3D NIfTI volume
    slice_range : range, optional
        Slice indices to process (default: all slices)
    orientation : str
        Orientation: 'coronal', 'axial', or 'sagittal'
    white_mask_path : str, optional
        Path to white matter mask for filtering streamlines
    contrast_method : str
        Contrast enhancement: 'clahe' or 'none'
    gamma : float
        Gamma correction for dark field appearance (default: 2.2)
    scaling_factor : float
        Output intensity scaling (default: 40.0 for dark field appearance - dark tissue, bright fibers)
    fiber_intensity_min : float
        Minimum streamline intensity (default: 15.0 for bright white fibers)
    fiber_intensity_max : float
        Maximum streamline intensity (default: 25.0 for bright white fibers)
    tissue_threshold : float
        Minimum tissue intensity for streamline rendering (default: 2.0)
    min_bundle_size : int, optional
        Minimum bundle size for filtering
    use_cornucopia_3d : bool
        Enable cornucopia 3D augmentation with aggressive presets (default: False)
    cornucopia_allowed_presets : list, optional
        List of allowed presets (default: 4 aggressive presets)
    cornucopia_prob : float
        Probability of applying cornucopia (default: 0.9)
    random_state : int, optional
        Random seed for reproducibility
    save_mask : bool
        If True, save fiber mask as separate NIfTI file (default: True)
        Mask uses 3D Gaussian smoothing (sigma=2.0) and binary threshold (0.05)
    
    Returns
    -------
    str
        Path to saved 3D NIfTI volume file
    """
    # Load NIfTI with memory-mapped mode for large files
    # This prevents loading entire volume into memory at once
    nii_img = nib.load(nifti_file, mmap=True)
    nii_img = nib.as_closest_canonical(nii_img)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape
    
    print(f"Input NIfTI dimensions: {dims}")
    
    # Load white matter mask if provided and resample to match data (following 2D methodology)
    white_mask_data = None
    if white_mask_path and os.path.exists(white_mask_path):
        try:
            print(f"Loading white matter mask: {white_mask_path}")
            white_mask_img = nib.load(white_mask_path, mmap=True)
            white_mask_img = nib.as_closest_canonical(white_mask_img)
            white_mask_orig = white_mask_img.get_fdata()
            
            # Remove extra dimensions
            while white_mask_orig.ndim > 3:
                white_mask_orig = np.squeeze(white_mask_orig, axis=-1)
            
            print(f"  Original white mask shape: {white_mask_orig.shape}")
            print(f"  Target NIfTI shape: {dims}")
            
            # Resample to match NIfTI dimensions if needed (following patch_first_processing.py approach)
            if white_mask_orig.shape != dims:
                print(f"  Resampling white mask to match data dimensions...")
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                zoom_factors = np.array(dims) / np.array(white_mask_orig.shape)
                print(f"  Zoom factors: {zoom_factors}")
                
                # Resample using nearest neighbor to preserve binary mask values (order=0)
                white_mask_resampled = zoom(white_mask_orig, zoom_factors, order=0)
                print(f"  Resampled white mask shape: {white_mask_resampled.shape}")
                
                # Convert to binary mask
                white_mask_data = (white_mask_resampled > 0.5).astype(np.uint8)
            else:
                # Dimensions match - use directly
                white_mask_data = (white_mask_orig > 0.5).astype(np.uint8)
            
            print(f"  White matter mask ready: {np.count_nonzero(white_mask_data)} / {white_mask_data.size} voxels ({100*np.count_nonzero(white_mask_data)/white_mask_data.size:.1f}%)")
        except Exception as e:
            print(f"Warning: Could not load/resample white matter mask: {e}")
            import traceback
            traceback.print_exc()
            white_mask_data = None
    
    # Load streamlines
    try:
        tractogram = load_trk(trk_file)
        streamlines = tractogram.streamlines
        print(f"Loaded {len(streamlines)} streamlines")
        
        # Transform to voxel space
        trk_affine = tractogram.affine
        affine_diff = np.abs(trk_affine - nii_img.affine).max()
        if affine_diff > 0.1:
            print("Pre-registered TRK detected, using TRK affine")
            affine_inv = np.linalg.inv(trk_affine)
        else:
            affine_inv = np.linalg.inv(nii_img.affine)
        
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        # Apply bundle size filtering if specified
        if min_bundle_size and len(streamlines_voxel) > min_bundle_size:
            print(f"Note: Total streamlines ({len(streamlines_voxel)}) exceeds min_bundle_size ({min_bundle_size})")
            print(f"      Bundle filtering not applied (would need clustering implementation)")
        
    except Exception as e:
        print(f"Error loading streamlines: {e}")
        return
    
    # Determine slice range
    if slice_range is None:
        if orientation == 'coronal':
            slice_range = range(0, dims[1])
        elif orientation == 'axial':
            slice_range = range(0, dims[2])
        else:  # sagittal
            slice_range = range(0, dims[0])
    
    num_slices = len(slice_range)
    print(f"Processing {num_slices} {orientation} slices from {min(slice_range)} to {max(slice_range)}")
    
    # Create output volume
    if orientation == 'coronal':
        volume_shape = (dims[0], num_slices, dims[2])
    elif orientation == 'axial':
        volume_shape = (dims[0], dims[1], num_slices)
    else:  # sagittal
        volume_shape = (num_slices, dims[1], dims[2])
    
    # ==================================================================
    # PHASE 1: TRUE 3D VOLUMETRIC TISSUE PROCESSING
    # ==================================================================
    # Process the ENTIRE 3D volume at once to eliminate slice boundaries
    print(f"\nPhase 1: Processing full 3D tissue volume (no slice artifacts)...")
    
    # Extract the relevant portion of the 3D volume
    if orientation == 'coronal':
        volume_data = nii_data[:, list(slice_range), :].copy()
    elif orientation == 'axial':
        volume_data = nii_data[:, :, list(slice_range)].copy()
    else:  # sagittal
        volume_data = nii_data[list(slice_range), :, :].copy()
    
    print(f"  Extracted {orientation} volume: {volume_data.shape}")
    
    # Apply TRUE 3D CLAHE - processes ENTIRE volume as single unit
    # Uses GLOBAL mode (single histogram) for maximum smoothness
    use_clahe = (contrast_method == 'clahe' or contrast_method is None)
    
    # Set default cornucopia presets (only the 4 aggressive ones)
    if cornucopia_allowed_presets is None and use_cornucopia_3d:
        cornucopia_allowed_presets = [
            'extreme_noise',
            'random_shapes_background',
            'comprehensive_aggressive',
            'ultra_heavy_speckle'
        ]
    
    output_volume = process_volume_full_3d(
        volume_data,
        use_clahe=use_clahe,                  # TRUE 3D CLAHE
        clahe_adaptive=False,                 # GLOBAL: entire volume = single tile (no boundaries!)
        clahe_clip_limit=0.01,                # Standard CLAHE clip limit
        add_texture=use_cornucopia_3d,         # Ultra-smooth 3D texture
        texture_intensity=0.02,               # Subtle texture
        texture_sigma=8.0,                    # Heavy smoothing = no patterns
        gamma=gamma,
        scaling_factor=scaling_factor,
        use_cornucopia=use_cornucopia_3d,
        cornucopia_preset=None,               # Random selection from allowed list
        cornucopia_allowed_presets=cornucopia_allowed_presets,
        cornucopia_prob=cornucopia_prob,
        random_state=random_state
    )
    
    # ==================================================================
    # PHASE 2: RENDER STREAMLINES IN TRUE 3D SPACE
    # ==================================================================
    print(f"\nPhase 2: Rendering streamlines in 3D space...")
    
    # Calculate slice offset for coordinate mapping
    slice_start = min(slice_range)
    
    # Helper: 3D line drawing between two 3D points
    def draw_line_3d(p0, p1, intensity, volume, tissue_threshold, mask_volume=None):
        """Draw 3D line between two points using 3D Bresenham"""
        x0, y0, z0 = int(np.round(p0[0])), int(np.round(p0[1])), int(np.round(p0[2]))
        x1, y1, z1 = int(np.round(p1[0])), int(np.round(p1[1])), int(np.round(p1[2]))
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1
        
        # Helper to update voxel
        def update_voxel(vx, vy, vz):
            if 0 <= vx < volume.shape[0] and 0 <= vy < volume.shape[1] and 0 <= vz < volume.shape[2]:
                if volume[vx, vy, vz] >= tissue_threshold:
                    volume[vx, vy, vz] = min(255.0, volume[vx, vy, vz] + intensity)
                if mask_volume is not None:
                    mask_volume[vx, vy, vz] += 1.0

        # Driving axis
        if dx >= dy and dx >= dz:  # X-axis is driving
            p1_d, p2_d = dy, dz
            p1_inc, p2_inc = ys, zs
            d_axis = dx
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    y += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    z += p2_inc
                    p2_err -= d_axis
                
                x += xs
                
        elif dy >= dx and dy >= dz:  # Y-axis is driving
            p1_d, p2_d = dx, dz
            p1_inc, p2_inc = xs, zs
            d_axis = dy
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    x += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    z += p2_inc
                    p2_err -= d_axis
                
                y += ys
                
        else:  # Z-axis is driving
            p1_d, p2_d = dx, dy
            p1_inc, p2_inc = xs, ys
            d_axis = dz
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    x += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    y += p2_inc
                    p2_err -= d_axis
                
                z += zs
    
    # Render all streamlines in 3D
    fiber_intensity = (fiber_intensity_max + fiber_intensity_min) / 2.0
    streamlines_rendered = 0
    
    # Initialize mask accumulator if needed
    mask_accumulator = xp.zeros(output_volume.shape, dtype=xp.float32) if save_mask else None
    
    # Transfer to GPU if available
    if use_gpu:
        output_volume_gpu = xp.asarray(output_volume)
        if white_mask_data is not None:
            white_mask_gpu = xp.asarray(white_mask_data)
        else:
            white_mask_gpu = None
    
    for sl in tqdm(streamlines_voxel, desc="Rendering streamlines"):
        # Map streamlines to output volume coordinates
        sl_mapped = sl.copy()
        
        # Adjust coordinates based on orientation and slice offset
        if orientation == 'coronal':
            # Y dimension is sliced, adjust Y coordinates
            sl_mapped[:, 1] = sl[:, 1] - slice_start
        elif orientation == 'axial':
            # Z dimension is sliced, adjust Z coordinates  
            sl_mapped[:, 2] = sl[:, 2] - slice_start
        else:  # sagittal
            # X dimension is sliced, adjust X coordinates
            sl_mapped[:, 0] = sl[:, 0] - slice_start
        
        # Apply z-flip for correct alignment
        sl_mapped[:, 2] = (dims[2] if orientation != 'axial' else num_slices) - sl_mapped[:, 2] - 1
        
        # Draw lines between consecutive points in 3D
        for i in range(len(sl_mapped) - 1):
            p0 = sl_mapped[i]
            p1 = sl_mapped[i + 1]
            
            # White matter mask check for both endpoints (use original coordinates)
            if white_mask_data is not None:
                x0, y0, z0 = int(np.round(np.clip(sl[i, 0], 0, dims[0]-1))), \
                             int(np.round(np.clip(sl[i, 1], 0, dims[1]-1))), \
                             int(np.round(np.clip(sl[i, 2], 0, dims[2]-1)))
                x1, y1, z1 = int(np.round(np.clip(sl[i+1, 0], 0, dims[0]-1))), \
                             int(np.round(np.clip(sl[i+1, 1], 0, dims[1]-1))), \
                             int(np.round(np.clip(sl[i+1, 2], 0, dims[2]-1)))
                if (white_mask_data[x0, y0, z0] < 0.5 or white_mask_data[x1, y1, z1] < 0.5):
                    continue
            
            # Clip to output volume bounds
            p0_clipped = np.clip(p0, [0, 0, 0], [output_volume.shape[0]-1, output_volume.shape[1]-1, output_volume.shape[2]-1])
            p1_clipped = np.clip(p1, [0, 0, 0], [output_volume.shape[0]-1, output_volume.shape[1]-1, output_volume.shape[2]-1])
            
            # Draw 3D line (use GPU volume if available)
            if use_gpu:
                draw_line_3d(p0_clipped, p1_clipped, fiber_intensity, output_volume_gpu, tissue_threshold, mask_accumulator)
            else:
                draw_line_3d(p0_clipped, p1_clipped, fiber_intensity, output_volume, tissue_threshold, mask_accumulator)
        
        streamlines_rendered += 1
    
    print(f"Rendered {streamlines_rendered} streamlines in 3D volume")
    
    # Transfer back from GPU if used
    if use_gpu:
        output_volume = xp.asnumpy(output_volume_gpu)
    
    # No post-processing needed - artifacts eliminated in Phase 1 tissue processing
    # Streamlines remain perfectly sharp
    
    # Save output
    output_nii = nib.Nifti1Image(output_volume, affine=nii_img.affine)
    nib.save(output_nii, output_file)
    print(f"\nSaved 3D volume: {output_file}")
    print(f"  Shape: {output_volume.shape}")
    print(f"  Value range: [{output_volume.min():.2f}, {output_volume.max():.2f}]")
    
    # ==================================================================
    # GENERATE AND SAVE FIBER MASK (TRUE 3D GENERATION)
    # ==================================================================
    if save_mask and mask_accumulator is not None:
        print(f"\nPhase 3: Generating TRUE 3D fiber mask (isotropic consistency)...")
        
        # Apply 3D Gaussian smoothing to create flowing structures
        # sigma=2.0 matches the user-approved "flowing" look, but applied isotropically
        print("  Applying 3D Gaussian smoothing (sigma=2.0) for flowing connectivity...")
        if use_gpu:
            mask_density_smooth = gpu_ndimage.gaussian_filter(mask_accumulator, sigma=2.0)
            mask_density_smooth = xp.asnumpy(mask_density_smooth)
        else:
            mask_density_smooth = gaussian_filter(mask_accumulator, sigma=2.0)
        
        # Normalize for consistent thresholding (0.0 to 1.0)
        # This handles varying bundle densities gracefully
        max_val = mask_density_smooth.max()
        if max_val > 0:
            mask_density_smooth /= max_val
            
        # Threshold to create binary mask
        # 0.05 is the calibrated threshold for "thin but connected" flow
        binary_threshold = 0.05
        print(f"  Thresholding normalized density map at {binary_threshold}...")
        
        mask_volume = (mask_density_smooth > binary_threshold).astype(np.uint8)
        
        # Save mask
        mask_file = output_file.replace('.nii.gz', '_mask.nii.gz')
        mask_nii = nib.Nifti1Image(mask_volume, affine=nii_img.affine)
        nib.save(mask_nii, mask_file)
        
        mask_voxels = np.count_nonzero(mask_volume)
        mask_percentage = 100 * mask_voxels / mask_volume.size
        print(f"\nSaved fiber mask: {mask_file}")
        print(f"  Shape: {mask_volume.shape}")
        print(f"  Mask coverage: {mask_voxels} voxels ({mask_percentage:.2f}%)")
    
    # Explicit memory cleanup to ensure resources are released
    if use_gpu:
        # Free GPU memory
        del mask_accumulator
        if 'output_volume_gpu' in locals():
            del output_volume_gpu
        if 'white_mask_gpu' in locals():
            del white_mask_gpu
        xp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti', required=True)
    parser.add_argument('--trk', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--orientation', default='coronal', choices=['coronal', 'axial', 'sagittal'])
    
    args = parser.parse_args()
    
    # Determine slice range
    nii = nib.load(args.nifti)
    dims = nii.shape
    
    if args.end is None:
        if args.orientation == 'coronal':
            args.end = dims[1]
        elif args.orientation == 'axial':
            args.end = dims[2]
        else:
            args.end = dims[0]
    
    slice_range = range(args.start, args.end)
    
    create_3d_volume_with_streamlines(
        nifti_file=args.nifti,
        trk_file=args.trk,
        output_file=args.output,
        slice_range=slice_range,
        orientation=args.orientation
    )
