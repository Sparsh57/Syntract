#!/usr/bin/env python
"""
Generate 3D NIfTI volumes with streamlines overlaid.
Applies dark field microscopy-style rendering with subtle fiber visualization.
"""

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk
from dipy.tracking.streamline import transform_streamlines
from tqdm import tqdm
import os
import sys
import random
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from syntract_viewer.core import visualize_nifti_with_trk_coronal
    from syntract_viewer.contrast import apply_comprehensive_slice_processing
    from syntract_viewer.effects import apply_blockface_preserving_dark_field_effect
except ImportError:
    from core import visualize_nifti_with_trk_coronal
    from contrast import apply_comprehensive_slice_processing
    from effects import apply_blockface_preserving_dark_field_effect


def create_3d_volume_with_streamlines(nifti_file, trk_file, output_file,
                                       slice_range=None, orientation='coronal',
                                       save_2d_images=False, output_dir='output_slices',
                                       white_mask_path=None, contrast_method='clahe',
                                       background_enhancement=None, cornucopia_preset=None,
                                       gamma=2.2, scaling_factor=25.0,
                                       fiber_intensity_min=3.0, fiber_intensity_max=8.0,
                                       distance_threshold=1.5, tissue_threshold=2.0,
                                       min_bundle_size=None, density_threshold=None):
    """
    Create 3D NIfTI volume with streamlines rendered as subtle fibers on dark tissue.
    
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
    save_2d_images : bool
        If True, also generate 2D reference images using core.py
    output_dir : str
        Directory for 2D reference images
    white_mask_path : str, optional
        Path to white matter mask for filtering streamlines
    contrast_method : str
        Contrast enhancement method: 'clahe', 'histogram_equalization', 'none' (default: 'clahe')
    background_enhancement : str, optional
        Background enhancement style: 'preserve_edges', 'smooth', etc. (default: None)
    cornucopia_preset : str, optional
        Cornucopia augmentation preset for background artifacts/noise patterns
        Options: 'clean_optical', 'gamma_speckle', 'heavy_speckle', etc. (default: None)
    gamma : float
        Gamma correction for darkening bright tissues (default: 2.2, range: 1.0-3.0)
    scaling_factor : float
        Output brightness scaling (default: 25.0, range: 10.0-100.0)
    fiber_intensity_min : float
        Minimum fiber intensity (default: 3.0)
    fiber_intensity_max : float
        Maximum fiber intensity (default: 8.0)
    distance_threshold : float
        Maximum distance for fiber rendering (default: 1.5, affects thickness)
    tissue_threshold : float
        Minimum tissue intensity for fiber rendering (default: 2.0)
    min_bundle_size : int, optional
        Minimum streamlines per bundle (filters small bundles)
    density_threshold : float, optional
        Density threshold for filtering sparse regions
    
    Notes
    -----
    Processing pipeline:
    - Applies configurable contrast enhancement to tissue
    - Renders streamlines as discrete points (no densification)
    - Uses z-flip transformation for correct alignment (z_plot = dims[2] - z - 1)
    - Fibers only drawn on tissue regions above tissue_threshold
    - Optional white matter mask filtering
    """
    # Load NIfTI
    nii_img = nib.load(nifti_file)
    nii_img = nib.as_closest_canonical(nii_img)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape
    
    print(f"Input NIfTI dimensions: {dims}")
    
    # Load white matter mask if provided and resample to match data (following 2D methodology)
    white_mask_data = None
    if white_mask_path and os.path.exists(white_mask_path):
        try:
            print(f"Loading white matter mask: {white_mask_path}")
            white_mask_img = nib.load(white_mask_path)
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
    
    # Create output directory for 2D images if needed
    if save_2d_images:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save 2D images to: {output_dir}")
    
    # Create output volume
    if orientation == 'coronal':
        volume_shape = (dims[0], num_slices, dims[2])
    elif orientation == 'axial':
        volume_shape = (dims[0], dims[1], num_slices)
    else:  # sagittal
        volume_shape = (num_slices, dims[1], dims[2])
    
    output_volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Select ONE random cornucopia preset for entire volume (consistency)
    if cornucopia_preset is not None:
        presets = ['extreme_noise', 'random_shapes_background', 
                  'comprehensive_aggressive', 'ultra_heavy_speckle']
        # Equal weights for selected presets
        weights = [0.25] * 4
        random.seed(int(time.time() * 1000000) % (2**32))
        selected_preset = random.choices(presets, weights=weights, k=1)[0]
        print(f"Cornucopia enabled: Using '{selected_preset}' for all slices (consistent volume appearance)")
    else:
        selected_preset = None
    
    # Process each slice
    for slice_offset, slice_idx in enumerate(tqdm(list(slice_range), desc="Processing slices")):
        # Save 2D visualization image if requested
        if save_2d_images and orientation == 'coronal':
            image_output = os.path.join(output_dir, f'slice_{slice_idx:04d}.png')
            visualize_nifti_with_trk_coronal(
                nifti_file=nifti_file,
                trk_file=trk_file,
                output_file=image_output,
                n_slices=1,
                slice_idx=slice_idx,
                output_image_size=(1024, 1024),
                save_masks=False
            )
        
        # Extract slice from original NIfTI
        if orientation == 'coronal':
            slice_data = nii_data[:, slice_idx, :]
            vol_slice = output_volume[:, slice_offset, :]
        elif orientation == 'axial':
            slice_data = nii_data[:, :, slice_idx]
            vol_slice = output_volume[:, :, slice_offset]
        else:  # sagittal
            slice_data = nii_data[slice_idx, :, :]
            vol_slice = output_volume[slice_offset, :, :]
        
        # Apply contrast enhancement with consistent cornucopia preset but varied granules per slice
        slice_random_state = slice_offset if selected_preset is not None else None
        slice_enhanced = apply_comprehensive_slice_processing(
            slice_data,
            background_preset=background_enhancement,
            cornucopia_preset=selected_preset,
            contrast_method=contrast_method,
            random_state=slice_random_state,
            debug=False
        )
        
        # Dark field effect with configurable gamma for tissue darkening
        intensity_params = {
            'gamma': gamma,
            'threshold': 0.04,
            'color_scheme': 'bw',
            'blue_tint': 0.2
        }
        dark_field_slice = apply_blockface_preserving_dark_field_effect(
            slice_enhanced,
            intensity_params=intensity_params,
            random_state=None,
            force_background_black=True
        )
        
        # Scale with configurable brightness
        dark_field_slice = dark_field_slice * scaling_factor
        vol_slice[:] = dark_field_slice
        
        # Render streamlines with distance-based intensity
        for sl in streamlines_voxel:
            
            if orientation == 'coronal':
                x, y, z = sl[:, 0], sl[:, 1], sl[:, 2]
                
                # Apply z-flip transformation for correct alignment
                for i in range(len(x)):
                    xi = int(np.round(np.clip(x[i], 0, vol_slice.shape[0]-1)))
                    z_plot = dims[2] - z[i] - 1  # Critical: matches core.py line 642
                    zi = int(np.round(np.clip(z_plot, 0, vol_slice.shape[1]-1)))
                    
                    # White matter mask filtering
                    if white_mask_data is not None:
                        xi_orig = int(np.round(np.clip(x[i], 0, dims[0]-1)))
                        yi_orig = int(np.round(np.clip(y[i], 0, dims[1]-1)))
                        zi_orig = int(np.round(np.clip(z[i], 0, dims[2]-1)))
                        if white_mask_data[xi_orig, yi_orig, zi_orig] < 0.5:
                            continue
                    
                    # Only draw on tissue (not background)
                    if vol_slice[xi, zi] < tissue_threshold:
                        continue
                    
                    # Constant fiber intensity (no distance-based variation)
                    fiber_intensity = (fiber_intensity_max + fiber_intensity_min) / 2.0
                    
                    # Additive blending
                    vol_slice[xi, zi] = min(255.0, vol_slice[xi, zi] + fiber_intensity)
            
            elif orientation == 'axial':
                x, y, z = sl[:, 0], sl[:, 1], sl[:, 2]
                
                for i in range(len(x)):
                    xi = int(np.round(np.clip(x[i], 0, vol_slice.shape[0]-1)))
                    yi = int(np.round(np.clip(y[i], 0, vol_slice.shape[1]-1)))
                    
                    # White matter mask filtering
                    if white_mask_data is not None:
                        xi_orig = int(np.round(np.clip(x[i], 0, dims[0]-1)))
                        yi_orig = int(np.round(np.clip(y[i], 0, dims[1]-1)))
                        zi_orig = int(np.round(np.clip(z[i], 0, dims[2]-1)))
                        if white_mask_data[xi_orig, yi_orig, zi_orig] < 0.5:
                            continue
                    
                    if vol_slice[xi, yi] < tissue_threshold:
                        continue
                    
                    # Constant fiber intensity (no distance-based variation)
                    fiber_intensity = (fiber_intensity_max + fiber_intensity_min) / 2.0
                    vol_slice[xi, yi] = min(255.0, vol_slice[xi, yi] + fiber_intensity)
            
            else:  # sagittal
                x, y, z = sl[:, 0], sl[:, 1], sl[:, 2]
                
                for i in range(len(y)):
                    yi = int(np.round(np.clip(y[i], 0, vol_slice.shape[0]-1)))
                    zi = int(np.round(np.clip(z[i], 0, vol_slice.shape[1]-1)))
                    
                    # White matter mask filtering
                    if white_mask_data is not None:
                        xi_orig = int(np.round(np.clip(x[i], 0, dims[0]-1)))
                        yi_orig = int(np.round(np.clip(y[i], 0, dims[1]-1)))
                        zi_orig = int(np.round(np.clip(z[i], 0, dims[2]-1)))
                        if white_mask_data[xi_orig, yi_orig, zi_orig] < 0.5:
                            continue
                    
                    if vol_slice[yi, zi] < tissue_threshold:
                        continue
                    
                    # Constant fiber intensity (no distance-based variation)
                    fiber_intensity = (fiber_intensity_max + fiber_intensity_min) / 2.0
                    vol_slice[yi, zi] = min(255.0, vol_slice[yi, zi] + fiber_intensity)
    
    # Save output
    output_nii = nib.Nifti1Image(output_volume, affine=nii_img.affine)
    nib.save(output_nii, output_file)
    print(f"\nSaved 3D volume: {output_file}")
    print(f"  Shape: {output_volume.shape}")
    print(f"  Value range: [{output_volume.min():.2f}, {output_volume.max():.2f}]")
    
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
    parser.add_argument('--save-2d', action='store_true', help='Save 2D visualization images for each slice')
    parser.add_argument('--output-dir', default='output_slices', help='Directory for 2D images')
    
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
        orientation=args.orientation,
        save_2d_images=args.save_2d,
        output_dir=args.output_dir
    )
